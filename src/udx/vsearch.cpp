// vsearch: the k nearest neighbours of every query vector.
//   SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
//                          USING PARAMETERS index_name='docs', query='[0.1, 0.2, 0.3]', k=10) OVER()
//   FROM app.docs_snap;
// Input rows and their roles: see udx_common.h. Output (qid, id, score, rank): score is what the
// built-in function of the index metric returns (VECTOR_L2, COSINE_SIMILARITY, DOT_PRODUCT; l1: the
// Manhattan distance); rank 1 is the closest; ties by id ascending.
// Parameters: index_name, k, precision, freshness, ef_search, exact, radius, threads, query,
// rescore, oversampling, filtered, cache_dir.
// Filtered search: allow-list rows (id set, vec and del NULL) limit the results to their ids, the
// journal's rows included. filtered=true limits them also when no allow-list row arrives (a filter
// that matched nothing returns nothing, not the unfiltered result).
// Tuning values: function parameter, then session parameter, then index default
// (set_index_options), then built-in default.
// On an HNSW index the precision levels are presets of ef_search (fast: max(2 x k, 32), balanced,
// the default: 100, best: 400); precision exact or exact=true search every vector (flat).
// Thin adapter: the search is src/engine/flat.h and src/engine/hnsw.h.
#include "udx_common.h"
#include "../engine/flat.h"
#include "../engine/hnsw.h"
#include "../engine/kernels.h"
#include "../engine/parallel.h"
#include "../engine/text.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vsearch";

namespace {

// Journal rows of the input: the latest row of every id by version; with equal versions a delete
// wins. Vectors of adds are kept in contiguous rows of the index's stride.
class Journal {
public:
    explicit Journal(std::uint32_t stride) : stride_(stride) {}

    // Offers a row. vector(dst) is called only when the row is the newest of its id so far; it
    // fills dst (stride floats, zero padded) with the vector of an add.
    template <class ReadVector>
    void offer(vint id, vint ver, bool del, ReadVector vector)
    {
        const auto found = entries_.find(id);
        const bool is_new = found == entries_.end();
        if (!is_new) {
            const Entry &e = found->second;
            if (!(ver > e.ver || (ver == e.ver && del && !e.del))) return;
        }
        Entry &e = entries_[id];
        if (is_new) e.slot = NO_SLOT;
        e.ver = ver;
        e.del = del;
        if (del) {
            if (e.slot != NO_SLOT) { free_.push_back(e.slot); e.slot = NO_SLOT; }
            return;
        }
        if (e.slot == NO_SLOT) {
            if (!free_.empty()) { e.slot = free_.back(); free_.pop_back(); }
            else { e.slot = static_cast<std::uint32_t>(rows_.size() / stride_); rows_.resize(rows_.size() + stride_); }
        }
        float *dst = rows_.data() + std::uint64_t(e.slot) * stride_;
        std::memset(dst, 0, stride_ * sizeof(float));
        vector(dst);
    }

    bool empty() const { return entries_.empty(); }

    // Marks every journal id that is in the snapshot, and copies the live vectors out; with an
    // allow-list (sorted ids, may be null) only those of allowed ids.
    void apply(const vvector::VectorSet &set, std::vector<std::uint64_t> &skip, std::vector<std::int64_t> &ids,
               std::vector<float> &rows, const std::vector<std::int64_t> *allow) const
    {
        for (const auto &x : entries_) {
            const std::int64_t pos = set.find(x.first);
            if (pos >= 0) skip[pos >> 6] |= 1ull << (pos & 63);
            if (!x.second.del && (!allow || std::binary_search(allow->begin(), allow->end(), x.first))) {
                ids.push_back(x.first);
                const float *src = rows_.data() + std::uint64_t(x.second.slot) * stride_;
                rows.insert(rows.end(), src, src + stride_);
            }
        }
    }

private:
    static const std::uint32_t NO_SLOT = 0xFFFFFFFFu;
    struct Entry { vint ver; bool del; std::uint32_t slot; };
    std::uint32_t stride_;
    std::unordered_map<vint, Entry> entries_;
    std::vector<float> rows_;
    std::vector<std::uint32_t> free_;
};

} // namespace

class VSearch : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &in,
                                  PartitionWriter &out)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            const std::string cache_dir = resolve_cache_dir(srvInterface);
            vvector::MappedSnapshot snap;
            snap.open_active(cache_dir, name);

            // Tuning values (udx_common.h), and freshness.
            Settings cfg(srvInterface, snap.options());
            const SearchSettings ss(cfg, params);
            const std::string freshness = cfg.text("freshness", "snapshot");
            if (!one_of(freshness, {"snapshot", "exact"})) fail("freshness must be snapshot or exact, not '" + freshness + "'");
            const bool apply_journal = freshness == "exact";

            const vvector::VectorSet set = snap.vectors();
            const std::uint32_t dims = set.dims, stride = set.row_stride;
            const bool cosine = set.metric == vvector::Metric::Cosine;
            auto check_dims = [&](std::uint32_t n, const std::string &what) {
                if (n != dims)
                    fail("index '" + name + "' has " + std::to_string(dims) + " dimensions, " + what + " has " + std::to_string(n));
            };

            // Queries, contiguous, zero padded; the query parameter is qid 0.
            std::vector<vint> qids;
            std::vector<float> queries;
            if (params.containsParameter("query")) {
                const VString &text = params.getStringRef("query");
                const std::vector<float> v = vvector::parse_vector_text(text.data(), text.length());
                check_dims(static_cast<std::uint32_t>(v.size()), "the query parameter");
                qids.push_back(0);
                queries.resize(stride, 0.0f);
                std::memcpy(queries.data(), v.data(), v.size() * sizeof(float));
                if (cosine) vvector::normalize(queries.data(), dims);
            }

            Journal journal(stride);
            std::vector<std::int64_t> allow_ids;
            vint wanted_snapshot = -1, rows = 0;
            do {
                if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
                if (!in.isNull(COL_SNAPSHOT_ID) && in.getIntRef(COL_SNAPSHOT_ID) > wanted_snapshot)
                    wanted_snapshot = in.getIntRef(COL_SNAPSHOT_ID);
                const bool has_qid = !in.isNull(COL_QID), has_qvec = !in.isNull(COL_QVEC);
                if (has_qid || has_qvec) {
                    if (!has_qid) fail("a query row has a vector (qvec) but no qid");
                    const vint qid = in.getIntRef(COL_QID);
                    const std::string what = "query " + std::to_string(qid);
                    if (!has_qvec) fail(what + " has no vector (qvec is NULL)");
                    const std::size_t at = queries.size();
                    read_floats(in, COL_QVEC, [&](std::uint32_t n) {
                        check_dims(n, what);
                        queries.resize(at + stride, 0.0f);
                        return queries.data() + at;
                    }, what);
                    if (cosine) vvector::normalize(queries.data() + at, dims);
                    qids.push_back(qid);
                    continue;
                }
                if (in.isNull(COL_ID)) continue;                     // the sentinel
                const vint id = in.getIntRef(COL_ID);
                const bool has_vec = !in.isNull(COL_VEC), has_del = !in.isNull(COL_DEL);
                if (!has_vec && !has_del) { allow_ids.push_back(id); continue; }  // allow-list member
                const bool del = has_del && in.getBoolRef(COL_DEL) == vbool_true;
                if (!del && !has_vec) fail("journal row with id " + std::to_string(id) + " has no vector and is not a delete");
                if (!apply_journal) continue;
                const vint ver = in.isNull(COL_VER) ? vint_null : in.getIntRef(COL_VER);   // NULL: older than any version
                journal.offer(id, ver, del, [&](float *dst) {
                    const std::string what = "the journal vector of id " + std::to_string(id);
                    read_floats(in, COL_VEC, [&](std::uint32_t n) { check_dims(n, what); return dst; }, what);
                    if (cosine) vvector::normalize(dst, dims);
                });
            } while (in.next());

            // The view belongs to a newer snapshot than the one this process holds: read ACTIVE again.
            if (snap.snapshot_id() < wanted_snapshot) {
                snap.open_active(cache_dir, name, wanted_snapshot);
                if (snap.snapshot_id() < wanted_snapshot) fail("snapshot cache stale on " + srvInterface.getCurrentNodeName() + ": run vload");
                if (snap.vectors().dims != dims) fail("the snapshot of index '" + name + "' changed while the query ran: run it again");
            }
            if (qids.empty()) fail("no query: give the query parameter, or query rows (qid, qvec)");
            const vvector::VectorSet &now = snap.vectors();

            // The allow-list: sorted ids, and their positions in the snapshot.
            const bool filtered = !allow_ids.empty() || (params.containsParameter("filtered") && params.getBoolRef("filtered") == vbool_true);
            std::vector<std::uint32_t> allow_pos;
            if (filtered) {
                std::sort(allow_ids.begin(), allow_ids.end());
                allow_ids.erase(std::unique(allow_ids.begin(), allow_ids.end()), allow_ids.end());
                std::vector<std::int64_t> found(allow_ids.size());
                now.find_sorted(allow_ids.data(), allow_ids.size(), found.data());
                allow_pos.reserve(allow_ids.size());
                for (const std::int64_t pos : found)
                    if (pos >= 0) allow_pos.push_back(static_cast<std::uint32_t>(pos));
            }

            // The snapshot without the ids the journal changed or deleted, and the journal's live vectors.
            std::vector<std::uint64_t> skip;
            std::vector<std::int64_t> extra_ids;
            std::vector<float> extra_rows;
            if (!journal.empty() || now.tombstone_bits) {
                skip.assign((now.count + 63) / 64, 0);
                if (now.tombstone_bits) std::memcpy(skip.data(), now.tombstone_bits, skip.size() * 8);
                journal.apply(now, skip, extra_ids, extra_rows, filtered ? &allow_ids : nullptr);
            }
            const vvector::RowBlock extra{extra_rows.data(), extra_ids.data(), extra_ids.size(), nullptr};
            const vvector::FlatSearch fs = ss.describe(now, queries.data(), qids.size());
            std::vector<vvector::Neighbor> found;
            std::vector<std::uint32_t> count;
            try {
                ss.search(fs, now, skip.empty() ? nullptr : skip.data(), &extra, found, count, [this] { return isCanceled(); },
                          filtered ? &allow_pos : nullptr);
            } catch (const vvector::Cancelled &) {
                return;
            }

            for (std::size_t q = 0; q < qids.size(); ++q)
                for (std::uint32_t i = 0; i < count[q]; ++i) {
                    const vvector::Neighbor &nb = found[q * fs.k + i];
                    out.setInt(0, qids[q]);
                    out.setInt(1, nb.id);
                    out.setFloat(2, vvector::key_to_score(fs.metric, nb.key));
                    out.setInt(3, static_cast<vint>(i) + 1);
                    out.next();
                }
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VSearchFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        add_query_input(argTypes);
        returnType.addInt();
        returnType.addInt();
        returnType.addFloat();
        returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("qid");
        outputTypes.addInt("id");
        outputTypes.addFloat("score");
        outputTypes.addInt("rank");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        add_search_parameters(parameterTypes);
        parameterTypes.addVarchar(16, "freshness");
        parameterTypes.addBool("filtered");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VSearch>(srvInterface.allocator); }
};

RegisterFactory(VSearchFactory);
