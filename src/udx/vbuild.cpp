// vbuild: builds a snapshot from consolidated vectors and returns it in chunks.
//   vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='docs', metric='cosine', max_ver=0) OVER()
// id INT, vec ARRAY[FLOAT] (ARRAY[INT] and ARRAY[NUMERIC] are accepted too), del BOOLEAN.
// No ORDER BY: the builder sorts by id itself. Rows with del = true are skipped in a full build.
// With base_snapshot the build is incremental: the rows are the changes since that snapshot (one
// per id: an add or change, or a delete with del = true), and the snapshot is read from the cache
// of the node that runs vbuild. Changes that change nothing give no rows. The build runs in place
// on a copy-on-write mapping of the base when its layout has room (milestone M7, delta.h), and
// then, with send = patch, only the changed bytes are returned: rows of (byte_offset, chunk) that
// hold up to 8 MB of packed runs of changed bytes (delta.h pack_runs; byte_offset = the first run's),
// with base_snapshot set, which vload unpacks and writes over its copy of the base.
// send = whole, a base without room, or a full build: whole chunks of 8 MB, base_snapshot NULL,
// all-zero chunks (the room to grow) left out.
// Parameters: index_name, metric (l2 | cosine | dot | l1), index_type (flat | hnsw), max_ver (the
// journal watermark: a parameter, not a column, because every input column costs transfer time
// per row), m (HNSW links per node, 16), ef_construction (200), threads (graph build threads,
// 0 = one per core), quantization (none | sq8), base_snapshot, cache_dir, build_in (ram | file:
// build in an unlinked file in <cache_dir>/<index_name> instead of anonymous memory, so the kernel
// can write the snapshot out and reclaim its pages; for builds larger than the free memory),
// growth (room to grow in the layout, percent of the count, default 5, 0 = none; milestone M7),
// send (patch | whole, default patch: what an incremental build returns, see above), reachability
// (auto | on | off, milestone M7: whether an HNSW build counts the live vectors no search can reach
// and stores the count in the graph header; auto = every full build and incremental builds below
// 8M positions).
// Output (byte_offset, chunk, base_snapshot, vector_count, dims, max_ver, format_version);
// vector_count counts the live vectors (tombstoned positions are not counted).
// Thin adapter around src/engine/snapshot.h, src/engine/delta.h and src/engine/hnsw.h.
#include "udx_common.h"
#include "../engine/delta.h"
#include "../engine/hnsw.h"
#include "../engine/parallel.h"
#include "../engine/version.h"

#include <algorithm>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vbuild";

class VBuild : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &in,
                                  PartitionWriter &out)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            const std::string metric = params.containsParameter("metric") ? params.getStringRef("metric").str() : "l2";
            const std::string type = params.containsParameter("index_type") ? params.getStringRef("index_type").str() : "flat";
            const std::string quant = params.containsParameter("quantization") ? params.getStringRef("quantization").str() : "none";
            const vint max_ver = params.containsParameter("max_ver") ? params.getIntRef("max_ver") : 0;
            const vint m = params.containsParameter("m") ? params.getIntRef("m") : 16;
            const vint efc = params.containsParameter("ef_construction") ? params.getIntRef("ef_construction") : 200;
            const vint base = params.containsParameter("base_snapshot") ? params.getIntRef("base_snapshot") : 0;
            const int threads = vvector::resolve_threads(params.containsParameter("threads") ? params.getIntRef("threads") : 0);
            const std::string build_in = params.containsParameter("build_in") ? params.getStringRef("build_in").str() : "ram";
            const vint growth = params.containsParameter("growth") ? params.getIntRef("growth") : vvector::DEFAULT_GROWTH_PERCENT;
            if (growth < 0 || growth > vvector::MAX_GROWTH_PERCENT) fail("growth must be 0 to " + std::to_string(vvector::MAX_GROWTH_PERCENT) + " percent");
            const std::string send = params.containsParameter("send") ? params.getStringRef("send").str() : "patch";
            if (send != "patch" && send != "whole") fail("send must be patch or whole, not '" + send + "'");
            const std::string reach = params.containsParameter("reachability") ? params.getStringRef("reachability").str() : "auto";
            if (reach != "auto" && reach != "on" && reach != "off") fail("reachability must be auto, on or off, not '" + reach + "'");
            if (build_in != "ram" && build_in != "file") fail("build_in must be ram or file, not '" + build_in + "'");
            const std::string build_dir = build_in == "file" ? vvector::ensure_index_dir(resolve_cache_dir(srvInterface), name) : "";
            if (type != "flat" && type != "hnsw") fail("index_type must be flat or hnsw, not '" + type + "'");
            if (quant != "none" && quant != "sq8") fail("quantization must be none or sq8, not '" + quant + "'");
            if (m < 2 || m > 256) fail("m must be 2 to 256");
            if (efc < 1 || efc > 100000) fail("ef_construction must be 1 to 100000");
            if (base < 0) fail("base_snapshot must be a snapshot id, or 0 for a full build");

            vvector::HnswParams hp;
            hp.m = static_cast<std::uint32_t>(m);
            hp.ef_construction = static_cast<std::uint32_t>(efc);
            hp.threads = threads;
            hp.reachability = reach == "on" ? vvector::Reachability::On : reach == "off" ? vvector::Reachability::Off : vvector::Reachability::Auto;
            const vvector::HnswParams *graph = type == "hnsw" ? &hp : nullptr;
            const auto poll = [this] { return isCanceled(); };
            vvector::SnapshotBuffer buffer;
            if (!build_dir.empty()) buffer.back_with_file(build_dir);
            vint rows = 0;
            // What goes out: whole chunks of `buffer` (or of `copy`), or the runs of `copy`.
            vvector::MappedSnapshot from, copy;
            std::vector<vvector::ByteRange> runs;
            const std::uint8_t *out_bytes = nullptr;
            std::uint64_t out_size = 0;
            if (base != 0) {
                const std::string node = srvInterface.getCurrentNodeName();
                const std::string path = vvector::snapshot_path(resolve_cache_dir(srvInterface), name, base);
                try {
                    from.open(path, true);
                    copy.open(path, false, false, false, true);
                } catch (const std::runtime_error &e) {
                    fail("index '" + name + "': base snapshot " + std::to_string(base) + " is not usable in the cache of " + node +
                         " (" + e.what() + "): refresh with mode full");
                }
                const vvector::VectorSet &bs = from.vectors();
                if (bs.metric != vvector::parse_metric(metric))
                    fail("index '" + name + "': the base snapshot has metric " + vvector::metric_name(bs.metric) + ", not " + metric +
                         ": refresh with mode full");
                if (((bs.flags & vvector::FLAG_SQ8) != 0) != (quant == "sq8"))
                    fail("index '" + name + "': the base snapshot has quantization " + ((bs.flags & vvector::FLAG_SQ8) ? "sq8" : "none") +
                         ", not " + quant + ": refresh with mode full");
                vvector::IncrementalBuilder builder(bs);
                builder.set_growth(static_cast<std::uint32_t>(growth));
                do {
                    if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
                    if (in.isNull(0)) fail("index '" + name + "': id must not be NULL");
                    const vint id = in.getIntRef(0);
                    if (!in.isNull(2) && in.getBoolRef(2) == vbool_true) { builder.remove(id); continue; }
                    if (in.isNull(1)) fail("index '" + name + "': the vector of id " + std::to_string(id) + " is NULL (a delete needs del = true)");
                    read_floats(in, 1, [&](std::uint32_t n) { return builder.begin_add(id, n); },
                                "the vector of id " + std::to_string(id));
                    builder.end_add();
                } while (in.next());
                try {
                    // In place on the copy-on-write mapping when the base's layout has room; the
                    // changed bytes are found against the read-only mapping and the checksum comes
                    // from them. Else the copying build lays the snapshot out anew.
                    std::vector<vvector::ByteRange> candidates;
                    const vvector::InPlace r = builder.finish_in_place(max_ver, base, copy.writable(), copy.size(), graph, candidates, poll);
                    if (r == vvector::InPlace::Unchanged) return;           // nothing changed: no rows out
                    if (r == vvector::InPlace::Built) {
                        runs = vvector::snapshot_diff(from.data(), copy.data(), copy.size(), candidates, vvector::CHUNK_BYTES);
                        vvector::seal_in_place(from.data(), copy.writable(), copy.size(), runs);
                        out_bytes = copy.data();
                        out_size = copy.size();
                        if (send == "whole") runs.clear();
                    } else if (!builder.finish(max_ver, base, buffer, graph, poll)) {
                        return;
                    }
                } catch (const vvector::Cancelled &) {
                    return;
                }
            } else {
                vvector::SnapshotBuilder builder(vvector::parse_metric(metric));
                builder.set_growth(static_cast<std::uint32_t>(growth));
                if (!build_dir.empty()) builder.build_in_file(build_dir);
                do {
                    if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
                    if (in.isNull(0)) fail("index '" + name + "': id must not be NULL");
                    const vint id = in.getIntRef(0);
                    if (!in.isNull(2) && in.getBoolRef(2) == vbool_true) continue;
                    if (in.isNull(1)) fail("index '" + name + "': the vector of id " + std::to_string(id) + " is NULL (a delete needs del = true)");
                    read_floats(in, 1, [&](std::uint32_t n) { return builder.begin_row(id, n); },
                                "the vector of id " + std::to_string(id));
                    builder.end_row();
                } while (in.next());
                if (builder.count() == 0) return;          // nothing to build: no rows out
                try {
                    const vvector::CodeSection codes = vvector::sq8_code_section();
                    const vvector::CodeSection *with_codes = quant == "sq8" ? &codes : nullptr;
                    if (graph) {
                        const vvector::GraphSection g = vvector::hnsw_graph_section(hp, poll);
                        builder.finish(max_ver, buffer, &g, with_codes);
                    } else {
                        builder.finish(max_ver, buffer, nullptr, with_codes);
                    }
                } catch (const vvector::Cancelled &) {
                    return;
                }
            }
            if (!out_bytes) {
                out_bytes = buffer.data();
                out_size = buffer.size();
            }
            const vvector::VectorSet set = vvector::snapshot_open(out_bytes, out_size, false);
            const char *bytes = reinterpret_cast<const char *>(out_bytes);
            auto emit = [&](std::uint64_t off, const char *data, std::uint64_t len, bool patch) {
                out.setInt(0, (vint)off);
                out.getStringRef(1).copy(data, len);
                if (patch) out.setInt(2, base); else out.setNull(2);
                out.setInt(3, (vint)(set.count - set.tombstones));
                out.setInt(4, (vint)set.dims);
                out.setInt(5, set.max_ver);
                out.setInt(6, vvector::FORMAT_VERSION);
                out.next();
            };
            if (!runs.empty()) {
                // The runs packed into rows of up to 8 MB (delta.h pack_runs): Vertica's planner
                // reserves the declared 8 MB for every row of the load's broadcast join, so one row
                // per run would ask for gigabytes for a few MB of patch (session 18).
                try {
                    vvector::pack_runs(out_bytes, runs, vvector::CHUNK_BYTES, [&](std::uint64_t first, const std::uint8_t *row, std::uint64_t len) {
                        emit(first, reinterpret_cast<const char *>(row), len, true);
                        if (isCanceled()) throw vvector::Cancelled();
                    });
                } catch (const vvector::Cancelled &) {
                    return;
                }
            } else {
                // Whole chunks; the all-zero ones (the room to grow) stay home: vload sizes the file
                // from the header and a hole reads as zeros. Chunk 0 always goes (the header).
                for (std::uint64_t off = 0; off < out_size; off += vvector::CHUNK_BYTES) {
                    const std::uint64_t len = std::min<std::uint64_t>(vvector::CHUNK_BYTES, out_size - off);
                    if (off > 0 && vvector::all_zero(out_bytes + off, len)) continue;
                    emit(off, bytes + off, len, false);
                    if (isCanceled()) return;
                }
            }
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VBuildFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        argTypes.addArrayType(Float8OID);
        argTypes.addBool();
        returnType.addInt();
        returnType.addLongVarbinary();
        for (int i = 0; i < 5; ++i) returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("byte_offset");
        outputTypes.addLongVarbinary((int32)vvector::CHUNK_BYTES, "chunk");
        outputTypes.addInt("base_snapshot");
        outputTypes.addInt("vector_count");
        outputTypes.addInt("dims");
        outputTypes.addInt("max_ver");
        outputTypes.addInt("format_version");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        parameterTypes.addVarchar(16, "metric");
        parameterTypes.addVarchar(16, "index_type");
        parameterTypes.addInt("max_ver");
        parameterTypes.addInt("m");
        parameterTypes.addInt("ef_construction");
        parameterTypes.addInt("threads");
        parameterTypes.addVarchar(16, "quantization");
        parameterTypes.addInt("base_snapshot");
        parameterTypes.addVarchar(16, "build_in");
        parameterTypes.addInt("growth");
        parameterTypes.addVarchar(16, "send");
        parameterTypes.addVarchar(16, "reachability");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VBuild>(srvInterface.allocator); }
};

RegisterFactory(VBuildFactory);
