// vscan: exact k-nearest-neighbour search over a table without an index (milestone M7).
//   SELECT id, score FROM (SELECT vvector.vscan(id, vec USING PARAMETERS query='[0.1, 0.2, ...]', k=10, metric='l2')
//                          OVER(PARTITION BEST) FROM app.docs) s ORDER BY score LIMIT 10;
// Input (id INT, vec ARRAY[FLOAT] | ARRAY[INT] | ARRAY[NUMERIC]). Vertica runs one instance per node
// and thread over the rows it holds (PARTITION BEST); every instance scores its rows with the
// vvector kernels (the same scores as vsearch and the built-ins; cosine rows are normalised on the
// fly) and returns its local k best per query as (qid, id, score). The SQL around it merges the
// instances: ORDER BY score LIMIT k for one query (ASC for l2 and l1, DESC for cosine and dot), or
// ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score ...) <= k for several. Rows with a NULL id or a
// NULL vector are skipped; a vector with the wrong number of elements is an error.
// Parameters: query (VARCHAR '[...]', qid 0) or queries (LONG VARCHAR, vectors separated by ';',
// qid 1, 2, ...; one of the two is required), k (10, at most 16384), metric (l2 | cosine | dot |
// l1), radius (only rows whose score is within it: l2, l1 score <= radius, cosine, dot >= radius),
// threads (1: Vertica supplies the parallelism; more for a call with OVER() on one instance).
// Memory per instance: the queries plus a block of 4096 rows; the table is read once, in blocks.
#include "udx_common.h"
#include "../engine/flat.h"
#include "../engine/kernels.h"
#include "../engine/text.h"

#include <cstring>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vscan";
static const std::uint64_t BLOCK_ROWS = 4096;

class VScan : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &in,
                                  PartitionWriter &out)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string metric_name = params.containsParameter("metric") ? params.getStringRef("metric").str() : "l2";
            if (!one_of(metric_name, {"l2", "cosine", "dot", "l1"})) fail("metric must be l2, cosine, dot or l1, not '" + metric_name + "'");
            const vvector::Metric metric = vvector::parse_metric(metric_name);
            const vint k = params.containsParameter("k") ? params.getIntRef("k") : 10;
            if (k < 1 || k > 16384) fail("k must be 1 to 16384, not " + std::to_string(k));
            const vint threads = params.containsParameter("threads") ? params.getIntRef("threads") : 1;
            if (threads < 0 || threads > 64) fail("threads must be 0 (one per core) to 64");
            const bool cosine = metric == vvector::Metric::Cosine;

            // The queries: one in `query` (qid 0), or several in `queries` (qid = position from 1).
            std::vector<std::vector<float>> qv;
            std::vector<vint> qids;
            if (params.containsParameter("query")) {
                const VString &text = params.getStringRef("query");
                qv.push_back(vvector::parse_vector_text(text.data(), text.length()));
                qids.push_back(0);
            }
            if (params.containsParameter("queries")) {
                const VString &text = params.getStringRef("queries");
                const char *p = text.data(), *end = p + text.length();
                while (p < end) {
                    const char *semi = static_cast<const char *>(std::memchr(p, ';', static_cast<std::size_t>(end - p)));
                    const char *stop = semi ? semi : end;
                    bool blank = true;
                    for (const char *c = p; c < stop; ++c) blank &= (*c == ' ' || *c == '\n' || *c == '\t' || *c == '\r');
                    if (!blank) {
                        qv.push_back(vvector::parse_vector_text(p, static_cast<std::size_t>(stop - p)));
                        qids.push_back(static_cast<vint>(qids.size()) + 1);
                    }
                    p = stop + 1;
                }
            }
            if (qv.empty()) fail("a query is required: the parameter query='[...]' or queries='[...];[...]'");
            const std::uint32_t dims = static_cast<std::uint32_t>(qv[0].size());
            if (dims == 0) fail("the query has no elements");
            for (std::size_t i = 1; i < qv.size(); ++i)
                if (qv[i].size() != dims)
                    fail("query " + std::to_string(i + 1) + " has " + std::to_string(qv[i].size()) + " elements, the first has " + std::to_string(dims));
            const std::uint32_t stride = vvector::row_stride_for(dims);
            std::vector<float> queries(qv.size() * stride, 0.0f);
            for (std::size_t i = 0; i < qv.size(); ++i) {
                std::memcpy(queries.data() + i * stride, qv[i].data(), dims * sizeof(float));
                if (cosine) vvector::normalize(queries.data() + i * stride, dims);
            }
            std::vector<std::vector<float>>().swap(qv);

            vvector::FlatSearch fs;
            fs.metric = metric;
            fs.stride = stride;
            fs.queries = queries.data();
            fs.n_queries = qids.size();
            fs.k = static_cast<std::uint32_t>(k);
            fs.threads = vvector::resolve_threads(static_cast<int>(threads));
            if (params.containsParameter("radius")) {
                fs.has_radius = true;
                fs.radius = params.getFloatRef("radius");
            }

            // The rows, in blocks: each block is searched exactly and merged into the running result.
            std::vector<float> rows(BLOCK_ROWS * stride, 0.0f);
            std::vector<std::int64_t> ids(BLOCK_ROWS);
            std::vector<vvector::Neighbor> found(qids.size() * fs.k, vvector::Neighbor{0, 0});
            std::vector<std::uint32_t> count(qids.size(), 0);
            const auto poll = [this] { return isCanceled(); };
            auto flush = [&](std::uint64_t n) {
                if (n == 0) return;
                vvector::RowBlock block;
                block.rows = rows.data();
                block.ids = ids.data();
                block.n = n;
                vvector::merge_block(fs, &block, found, count, poll);
            };
            std::uint64_t n = 0;
            vint seen = 0;
            do {
                if ((++seen & 0xFF) == 0 && isCanceled()) return;
                if (in.isNull(0) || in.isNull(1)) continue;
                float *row = rows.data() + n * stride;
                std::fill(row, row + stride, 0.0f);
                read_floats(in, 1, [&](std::uint32_t got) {
                    if (got != dims) fail("the query has " + std::to_string(dims) + " elements, the vector of id " + std::to_string(in.getIntRef(0)) + " has " + std::to_string(got));
                    return row;
                }, "the vector of id " + std::to_string(in.getIntRef(0)));
                if (cosine) vvector::normalize(row, dims);
                ids[n] = in.getIntRef(0);
                if (++n == BLOCK_ROWS) {
                    flush(n);
                    n = 0;
                }
            } while (in.next());
            flush(n);

            for (std::size_t q = 0; q < qids.size(); ++q)
                for (std::uint32_t i = 0; i < count[q]; ++i) {
                    const vvector::Neighbor &nb = found[q * fs.k + i];
                    out.setInt(0, qids[q]);
                    out.setInt(1, nb.id);
                    out.setFloat(2, vvector::key_to_score(metric, nb.key));
                    out.next();
                }
        } catch (const vvector::Cancelled &) {
            return;
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VScanFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        argTypes.addArrayType(Float8OID);
        returnType.addInt();
        returnType.addInt();
        returnType.addFloat();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("qid");
        outputTypes.addInt("id");
        outputTypes.addFloat("score");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        parameterTypes.addVarchar(65000, "query");
        parameterTypes.addLongVarchar(32000000, "queries");
        parameterTypes.addInt("k");
        parameterTypes.addVarchar(16, "metric");
        parameterTypes.addFloat("radius");
        parameterTypes.addInt("threads");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VScan>(srvInterface.allocator); }
};

RegisterFactory(VScanFactory);
