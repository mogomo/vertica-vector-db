// vbuild: builds a snapshot from consolidated vectors and returns it in chunks.
//   vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='docs', metric='cosine', max_ver=0) OVER()
// id INT, vec ARRAY[FLOAT] (ARRAY[INT] and ARRAY[NUMERIC] are accepted too), del BOOLEAN.
// No ORDER BY: the builder sorts by id itself. Rows with del = true are skipped in a full build.
// With base_snapshot the build is incremental: the rows are the changes since that snapshot (one
// per id: an add or change, or a delete with del = true), and the snapshot is read from the cache
// of the node that runs vbuild. Changes that change nothing give no rows.
// Parameters: index_name, metric (l2 | cosine | dot | l1), index_type (flat | hnsw), max_ver (the
// journal watermark: a parameter, not a column, because every input column costs transfer time
// per row), m (HNSW links per node, 16), ef_construction (200), threads (graph build threads,
// 0 = one per core), quantization (none | sq8), base_snapshot, cache_dir.
// Output (byte_offset, chunk, vector_count, dims, max_ver, format_version); vector_count counts
// the live vectors (tombstoned positions are not counted).
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
            if (type != "flat" && type != "hnsw") fail("index_type must be flat or hnsw, not '" + type + "'");
            if (quant != "none" && quant != "sq8") fail("quantization must be none or sq8, not '" + quant + "'");
            if (m < 2 || m > 256) fail("m must be 2 to 256");
            if (efc < 1 || efc > 100000) fail("ef_construction must be 1 to 100000");
            if (base < 0) fail("base_snapshot must be a snapshot id, or 0 for a full build");

            vvector::HnswParams hp;
            hp.m = static_cast<std::uint32_t>(m);
            hp.ef_construction = static_cast<std::uint32_t>(efc);
            hp.threads = threads;
            const vvector::HnswParams *graph = type == "hnsw" ? &hp : nullptr;
            const auto poll = [this] { return isCanceled(); };
            vvector::SnapshotBuffer buffer;
            vint rows = 0;
            if (base != 0) {
                const std::string node = srvInterface.getCurrentNodeName();
                vvector::MappedSnapshot from;
                try {
                    from.open(vvector::snapshot_path(resolve_cache_dir(srvInterface), name, base), true);
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
                    if (!builder.finish(max_ver, base, buffer, graph, poll)) return;     // nothing changed: no rows out
                } catch (const vvector::Cancelled &) {
                    return;
                }
            } else {
                vvector::SnapshotBuilder builder(vvector::parse_metric(metric));
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
            const vvector::VectorSet set = vvector::snapshot_open(buffer.data(), buffer.size(), false);

            const char *bytes = reinterpret_cast<const char *>(buffer.data());
            for (std::uint64_t off = 0; off < buffer.size(); off += vvector::CHUNK_BYTES) {
                const std::uint64_t len = std::min<std::uint64_t>(vvector::CHUNK_BYTES, buffer.size() - off);
                out.setInt(0, (vint)off);
                out.getStringRef(1).copy(bytes + off, len);
                out.setInt(2, (vint)(set.count - set.tombstones));
                out.setInt(3, (vint)set.dims);
                out.setInt(4, set.max_ver);
                out.setInt(5, vvector::FORMAT_VERSION);
                out.next();
                if (isCanceled()) return;
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
        for (int i = 0; i < 4; ++i) returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("byte_offset");
        outputTypes.addLongVarbinary((int32)vvector::CHUNK_BYTES, "chunk");
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
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VBuild>(srvInterface.allocator); }
};

RegisterFactory(VBuildFactory);
