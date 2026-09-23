// vbuild: builds a snapshot from consolidated vectors and returns it in chunks.
//   vvector.vbuild(id, vec, del USING PARAMETERS index_name='docs', metric='cosine', max_ver=0) OVER()
// id INT, vec ARRAY[FLOAT] (ARRAY[INT] and ARRAY[NUMERIC] are accepted too), del BOOLEAN.
// No ORDER BY: the builder sorts by id itself. Rows with del = true are skipped in a full build
// (they matter for incremental builds, milestone M3).
// Parameters: index_name, metric (l2 | cosine | dot | l1), index_type (flat | hnsw), max_ver (the
// journal watermark: a parameter, not a column, because every input column costs transfer time
// per row), m (HNSW links per node, 16), ef_construction (200), threads (graph build threads,
// 0 = one per core), quantization (none | sq8), base_snapshot, cache_dir.
// Output (byte_offset, chunk, vector_count, dims, max_ver, format_version).
// Thin adapter around src/engine/snapshot.h and src/engine/hnsw.h.
#include "udx_common.h"
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
            if (quant == "sq8") fail("quantization sq8 is not implemented yet (milestone M4)");
            if (base != 0) fail("base_snapshot: incremental builds are not implemented yet (milestone M3)");

            vvector::SnapshotBuilder builder(vvector::parse_metric(metric));
            vint rows = 0, skipped = 0;
            do {
                if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
                if (in.isNull(0)) fail("index '" + name + "': id must not be NULL");
                const vint id = in.getIntRef(0);
                if (!in.isNull(2) && in.getBoolRef(2) == vbool_true) { ++skipped; continue; }
                if (in.isNull(1)) fail("index '" + name + "': the vector of id " + std::to_string(id) + " is NULL (a delete needs del = true)");
                read_floats(in, 1, [&](std::uint32_t n) { return builder.begin_row(id, n); },
                            "the vector of id " + std::to_string(id));
                builder.end_row();
            } while (in.next());
            if (builder.count() == 0) return;          // nothing to build: no rows out

            vvector::SnapshotBuffer buffer;
            if (type == "hnsw") {
                vvector::HnswParams hp;
                hp.m = static_cast<std::uint32_t>(m);
                hp.ef_construction = static_cast<std::uint32_t>(efc);
                hp.threads = threads;
                const vvector::GraphSection graph = vvector::hnsw_graph_section(hp, [this] { return isCanceled(); });
                try {
                    builder.finish(max_ver, buffer, &graph);
                } catch (const vvector::Cancelled &) {
                    return;
                }
            } else {
                builder.finish(max_ver, buffer);
            }
            const vvector::VectorSet set = vvector::snapshot_open(buffer.data(), buffer.size(), false);

            const char *bytes = reinterpret_cast<const char *>(buffer.data());
            for (std::uint64_t off = 0; off < buffer.size(); off += vvector::CHUNK_BYTES) {
                const std::uint64_t len = std::min<std::uint64_t>(vvector::CHUNK_BYTES, buffer.size() - off);
                out.setInt(0, (vint)off);
                out.getStringRef(1).copy(bytes + off, len);
                out.setInt(2, (vint)set.count);
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
