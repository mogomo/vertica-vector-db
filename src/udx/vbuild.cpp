// vbuild: builds a snapshot from consolidated vectors and returns it in chunks.
//   vvector.vbuild(id, vec USING PARAMETERS index_name='docs', metric='cosine', max_ver=0) OVER(ORDER BY id)
// id INT, vec ARRAY[FLOAT] (cast ARRAY[INT] and ARRAY[NUMERIC] columns to ARRAY[FLOAT]).
// max_ver is the journal watermark. It is a parameter, not a column: every input column costs
// transfer time per row.
// index_type: 'flat' (default: the vectors only) or 'hnsw' (not implemented yet).
// Output (byte_offset, chunk, vector_count, dims, max_ver, format_version).
// Thin adapter around src/engine/snapshot.h.
#include "udx_common.h"
#include "../engine/hnsw.h"
#include "../engine/version.h"

#include <algorithm>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vbuild";

class VBuild : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(FN, srvInterface);
            const std::string metric = params.containsParameter("metric") ? params.getStringRef("metric").str() : "l2";
            const std::string type = params.containsParameter("index_type") ? params.getStringRef("index_type").str() : "flat";
            if (type != "flat" && type != "hnsw")
                vt_report_error(0, "%s: index_type must be flat or hnsw, not '%s'", FN, type.c_str());
            const vint max_ver = params.containsParameter("max_ver") ? params.getIntRef("max_ver") : 0;

            vvector::SnapshotBuilder builder(vvector::parse_metric(metric));
            std::vector<float> v;
            vint rows = 0;
            do {
                if (inputReader.isNull(0) || inputReader.isNull(1))
                    vt_report_error(0, "%s: index '%s': id and vec must not be NULL", FN, name.c_str());
                const vint id = inputReader.getIntRef(0);
                read_vector(inputReader, 1, v);
                builder.add(id, v.data(), static_cast<std::uint32_t>(v.size()));
                if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
            } while (inputReader.next());

            vvector::SnapshotBuffer buffer;
            builder.finish(max_ver, buffer);
            const vvector::VectorSet set = vvector::snapshot_open(buffer.data(), buffer.size(), false);
            if (type == "hnsw") vvector::hnsw_build(set, vvector::HnswParams());     // stub: throws until M2

            const char *bytes = reinterpret_cast<const char *>(buffer.data());
            for (std::uint64_t off = 0; off < buffer.size(); off += vvector::CHUNK_BYTES) {
                const std::uint64_t len = std::min<std::uint64_t>(vvector::CHUNK_BYTES, buffer.size() - off);
                outputWriter.setInt(0, (vint)off);
                outputWriter.getStringRef(1).copy(bytes + off, len);
                outputWriter.setInt(2, (vint)set.count);
                outputWriter.setInt(3, (vint)set.dims);
                outputWriter.setInt(4, set.max_ver);
                outputWriter.setInt(5, vvector::FORMAT_VERSION);
                outputWriter.next();
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
        parameterTypes.addVarchar(128, "index_name");
        parameterTypes.addVarchar(16, "metric");
        parameterTypes.addVarchar(16, "index_type");
        parameterTypes.addInt("max_ver");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VBuild>(srvInterface.allocator); }
};

RegisterFactory(VBuildFactory);
