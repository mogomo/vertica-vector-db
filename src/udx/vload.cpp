// vload: writes the snapshot cache file on every node and makes it active.
//   vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='docs', snapshot_id=7) OVER(PARTITION NODES)
// Output (node_name, snapshot_id, bytes, status). Idempotent: run it again any time.
// Thin adapter around src/engine/cache.h.
#include "udx_common.h"

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vload";

class VLoad : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        const std::string node = srvInterface.getCurrentNodeName();
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            if (!params.containsParameter("snapshot_id")) fail("parameter snapshot_id is required");
            const vint snapshot_id = params.getIntRef("snapshot_id");

            vvector::CacheWriter writer;
            writer.begin(resolve_cache_dir(srvInterface), name, snapshot_id);
            do {
                if (inputReader.isNull(0) || inputReader.getStringRef(1).isNull())
                    fail("index '" + name + "': NULL byte_offset or chunk");
                const VString &chunk = inputReader.getStringRef(1);
                writer.write_at(inputReader.getIntRef(0), chunk.data(), chunk.length());
                if (isCanceled()) return;
            } while (inputReader.next());
            const std::uint64_t bytes = writer.commit();

            outputWriter.getStringRef(0).copy(node);
            outputWriter.setInt(1, snapshot_id);
            outputWriter.setInt(2, (vint)bytes);
            outputWriter.getStringRef(3).copy("loaded");
            outputWriter.next();
        } catch (std::exception &e) {
            vt_report_error(0, "%s: on %s: %s", FN, node.c_str(), e.what());
        }
    }
};

class VLoadFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        argTypes.addLongVarbinary();
        returnType.addVarchar();
        returnType.addInt();
        returnType.addInt();
        returnType.addVarchar();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(128, "node_name");
        outputTypes.addInt("snapshot_id");
        outputTypes.addInt("bytes");
        outputTypes.addVarchar(32, "status");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        parameterTypes.addInt("snapshot_id");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VLoad>(srvInterface.allocator); }
};

RegisterFactory(VLoadFactory);
