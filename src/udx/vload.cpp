// vload: writes the snapshot cache file on every node and makes it active.
//   vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='docs', snapshot_id=7) OVER(PARTITION NODES)
// Output (node_name, snapshot_id, bytes, status). Idempotent: run it again any time.
// A large snapshot is loaded in passes (vvector.load_on_nodes: a broadcast join holds its inner in
// memory): parameters part (letters and digits naming the load), pass and passes; pass 1 creates a
// partial file, every pass writes its pieces into it, the last one verifies and activates it
// (status 'loaded'), the others return status 'partial'.
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

            const bool in_parts = params.containsParameter("part");
            vint pass = 1, passes = 1;
            if (in_parts) {
                if (!params.containsParameter("pass") || !params.containsParameter("passes"))
                    fail("parameter part needs pass and passes");
                pass = params.getIntRef("pass");
                passes = params.getIntRef("passes");
                if (passes < 1 || pass < 1 || pass > passes) fail("pass must be 1 to passes");
            }

            vvector::CacheWriter writer;
            if (in_parts)
                writer.begin_part(resolve_cache_dir(srvInterface), name, snapshot_id,
                                  params.getStringRef("part").str(), pass > 1);
            else
                writer.begin(resolve_cache_dir(srvInterface), name, snapshot_id);
            do {
                if (inputReader.isNull(0) || inputReader.getStringRef(1).isNull())
                    fail("index '" + name + "': NULL byte_offset or chunk");
                const VString &chunk = inputReader.getStringRef(1);
                writer.write_at(inputReader.getIntRef(0), chunk.data(), chunk.length());
                if (isCanceled()) return;
            } while (inputReader.next());
            std::uint64_t bytes = 0;
            if (pass < passes)
                writer.keep();
            else
                bytes = writer.commit();

            outputWriter.getStringRef(0).copy(node);
            outputWriter.setInt(1, snapshot_id);
            outputWriter.setInt(2, (vint)bytes);
            outputWriter.getStringRef(3).copy(pass < passes ? "partial" : "loaded");
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
        parameterTypes.addVarchar(64, "part");
        parameterTypes.addInt("pass");
        parameterTypes.addInt("passes");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VLoad>(srvInterface.allocator); }
};

RegisterFactory(VLoadFactory);
