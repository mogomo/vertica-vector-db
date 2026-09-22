// vinfo: what each node has in its snapshot cache.
//   vvector.vinfo([USING PARAMETERS index_name='docs']) OVER(PARTITION NODES) FROM vvector.probe
// Output (node_name, index_name, snapshot_id, max_ver, vector_count, dims, metric, cache_file, loaded).
#include "udx_common.h"

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vinfo";

class VInfo : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        const std::string node = srvInterface.getCurrentNodeName();
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string cache_dir = resolve_cache_dir(srvInterface);
            std::vector<std::string> names;
            if (params.containsParameter("index_name")) names.push_back(params.getStringRef("index_name").str());
            else names = vvector::list_cached_indexes(cache_dir);

            for (const std::string &name : names) {
                outputWriter.getStringRef(0).copy(node);
                outputWriter.getStringRef(1).copy(name);
                try {
                    vvector::MappedSnapshot snap;
                    snap.open_active(cache_dir, name);
                    const vvector::VectorSet &s = snap.vectors();
                    outputWriter.setInt(2, snap.snapshot_id());
                    outputWriter.setInt(3, s.max_ver);
                    outputWriter.setInt(4, (vint)s.count);
                    outputWriter.setInt(5, (vint)s.dims);
                    outputWriter.getStringRef(6).copy(vvector::metric_name(s.metric));
                    outputWriter.getStringRef(7).copy(snap.path());
                    outputWriter.setBool(8, vbool_true);
                } catch (std::runtime_error &e) {
                    // Not loaded or damaged: say why in the cache_file column.
                    for (int c = 2; c <= 6; ++c) outputWriter.setNull(c);
                    outputWriter.getStringRef(7).copy(std::string(e.what()).substr(0, 1024));
                    outputWriter.setBool(8, vbool_false);
                }
                outputWriter.next();
            }
            if (names.empty()) {
                outputWriter.getStringRef(0).copy(node);
                for (int c = 1; c <= 6; ++c) outputWriter.setNull(c);
                outputWriter.getStringRef(7).copy("no indexes in " + cache_dir);
                outputWriter.setBool(8, vbool_false);
                outputWriter.next();
            }
        } catch (std::exception &e) {
            vt_report_error(0, "%s: on %s: %s", FN, node.c_str(), e.what());
        }
    }
};

class VInfoFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        returnType.addVarchar();
        returnType.addVarchar();
        for (int i = 0; i < 4; ++i) returnType.addInt();
        returnType.addVarchar();
        returnType.addVarchar();
        returnType.addBool();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(128, "node_name");
        outputTypes.addVarchar(128, "index_name");
        outputTypes.addInt("snapshot_id");
        outputTypes.addInt("max_ver");
        outputTypes.addInt("vector_count");
        outputTypes.addInt("dims");
        outputTypes.addVarchar(16, "metric");
        outputTypes.addVarchar(1200, "cache_file");
        outputTypes.addBool("loaded");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    { add_common_parameters(parameterTypes); }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VInfo>(srvInterface.allocator); }
};

RegisterFactory(VInfoFactory);
