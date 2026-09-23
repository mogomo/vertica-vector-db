// vconfig: writes the index defaults of set_index_options into the cache of every node, where
// vsearch reads them (a query cannot read the manifest table).
//   vvector.vconfig(k USING PARAMETERS index_name='docs', options='precision=best,threads=4')
//       OVER(PARTITION NODES) FROM vvector.probe
// options: name=value items, names precision, freshness, ef_search, threads; an empty value or a
// missing name = the built-in default. Output (node_name, status). Called by the procedures; role
// vvector_admin only. It writes one file, <cache_dir>/<index_name>/OPTIONS, and nothing else.
#include "udx_common.h"

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vconfig";

class VConfig : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &out)
    {
        const std::string node = srvInterface.getCurrentNodeName();
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            const std::string text = params.containsParameter("options") ? params.getStringRef("options").str() : "";
            vvector::write_index_options(resolve_cache_dir(srvInterface), name, vvector::parse_index_options(text));
            out.getStringRef(0).copy(node);
            out.getStringRef(1).copy("written");
            out.next();
        } catch (std::exception &e) {
            vt_report_error(0, "%s: on %s: %s", FN, node.c_str(), e.what());
        }
    }
};

class VConfigFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        returnType.addVarchar();
        returnType.addVarchar();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(128, "node_name");
        outputTypes.addVarchar(32, "status");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        parameterTypes.addVarchar(1024, "options");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VConfig>(srvInterface.allocator); }
};

RegisterFactory(VConfigFactory);
