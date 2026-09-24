// vconfig: writes the index defaults of set_index_options into the cache of every node, where
// vsearch reads them (a query cannot read the manifest table).
//   vvector_admin.vconfig(k USING PARAMETERS index_name='docs', options='precision=best,threads=4')
//       OVER(PARTITION NODES) FROM vvector.probe
// options: name=value items, names precision, freshness, ef_search, threads, memory_mode; an empty
// value or a missing name = the built-in default. Output (node_name, status). Called by the
// procedures; role vvector_admin only. It writes <cache_dir>/<index_name>/OPTIONS and nothing else.
// index_cache_dir (the index option cache_dir; the procedures always pass it): a directory writes
// the defaults there and, in the default cache directory and in the one of the session, an OPTIONS
// file that also names that directory, so a query without a cache_dir parameter finds the index
// there; '' writes the defaults to the session's directory and removes such a redirect left behind.
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
            vvector::IndexOptions options = vvector::parse_index_options(text);
            options.erase("cache_dir");
            const std::string session_dir = resolve_cache_dir(srvInterface);
            if (!params.containsParameter("index_cache_dir")) {
                vvector::write_index_options(session_dir, name, options);
            } else {
                const std::string home = params.getStringRef("index_cache_dir").str();
                if (!home.empty() && !vvector::valid_cache_dir(home))
                    fail("index_cache_dir '" + home + "' is not an absolute path of letters, digits and / . _ -");
                const std::string target = home.empty() ? session_dir : home;
                vvector::write_index_options(target, name, options);
                vvector::IndexOptions redirect = options;
                if (!home.empty()) redirect["cache_dir"] = home;
                for (const std::string &dir : {std::string(vvector::DEFAULT_CACHE_DIR), session_dir}) {
                    if (dir == target) continue;
                    // Without a home directory, only a redirect left behind is rewritten.
                    if (home.empty()) {
                        bool stale = true;
                        try { stale = vvector::read_index_options(dir, name).count("cache_dir") > 0; } catch (const std::runtime_error &) {}
                        if (!stale) continue;
                    }
                    vvector::write_index_options(dir, name, redirect);
                }
            }
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
        parameterTypes.addVarchar(1024, "index_cache_dir");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VConfig>(srvInterface.allocator); }
};

RegisterFactory(VConfigFactory);
