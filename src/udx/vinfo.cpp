// vinfo: what each node has in its snapshot cache.
//   vvector.vinfo([USING PARAMETERS index_name='docs']) OVER(PARTITION NODES) FROM vvector.probe
// Output (node_name, index_name, snapshot_id, max_ver, vector_count, dims, metric, index_type,
// quantization, graph_bytes, tombstones, base_snapshot, precision_default, freshness_default,
// ef_search_default, threads_default, cache_file, loaded, resident_mb, capacity, file_bytes). The
// defaults are the index defaults of set_index_options (NULL = the built-in default). vector_count
// counts the live vectors; the snapshot has vector_count + tombstones positions. resident_mb: how much
// of the cache file is in the node's memory now (page cache; milestone M6). capacity: the positions
// the layout has room for (milestone M7: an incremental build appends into the room without moving a
// section); file_bytes: the size of the cache file.
// Without index_name it lists every index of the cache directory that has an ACTIVE or OPTIONS file.
#include "udx_common.h"

#include <limits>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vinfo";

enum { C_NODE, C_INDEX, C_SNAPSHOT, C_MAX_VER, C_COUNT, C_DIMS, C_METRIC, C_TYPE, C_QUANT, C_GRAPH, C_TOMB, C_BASE,
       C_PRECISION, C_FRESHNESS, C_EF, C_THREADS, C_FILE, C_LOADED, C_RESIDENT, C_CAPACITY, C_BYTES, C_COLUMNS };

class VInfo : public TransformFunction
{
    static void option(PartitionWriter &out, int col, const vvector::IndexOptions &o, const char *name)
    {
        auto it = o.find(name);
        if (it == o.end()) out.setNull(col);
        else out.getStringRef(col).copy(it->second);
    }

    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &out)
    {
        const std::string node = srvInterface.getCurrentNodeName();
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string cache_dir = resolve_cache_dir(srvInterface);
            std::vector<std::string> names;
            if (params.containsParameter("index_name")) names.push_back(read_index_name(srvInterface));
            else names = vvector::list_cached_indexes(cache_dir);

            for (const std::string &name : names) {
                out.getStringRef(C_NODE).copy(node);
                out.getStringRef(C_INDEX).copy(name);
                try {
                    // Always read ACTIVE again, not what this process trusted for the last 200 ms:
                    // refresh_index calls vinfo right after a vload in the same session.
                    vvector::MappedSnapshot snap;
                    // No read-ahead advice: vinfo reports what is in memory and must not load it.
                    snap.open_active(cache_dir, name, std::numeric_limits<std::int64_t>::max(), false);
                    const vvector::VectorSet &s = snap.vectors();
                    out.setInt(C_SNAPSHOT, snap.snapshot_id());
                    out.setInt(C_MAX_VER, s.max_ver);
                    out.setInt(C_COUNT, (vint)(s.count - s.tombstones));     // live vectors
                    out.setInt(C_DIMS, (vint)s.dims);
                    out.getStringRef(C_METRIC).copy(vvector::metric_name(s.metric));
                    out.getStringRef(C_TYPE).copy(s.has_graph() ? "hnsw" : "flat");
                    out.getStringRef(C_QUANT).copy((s.flags & vvector::FLAG_SQ8) ? "sq8" : "none");
                    out.setInt(C_GRAPH, (vint)s.graph_bytes);
                    out.setInt(C_TOMB, (vint)s.tombstones);
                    out.setInt(C_BASE, s.base_snapshot);
                    option(out, C_PRECISION, snap.options(), "precision");
                    option(out, C_FRESHNESS, snap.options(), "freshness");
                    option(out, C_EF, snap.options(), "ef_search");
                    option(out, C_THREADS, snap.options(), "threads");
                    out.getStringRef(C_FILE).copy(snap.path());
                    out.setBool(C_LOADED, vbool_true);
                    out.setInt(C_RESIDENT, (vint)(snap.resident_bytes() / 1048576));
                    out.setInt(C_CAPACITY, (vint)s.capacity);
                    out.setInt(C_BYTES, (vint)snap.size());
                } catch (std::runtime_error &e) {
                    // Not loaded or damaged: say why in the cache_file column.
                    for (int c = C_SNAPSHOT; c < C_FILE; ++c) out.setNull(c);
                    out.getStringRef(C_FILE).copy(std::string(e.what()).substr(0, 1024));
                    out.setBool(C_LOADED, vbool_false);
                    out.setNull(C_RESIDENT);
                    out.setNull(C_CAPACITY);
                    out.setNull(C_BYTES);
                }
                out.next();
            }
            if (names.empty()) {
                out.getStringRef(C_NODE).copy(node);
                for (int c = C_INDEX; c < C_FILE; ++c) out.setNull(c);
                out.getStringRef(C_FILE).copy("no indexes in " + cache_dir);
                out.setBool(C_LOADED, vbool_false);
                out.setNull(C_RESIDENT);
                out.setNull(C_CAPACITY);
                out.setNull(C_BYTES);
                out.next();
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
        returnType.addVarchar();           // node_name
        returnType.addVarchar();           // index_name
        for (int i = 0; i < 4; ++i) returnType.addInt();
        for (int i = 0; i < 3; ++i) returnType.addVarchar();
        for (int i = 0; i < 3; ++i) returnType.addInt();
        for (int i = 0; i < 4; ++i) returnType.addVarchar();
        returnType.addVarchar();           // cache_file
        returnType.addBool();
        returnType.addInt();               // resident_mb
        returnType.addInt();               // capacity
        returnType.addInt();               // file_bytes
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
        outputTypes.addVarchar(16, "index_type");
        outputTypes.addVarchar(16, "quantization");
        outputTypes.addInt("graph_bytes");
        outputTypes.addInt("tombstones");
        outputTypes.addInt("base_snapshot");
        outputTypes.addVarchar(16, "precision_default");
        outputTypes.addVarchar(16, "freshness_default");
        outputTypes.addVarchar(16, "ef_search_default");
        outputTypes.addVarchar(16, "threads_default");
        outputTypes.addVarchar(1200, "cache_file");
        outputTypes.addBool("loaded");
        outputTypes.addInt("resident_mb");
        outputTypes.addInt("capacity");
        outputTypes.addInt("file_bytes");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    { add_common_parameters(parameterTypes); }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VInfo>(srvInterface.allocator); }
};

RegisterFactory(VInfoFactory);
