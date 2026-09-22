// vsearch: k nearest neighbours of each query vector. STUB: the search itself is milestone M1.
//   vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='docs', k=10) OVER()
//   FROM (SELECT * FROM app.docs_delta UNION ALL SELECT 1, ARRAY[...], NULL, NULL, NULL, NULL, NULL) q
// Output (qid, id, score, rank). score is what the built-in function of the index metric returns
// (VECTOR_L2, COSINE_SIMILARITY or DOT_PRODUCT); rank 1 is the closest.
// What works now: the input is read and checked, the active snapshot is opened from the node's
// cache and the stale-cache check runs. Then the function stops with "not implemented yet".
#include "udx_common.h"

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vsearch";

class VSearch : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &in,
                                  PartitionWriter &outputWriter)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(FN, srvInterface);
            const vint k = params.containsParameter("k") ? params.getIntRef("k") : 10;
            if (k < 1) vt_report_error(0, "%s: k must be at least 1", FN);

            vint wanted_snapshot = -1, requests = 0, delta_rows = 0, rows = 0;
            std::vector<float> v;
            std::uint64_t query_dims = 0;
            do {
                if (!in.isNull(COL_ID)) {
                    const bool del = !in.isNull(COL_DEL) && in.getBoolRef(COL_DEL) == vbool_true;
                    if (!del && in.isNull(COL_VEC))
                        vt_report_error(0, "%s: journal row with id %lld has no vector and is not a delete",
                                        FN, (long long)in.getIntRef(COL_ID));
                    ++delta_rows;
                } else if (!in.isNull(COL_QVEC)) {
                    read_vector(in, COL_QVEC, v);
                    if (query_dims == 0) query_dims = v.size();
                    else if (v.size() != query_dims)
                        vt_report_error(0, "%s: query vectors of different lengths: %llu and %llu", FN,
                                        (unsigned long long)query_dims, (unsigned long long)v.size());
                    ++requests;
                }
                if (!in.isNull(COL_SNAPSHOT_ID) && in.getIntRef(COL_SNAPSHOT_ID) > wanted_snapshot)
                    wanted_snapshot = in.getIntRef(COL_SNAPSHOT_ID);
                if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
            } while (in.next());

            vvector::MappedSnapshot snap;
            snap.open_active(resolve_cache_dir(srvInterface), name);
            // The delta view belongs to a newer snapshot than this node has cached.
            if (snap.snapshot_id() < wanted_snapshot)
                vt_report_error(0, "%s: snapshot cache stale on %s: run vload", FN,
                                srvInterface.getCurrentNodeName().c_str());
            if (requests > 0 && query_dims != snap.vectors().dims)
                vt_report_error(0, "%s: index '%s' has %u dimensions, the query vectors have %llu", FN, name.c_str(),
                                snap.vectors().dims, (unsigned long long)query_dims);
            vt_report_error(0, "%s: index '%s': search is not implemented yet (%lld query rows, %lld journal rows read)",
                            FN, name.c_str(), (long long)requests, (long long)delta_rows);
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VSearchFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        add_query_input(argTypes);
        returnType.addInt();
        returnType.addInt();
        returnType.addFloat();
        returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("qid");
        outputTypes.addInt("id");
        outputTypes.addFloat("score");
        outputTypes.addInt("rank");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        parameterTypes.addInt("k");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VSearch>(srvInterface.allocator); }
};

RegisterFactory(VSearchFactory);
