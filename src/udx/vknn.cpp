// vknn: the k nearest neighbours of one vector, as a function of the row: no OVER(), no view.
//   SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='docs', k=10) FROM app.queries q;
//   SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='docs', query='[0.1, 0.2]') FROM dual;
// Every input row gives up to k output rows (id, score, rank); columns selected beside vknn are
// repeated on each of them. A NULL vector gives no rows. Snapshot only: the journal is not applied
// and the cache is not checked against the manifest (vsearch over the _snap or _delta view does
// both). Parameters as vsearch: index_name, k, precision, ef_search, exact, radius, threads, query
// (used for rows whose vector is NULL), rescore, oversampling, cache_dir.
// Declared an exploder (TransformFunctionFactory::Properties::isExploder, docs/VERTICA_NOTES.md):
// Vertica then calls it without OVER() and hands the rows to several instances in parallel.
#include "udx_common.h"
#include "../engine/kernels.h"
#include "../engine/text.h"

#include <cstring>

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vknn";

class VKnn : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &in,
                                  PartitionWriter &out)
    {
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            vvector::MappedSnapshot snap;
            snap.open_active(resolve_cache_dir(srvInterface), name);
            Settings cfg(srvInterface, snap.options());
            const SearchSettings ss(cfg, params);
            const vvector::VectorSet &set = snap.vectors();
            const std::uint32_t dims = set.dims, stride = set.row_stride;
            const bool cosine = set.metric == vvector::Metric::Cosine;
            auto check_dims = [&](std::uint32_t n, const std::string &what) {
                if (n != dims)
                    fail("index '" + name + "' has " + std::to_string(dims) + " dimensions, " + what + " has " + std::to_string(n));
            };

            // The query parameter, for rows without a vector.
            std::vector<float> fixed;
            if (params.containsParameter("query")) {
                const VString &text = params.getStringRef("query");
                const std::vector<float> v = vvector::parse_vector_text(text.data(), text.length());
                check_dims(static_cast<std::uint32_t>(v.size()), "the query parameter");
                fixed.assign(stride, 0.0f);
                std::memcpy(fixed.data(), v.data(), v.size() * sizeof(float));
                if (cosine) vvector::normalize(fixed.data(), dims);
            }

            std::vector<float> q(stride, 0.0f);
            std::vector<vvector::Neighbor> found;
            std::vector<std::uint32_t> count;
            const vvector::FlatSearch fs = ss.describe(set, q.data(), 1);
            vint rows = 0;
            do {
                if ((++rows & 0xFF) == 0 && isCanceled()) return;
                if (in.isNull(0)) {
                    if (fixed.empty()) continue;             // NULL vector: no rows
                    std::memcpy(q.data(), fixed.data(), stride * sizeof(float));
                } else {
                    std::fill(q.begin(), q.end(), 0.0f);
                    read_floats(in, 0, [&](std::uint32_t n) { check_dims(n, "the query vector"); return q.data(); },
                                "the query vector");
                    if (cosine) vvector::normalize(q.data(), dims);
                }
                try {
                    ss.search(fs, set, set.tombstone_bits, nullptr, found, count, [this] { return isCanceled(); });
                } catch (const vvector::Cancelled &) {
                    return;
                }
                // The rows of this input row are written before the reader moves on: that is how
                // Vertica pairs them with the columns selected beside vknn.
                for (std::uint32_t i = 0; i < count[0]; ++i) {
                    out.setInt(0, found[i].id);
                    out.setFloat(1, vvector::key_to_score(fs.metric, found[i].key));
                    out.setInt(2, static_cast<vint>(i) + 1);
                    out.next();
                }
            } while (in.next());
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VKnnFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addArrayType(Float8OID);
        returnType.addInt();
        returnType.addFloat();
        returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addInt("id");
        outputTypes.addFloat("score");
        outputTypes.addInt("rank");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        add_search_parameters(parameterTypes);
    }

    virtual void getFunctionProperties(ServerInterface &srvInterface, const SizedColumnTypes &argTypes,
                                       Properties &properties)
    {
        properties.isExploder = true;
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VKnn>(srvInterface.allocator); }
};

RegisterFactory(VKnnFactory);
