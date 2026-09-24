// vector_sum and vector_avg (milestone M5): the element sum and the element average of the vectors
// of a partition, as transform functions:
//   SELECT vvector.vector_avg(vec) OVER() FROM app.docs;                          -- one centroid
//   SELECT category, vvector.vector_avg(vec) OVER(PARTITION BY category) FROM app.docs;
// One row per partition: the ARRAY[FLOAT] result (NULL when every vector is NULL). NULL vectors are
// skipped; vectors of different lengths and NULL elements are errors. Computed in FLOAT64.
// Transform functions, not aggregates: a C++ aggregate cannot read an ARRAY argument in Vertica 26.2
// (docs/VERTICA_NOTES.md), and a transform function can run fenced.
// Thin adapter: the arithmetic is src/engine/vecmath.h.
#include "udx_common.h"
#include "../engine/vecmath.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace Vertica;
using namespace vvector_udx;

namespace {

template <bool Average>
class VectorAggregate : public TransformFunction
{
    void processPartition(ServerInterface &srvInterface, PartitionReader &in, PartitionWriter &out) override
    {
        const char *fn = Average ? "vector_avg" : "vector_sum";
        try {
            vvector::VectorSum sum;
            std::vector<double> v;
            vint rows = 0;
            do {
                if ((++rows & 0xFFFF) == 0 && isCanceled()) return;
                if (in.isNull(0)) continue;
                Array::ArrayReader a = in.getArrayRef(0);
                const int n = a->getNumRows();
                v.resize(n > 0 ? n : 0);
                if (n > 0 && a->getColStride(0) == static_cast<int>(sizeof(vfloat))) {
                    const vfloat *p = a->getFloatPtr(0);
                    for (int i = 0; i < n; ++i) {
                        if (vfloatIsNull(p[i])) fail(std::string(fn) + ": element " + std::to_string(i + 1) + " of a vector is NULL");
                        v[i] = p[i];
                    }
                } else {
                    for (int i = 0; n > 0 && a->hasData(); a->next(), ++i) {
                        if (a->isNull(0)) fail(std::string(fn) + ": element " + std::to_string(i + 1) + " of a vector is NULL");
                        v[i] = a->getFloatRef(0);
                    }
                }
                try {
                    sum.add(v.data(), v.size());
                } catch (const std::runtime_error &e) {
                    fail(std::string(fn) + ": " + e.what());
                }
            } while (in.next());

            if (sum.count() == 0) {
                out.setNull(0);
            } else {
                const std::vector<double> r = Average ? sum.average() : sum.sum();
                Array::ArrayWriter w = out.getArrayRef(0);
                for (const double x : r) {
                    w->setFloat(0, x);
                    w->next();
                }
                w.commit();
            }
            out.next();
        } catch (std::exception &e) {
            std::string m = e.what();
            if (m.compare(0, std::strlen(fn), fn) != 0) m = std::string(fn) + ": " + m;
            vt_report_error(0, "%s", m.c_str());
        }
    }
};

template <bool Average>
class VectorAggregateFactory : public TransformFunctionFactory
{
    void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType) override
    {
        argTypes.addArrayType(Float8OID);
        returnType.addArrayType(Float8OID);
    }

    void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes, SizedColumnTypes &outputTypes) override
    {
        SizedColumnTypes element;
        element.addFloat();
        outputTypes.addArrayType(Field(element.getColumnType(0), ""), Average ? "vector_avg" : "vector_sum",
                                 std::max(1, inputTypes.getColumnType(0).getMaxSize() / 8));
    }

    TransformFunction *createTransformFunction(ServerInterface &srvInterface) override
    { return vt_createFuncObject<VectorAggregate<Average>>(srvInterface.allocator); }
};

} // namespace

class VectorSumFactory : public VectorAggregateFactory<false> {};
class VectorAvgFactory : public VectorAggregateFactory<true> {};
RegisterFactory(VectorSumFactory);
RegisterFactory(VectorAvgFactory);
