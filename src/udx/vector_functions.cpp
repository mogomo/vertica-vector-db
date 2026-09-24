// Scalar vector functions (milestone M5), only those Vertica 26.2 has no equivalent for:
//   vector_add(a, b), vector_sub(a, b), vector_mul(a, b)       ARRAY[FLOAT], element by element
//   scalar_vector_mul(s, a)                                    ARRAY[FLOAT], s x every element
//   vector_normalize(a)                                        ARRAY[FLOAT], unit length (zero stays zero)
//   vector_l1(a, b), vector_l2sq(a, b)                         FLOAT, Manhattan and squared Euclidean distance
//   vector_hamming(a, b), vector_jaccard(a, b)                 INT and FLOAT, on ARRAY[INT] bit vectors
// Arguments ARRAY[FLOAT] (ARRAY[INT] and ARRAY[NUMERIC] are cast), computed in FLOAT64. A NULL
// argument gives NULL; two vectors of different lengths and a NULL element are errors.
// Native equivalents (not duplicated): APPLY_SUM(a) (sum of the elements), VECTOR_L2, DOT_PRODUCT,
// COSINE_SIMILARITY, VECTOR_MAGNITUDE, '[1, 2]'::ARRAY[FLOAT] and TO_JSON(a) (text).
// Thin adapter: the arithmetic is src/engine/vecmath.h.
#include "udx_common.h"
#include "../engine/vecmath.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace Vertica;
using namespace vvector_udx;

namespace {

// The error of a NULL element i. which: "first", "second", or "" for the only vector argument.
std::string element_error(const char *fn, int i, const char *which)
{
    return std::string(fn) + ": element " + std::to_string(i + 1) + " of the " + (*which ? std::string(which) + " " : "") + "vector is NULL";
}

// Reads the ARRAY[FLOAT] cell col (not NULL) into v as double.
void read_array(BlockReader &in, std::size_t col, std::vector<double> &v, const char *fn, const char *which)
{
    Array::ArrayReader a = in.getArrayRef(col);
    const int n = a->getNumRows();
    v.resize(n > 0 ? n : 0);
    if (n <= 0) return;
    if (a->getColStride(0) == static_cast<int>(sizeof(vfloat))) {
        const vfloat *p = a->getFloatPtr(0);
        for (int i = 0; i < n; ++i) {
            if (vfloatIsNull(p[i])) fail(element_error(fn, i, which));
            v[i] = p[i];
        }
        return;
    }
    for (int i = 0; a->hasData(); a->next(), ++i) {
        if (a->isNull(0)) fail(element_error(fn, i, which));
        v[i] = a->getFloatRef(0);
    }
}

// Reads the ARRAY[INT] cell col (not NULL) into v.
void read_int_array(BlockReader &in, std::size_t col, std::vector<std::int64_t> &v, const char *fn, const char *which)
{
    Array::ArrayReader a = in.getArrayRef(col);
    const int n = a->getNumRows();
    v.resize(n > 0 ? n : 0);
    for (int i = 0; n > 0 && a->hasData(); a->next(), ++i) {
        if (a->isNull(0)) fail(element_error(fn, i, which));
        v[i] = a->getIntRef(0);
    }
}

void same_length(const char *fn, std::size_t a, std::size_t b)
{
    if (a != b) fail(std::string(fn) + ": the vectors have different lengths: " + std::to_string(a) + " and " + std::to_string(b) + " elements");
}

void write_array(BlockWriter &out, const std::vector<double> &v)
{
    Array::ArrayWriter w = out.getArrayRef(0);
    for (const double x : v) {
        w->setFloat(0, x);
        w->next();
    }
    w.commit();
}

// The most elements an array argument can hold (8-byte elements): the bound of the result.
int max_elements(const SizedColumnTypes &args)
{
    int most = 0;
    for (std::size_t i = 0; i < args.getColumnCount(); ++i)
        if (args.getColumnType(i).isArrayType()) most = std::max(most, args.getColumnType(i).getMaxSize() / 8);
    return std::max(most, 1);
}

enum class Kind { Add, Sub, Mul, Scale, Normalize, L1, L2sq, Hamming, Jaccard };

const char *name_of(Kind k)
{
    switch (k) {
    case Kind::Add: return "vector_add";
    case Kind::Sub: return "vector_sub";
    case Kind::Mul: return "vector_mul";
    case Kind::Scale: return "scalar_vector_mul";
    case Kind::Normalize: return "vector_normalize";
    case Kind::L1: return "vector_l1";
    case Kind::L2sq: return "vector_l2sq";
    case Kind::Hamming: return "vector_hamming";
    case Kind::Jaccard: return "vector_jaccard";
    }
    return "vector function";
}

template <Kind K>
class VectorFunction : public ScalarFunction
{
    void processBlock(ServerInterface &srvInterface, BlockReader &in, BlockWriter &out) override
    {
        const char *fn = name_of(K);
        try {
            const std::size_t args = K == Kind::Normalize ? 1 : 2;
            do {
                bool null = false;
                for (std::size_t c = 0; c < args; ++c) null |= in.isNull(c);
                if (null) {
                    out.setNull();
                    out.next();
                    continue;
                }
                switch (K) {
                case Kind::Add: case Kind::Sub: case Kind::Mul:
                    read_array(in, 0, a_, fn, "first");
                    read_array(in, 1, b_, fn, "second");
                    same_length(fn, a_.size(), b_.size());
                    r_.resize(a_.size());
                    vvector::vec_elementwise(K == Kind::Add ? vvector::ElementOp::Add : K == Kind::Sub ? vvector::ElementOp::Sub : vvector::ElementOp::Mul,
                                             a_.data(), b_.data(), a_.size(), r_.data());
                    write_array(out, r_);
                    break;
                case Kind::Scale:
                    read_array(in, 1, a_, fn, "");
                    r_.resize(a_.size());
                    vvector::vec_scale(in.getFloatRef(0), a_.data(), a_.size(), r_.data());
                    write_array(out, r_);
                    break;
                case Kind::Normalize:
                    read_array(in, 0, a_, fn, "");
                    r_.resize(a_.size());
                    vvector::vec_normalize(a_.data(), a_.size(), r_.data());
                    write_array(out, r_);
                    break;
                case Kind::L1: case Kind::L2sq:
                    read_array(in, 0, a_, fn, "first");
                    read_array(in, 1, b_, fn, "second");
                    same_length(fn, a_.size(), b_.size());
                    out.setFloat(K == Kind::L1 ? vvector::vec_l1(a_.data(), b_.data(), a_.size())
                                               : vvector::vec_l2sq(a_.data(), b_.data(), a_.size()));
                    break;
                case Kind::Hamming: case Kind::Jaccard:
                    read_int_array(in, 0, ia_, fn, "first");
                    read_int_array(in, 1, ib_, fn, "second");
                    same_length(fn, ia_.size(), ib_.size());
                    if (K == Kind::Hamming) out.setInt(vvector::vec_hamming(ia_.data(), ib_.data(), ia_.size()));
                    else out.setFloat(vvector::vec_jaccard(ia_.data(), ib_.data(), ia_.size()));
                    break;
                }
                out.next();
            } while (in.next());
        } catch (std::exception &e) {
            // Messages of the SDK do not name the function; ours start with its name already.
            std::string m = e.what();
            if (m.compare(0, std::strlen(fn), fn) != 0) m = std::string(fn) + ": " + m;
            vt_report_error(0, "%s", m.c_str());
        }
    }

    std::vector<double> a_, b_, r_;
    std::vector<std::int64_t> ia_, ib_;
};

template <Kind K>
class VectorFunctionFactory : public ScalarFunctionFactory
{
public:
    VectorFunctionFactory() { vol = IMMUTABLE; strict = RETURN_NULL_ON_NULL_INPUT; }

    void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType) override
    {
        switch (K) {
        case Kind::Scale: argTypes.addFloat(); argTypes.addArrayType(Float8OID); break;
        case Kind::Normalize: argTypes.addArrayType(Float8OID); break;
        case Kind::Hamming: case Kind::Jaccard: argTypes.addArrayType(Int8OID); argTypes.addArrayType(Int8OID); break;
        default: argTypes.addArrayType(Float8OID); argTypes.addArrayType(Float8OID); break;
        }
        switch (K) {
        case Kind::L1: case Kind::L2sq: case Kind::Jaccard: returnType.addFloat(); break;
        case Kind::Hamming: returnType.addInt(); break;
        default: returnType.addArrayType(Float8OID); break;
        }
    }

    void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &argTypes, SizedColumnTypes &returnType) override
    {
        switch (K) {
        case Kind::L1: case Kind::L2sq: case Kind::Jaccard: returnType.addFloat(name_of(K)); break;
        case Kind::Hamming: returnType.addInt(name_of(K)); break;
        default: {
            SizedColumnTypes element;
            element.addFloat();
            returnType.addArrayType(Field(element.getColumnType(0), ""), name_of(K), max_elements(argTypes));
        }
        }
    }

    ScalarFunction *createScalarFunction(ServerInterface &srvInterface) override
    { return vt_createFuncObject<VectorFunction<K>>(srvInterface.allocator); }
};

} // namespace

class VectorAddFactory : public VectorFunctionFactory<Kind::Add> {};
class VectorSubFactory : public VectorFunctionFactory<Kind::Sub> {};
class VectorMulFactory : public VectorFunctionFactory<Kind::Mul> {};
class ScalarVectorMulFactory : public VectorFunctionFactory<Kind::Scale> {};
class VectorNormalizeFactory : public VectorFunctionFactory<Kind::Normalize> {};
class VectorL1Factory : public VectorFunctionFactory<Kind::L1> {};
class VectorL2sqFactory : public VectorFunctionFactory<Kind::L2sq> {};
class VectorHammingFactory : public VectorFunctionFactory<Kind::Hamming> {};
class VectorJaccardFactory : public VectorFunctionFactory<Kind::Jaccard> {};
RegisterFactory(VectorAddFactory);
RegisterFactory(VectorSubFactory);
RegisterFactory(VectorMulFactory);
RegisterFactory(ScalarVectorMulFactory);
RegisterFactory(VectorNormalizeFactory);
RegisterFactory(VectorL1Factory);
RegisterFactory(VectorL2sqFactory);
RegisterFactory(VectorHammingFactory);
RegisterFactory(VectorJaccardFactory);
