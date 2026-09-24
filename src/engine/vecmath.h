// vvector engine: the arithmetic of the vector functions (vector_add, vector_l1, vector_avg, ...,
// milestone M5). Everything is computed in double (FLOAT64, Vertica's FLOAT), one element after the
// other in index order, so every CPU gives the same result. No Vertica includes: the adapters are
// src/udx/vector_functions.cpp and src/udx/vector_aggregates.cpp.
#ifndef VVECTOR_ENGINE_VECMATH_H
#define VVECTOR_ENGINE_VECMATH_H

#include <cstdint>
#include <vector>

namespace vvector {

enum class ElementOp { Add, Sub, Mul };

// out[i] = a[i] op b[i] for i < n. out may be a or b.
void vec_elementwise(ElementOp op, const double *a, const double *b, std::uint64_t n, double *out);

// out[i] = s x a[i].
void vec_scale(double s, const double *a, std::uint64_t n, double *out);

// a scaled to unit length (Euclidean norm computed first, then every element divided by it). A zero
// vector stays zero, as the search treats it (kernels.h normalize).
void vec_normalize(const double *a, std::uint64_t n, double *out);

// sum |a[i] - b[i]|: the Manhattan distance.
double vec_l1(const double *a, const double *b, std::uint64_t n);

// sum (a[i] - b[i])^2: the squared Euclidean distance (VECTOR_L2 squared, without the square root).
double vec_l2sq(const double *a, const double *b, std::uint64_t n);

// Binary vectors packed into 64-bit integers (or elements 0 and 1). Hamming: the number of bits that
// differ, sum popcount(a[i] xor b[i]). Jaccard (Tanimoto): bits set in both / bits set in either,
// sum popcount(a[i] and b[i]) / sum popcount(a[i] or b[i]); two vectors without any bit set give 1.
std::int64_t vec_hamming(const std::int64_t *a, const std::int64_t *b, std::uint64_t n);
double vec_jaccard(const std::int64_t *a, const std::int64_t *b, std::uint64_t n);

// Element sum of many vectors of one length (vector_sum, vector_avg). add() in any row order gives
// the same sum only up to rounding; the adapters feed the rows in the order Vertica gives them.
class VectorSum {
public:
    // Adds one vector; throws std::runtime_error when its length differs from the first one's.
    void add(const double *v, std::uint64_t n);
    std::uint64_t count() const { return count_; }
    const std::vector<double> &sum() const { return sum_; }
    // The average: sum / count (empty when count is 0).
    std::vector<double> average() const;

private:
    std::vector<double> sum_;
    std::uint64_t count_ = 0;
};

} // namespace vvector

#endif
