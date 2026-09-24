#include "vecmath.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vvector {

void vec_elementwise(ElementOp op, const double *a, const double *b, std::uint64_t n, double *out)
{
    switch (op) {
    case ElementOp::Add: for (std::uint64_t i = 0; i < n; ++i) out[i] = a[i] + b[i]; break;
    case ElementOp::Sub: for (std::uint64_t i = 0; i < n; ++i) out[i] = a[i] - b[i]; break;
    case ElementOp::Mul: for (std::uint64_t i = 0; i < n; ++i) out[i] = a[i] * b[i]; break;
    }
}

void vec_scale(double s, const double *a, std::uint64_t n, double *out)
{
    for (std::uint64_t i = 0; i < n; ++i) out[i] = s * a[i];
}

void vec_normalize(const double *a, std::uint64_t n, double *out)
{
    double sq = 0;
    for (std::uint64_t i = 0; i < n; ++i) sq += a[i] * a[i];
    const double norm = std::sqrt(sq);
    for (std::uint64_t i = 0; i < n; ++i) out[i] = norm > 0 ? a[i] / norm : 0.0;
}

double vec_l1(const double *a, const double *b, std::uint64_t n)
{
    double s = 0;
    for (std::uint64_t i = 0; i < n; ++i) s += std::fabs(a[i] - b[i]);
    return s;
}

double vec_l2sq(const double *a, const double *b, std::uint64_t n)
{
    double s = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

std::int64_t vec_hamming(const std::int64_t *a, const std::int64_t *b, std::uint64_t n)
{
    std::int64_t bits = 0;
    for (std::uint64_t i = 0; i < n; ++i) bits += __builtin_popcountll(static_cast<std::uint64_t>(a[i] ^ b[i]));
    return bits;
}

double vec_jaccard(const std::int64_t *a, const std::int64_t *b, std::uint64_t n)
{
    std::int64_t both = 0, either = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint64_t x = static_cast<std::uint64_t>(a[i]), y = static_cast<std::uint64_t>(b[i]);
        both += __builtin_popcountll(x & y);
        either += __builtin_popcountll(x | y);
    }
    return either == 0 ? 1.0 : double(both) / double(either);
}

void VectorSum::add(const double *v, std::uint64_t n)
{
    if (count_ == 0) sum_.assign(n, 0.0);
    else if (n != sum_.size())
        throw std::runtime_error("the vectors have different lengths: " + std::to_string(sum_.size()) + " and " + std::to_string(n) + " elements");
    for (std::uint64_t i = 0; i < n; ++i) sum_[i] += v[i];
    ++count_;
}

std::vector<double> VectorSum::average() const
{
    std::vector<double> avg(sum_.size());
    for (std::size_t i = 0; i < sum_.size(); ++i) avg[i] = sum_[i] / double(count_);
    return avg;
}

} // namespace vvector
