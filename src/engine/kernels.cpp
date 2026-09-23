#include "kernels.h"

#include <cstring>

namespace vvector {

namespace {

typedef float f16v __attribute__((vector_size(64)));
typedef float f8v __attribute__((vector_size(32)));
typedef float f4v __attribute__((vector_size(16)));
typedef std::int32_t i16v __attribute__((vector_size(64)));

inline __attribute__((always_inline)) f16v load(const float *p)
{
    f16v v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Lanes added in one fixed order: j + (j + 8), then j + (j + 4), j + (j + 2), 0 + 1.
inline __attribute__((always_inline)) float hsum(f16v a)
{
    f8v lo, hi;
    std::memcpy(&lo, &a, 32);
    std::memcpy(&hi, reinterpret_cast<const char *>(&a) + 32, 32);
    const f8v s8 = lo + hi;
    f4v lo4, hi4;
    std::memcpy(&lo4, &s8, 16);
    std::memcpy(&hi4, reinterpret_cast<const char *>(&s8) + 16, 16);
    const f4v s4 = lo4 + hi4;
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

enum Kind { DOT, L2SQ, L1 };

template <Kind K> inline __attribute__((always_inline)) f16v step(f16v acc, f16v x, f16v q)
{
    if (K == DOT) return acc + x * q;
    const f16v d = x - q;
    if (K == L2SQ) return acc + d * d;
    return acc + (f16v)((i16v)d & 0x7fffffff);      // |d|: clear the sign bit
}

template <Kind K> inline __attribute__((always_inline)) float pair(const float *a, const float *b, std::uint32_t stride)
{
    f16v acc = {};
    for (std::uint32_t j = 0; j < stride; j += LANES) acc = step<K>(acc, load(a + j), load(b + j));
    return hsum(acc);
}

// dot and cosine keys are minus the dot product.
template <Kind K> inline __attribute__((always_inline)) float key_of(float v) { return K == DOT ? -v : v; }

template <Kind K> inline void rows_1q(const float *rows, std::uint64_t n, std::uint32_t stride, const float *q, float *keys)
{
    for (std::uint64_t i = 0; i < n; ++i) {
        const float *r = rows + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        keys[i] = key_of<K>(pair<K>(r, q, stride));
    }
}

template <Kind K> inline void rows_4q(const float *rows, std::uint64_t n, std::uint32_t stride, const float *const q[4],
                                      float *keys)
{
    const float *q0 = q[0], *q1 = q[1], *q2 = q[2], *q3 = q[3];
    for (std::uint64_t i = 0; i < n; ++i) {
        const float *r = rows + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        f16v a0 = {}, a1 = {}, a2 = {}, a3 = {};
        for (std::uint32_t j = 0; j < stride; j += LANES) {
            const f16v x = load(r + j);
            a0 = step<K>(a0, x, load(q0 + j));
            a1 = step<K>(a1, x, load(q1 + j));
            a2 = step<K>(a2, x, load(q2 + j));
            a3 = step<K>(a3, x, load(q3 + j));
        }
        keys[i] = key_of<K>(hsum(a0));
        keys[n + i] = key_of<K>(hsum(a1));
        keys[2 * n + i] = key_of<K>(hsum(a2));
        keys[3 * n + i] = key_of<K>(hsum(a3));
    }
}

inline Kind kind_of(Metric m)
{
    switch (m) {
    case Metric::L2: return L2SQ;
    case Metric::L1: return L1;
    default: return DOT;
    }
}

} // namespace

// x86_64: one copy of each entry point per instruction set, chosen by the loader at run time, so one
// .so runs on every node. The scores do not depend on the copy (see kernels.h).
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define VV_CLONES __attribute__((target_clones("default", "avx2", "avx512f")))
#else
#define VV_CLONES
#endif

VV_CLONES float dot(const float *a, const float *b, std::uint32_t stride) { return pair<DOT>(a, b, stride); }
VV_CLONES float l2sq(const float *a, const float *b, std::uint32_t stride) { return pair<L2SQ>(a, b, stride); }
VV_CLONES float l1(const float *a, const float *b, std::uint32_t stride) { return pair<L1>(a, b, stride); }

VV_CLONES float distance_key(Metric m, const float *a, const float *b, std::uint32_t stride)
{
    switch (kind_of(m)) {
    case L2SQ: return pair<L2SQ>(a, b, stride);
    case L1: return pair<L1>(a, b, stride);
    default: return -pair<DOT>(a, b, stride);
    }
}

VV_CLONES void keys_1q(Metric m, const float *rows, std::uint64_t n, std::uint32_t stride, const float *q, float *keys)
{
    switch (kind_of(m)) {
    case L2SQ: rows_1q<L2SQ>(rows, n, stride, q, keys); break;
    case L1: rows_1q<L1>(rows, n, stride, q, keys); break;
    default: rows_1q<DOT>(rows, n, stride, q, keys); break;
    }
}

VV_CLONES void keys_4q(Metric m, const float *rows, std::uint64_t n, std::uint32_t stride, const float *const q[4],
                       float *keys)
{
    switch (kind_of(m)) {
    case L2SQ: rows_4q<L2SQ>(rows, n, stride, q, keys); break;
    case L1: rows_4q<L1>(rows, n, stride, q, keys); break;
    default: rows_4q<DOT>(rows, n, stride, q, keys); break;
    }
}

void normalize(float *v, std::uint32_t dims)
{
    double sum = 0;
    for (std::uint32_t i = 0; i < dims; ++i) sum += static_cast<double>(v[i]) * v[i];
    if (sum == 0) return;
    const double norm = std::sqrt(sum);
    for (std::uint32_t i = 0; i < dims; ++i) v[i] = static_cast<float>(v[i] / norm);
}

const char *kernel_target()
{
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return "avx512f";
    if (__builtin_cpu_supports("avx2")) return "avx2";
    return "default";
#elif defined(__aarch64__)
    return "neon";
#else
    return "generic";
#endif
}

} // namespace vvector
