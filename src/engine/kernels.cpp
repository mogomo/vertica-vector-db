#include "kernels.h"

#include <cstring>

namespace vvector {

namespace {

// 16 lanes held as four native 16-byte vectors (SSE and NEON width). A single 64-byte vector type
// would be simpler to write, but GCC keeps such a variable in memory on targets without 64-byte
// registers (checked on aarch64: the hot loop stored and reloaded it through the stack).
// Lanes 0-3 are in a, 4-7 in b, 8-11 in c, 12-15 in d. Every operation is lane by lane.
typedef float f4v __attribute__((vector_size(16)));
typedef std::int32_t i4v __attribute__((vector_size(16)));

struct V16 {
    f4v a, b, c, d;
};

inline __attribute__((always_inline)) V16 zero16()
{
    const f4v z = {0, 0, 0, 0};
    return V16{z, z, z, z};
}

inline __attribute__((always_inline)) f4v load4(const float *p)
{
    f4v v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline __attribute__((always_inline)) V16 load(const float *p)
{
    return V16{load4(p), load4(p + 4), load4(p + 8), load4(p + 12)};
}

// Lanes added in one fixed order: j + (j + 8), then j + (j + 4), j + (j + 2), 0 + 1.
inline __attribute__((always_inline)) float hsum(const V16 &v)
{
    const f4v s8a = v.a + v.c, s8b = v.b + v.d;       // lanes 0-7 of j + (j + 8)
    const f4v s4 = s8a + s8b;                          // j + (j + 4)
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

enum Kind { DOT, L2SQ, L1 };

template <Kind K> inline __attribute__((always_inline)) f4v step4(f4v acc, f4v x, f4v q)
{
    if (K == DOT) return acc + x * q;
    const f4v d = x - q;
    if (K == L2SQ) return acc + d * d;
    return acc + (f4v)((i4v)d & 0x7fffffff);         // |d|: clear the sign bit
}

template <Kind K> inline __attribute__((always_inline)) V16 step(const V16 &acc, const V16 &x, const V16 &q)
{
    return V16{step4<K>(acc.a, x.a, q.a), step4<K>(acc.b, x.b, q.b), step4<K>(acc.c, x.c, q.c), step4<K>(acc.d, x.d, q.d)};
}

template <Kind K> inline __attribute__((always_inline)) float pair(const float *a, const float *b, std::uint32_t stride)
{
    V16 acc = zero16();
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

// Four queries against one row. Four 16-lane accumulators do not fit in the 32 NEON registers
// next to the row, so each row is done in two halves: lanes 0-7 (a, b) over the whole row, then
// lanes 8-15 (c, d). Every lane sees exactly the operations of pair(), in the same order.
template <Kind K> inline void rows_4q(const float *rows, std::uint64_t n, std::uint32_t stride, const float *const q[4],
                                      float *keys)
{
    const float *q0 = q[0], *q1 = q[1], *q2 = q[2], *q3 = q[3];
    const f4v z = {0, 0, 0, 0};
    for (std::uint64_t i = 0; i < n; ++i) {
        const float *r = rows + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        V16 a0, a1, a2, a3;
        for (int half = 0; half < 2; ++half) {
            const std::uint32_t o = half * 8;
            f4v x0 = z, x1 = z, y0 = z, y1 = z, u0 = z, u1 = z, w0 = z, w1 = z;
            for (std::uint32_t j = o; j < stride; j += LANES) {
                const f4v ra = load4(r + j), rb = load4(r + j + 4);
                x0 = step4<K>(x0, ra, load4(q0 + j)); x1 = step4<K>(x1, rb, load4(q0 + j + 4));
                y0 = step4<K>(y0, ra, load4(q1 + j)); y1 = step4<K>(y1, rb, load4(q1 + j + 4));
                u0 = step4<K>(u0, ra, load4(q2 + j)); u1 = step4<K>(u1, rb, load4(q2 + j + 4));
                w0 = step4<K>(w0, ra, load4(q3 + j)); w1 = step4<K>(w1, rb, load4(q3 + j + 4));
            }
            if (half == 0) { a0.a = x0; a0.b = x1; a1.a = y0; a1.b = y1; a2.a = u0; a2.b = u1; a3.a = w0; a3.b = w1; }
            else { a0.c = x0; a0.d = x1; a1.c = y0; a1.d = y1; a2.c = u0; a2.d = u1; a3.c = w0; a3.d = w1; }
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
