#include "kernels.h"

#include <cstring>

namespace vvector {

namespace {

// 16 lanes held as two groups of 8 (L8), each made of native vectors. A single 64-byte vector type
// would be simpler to write, but GCC keeps such a variable in memory on targets without 64-byte
// registers (checked on aarch64: the hot loop stored and reloaded it through the stack). GCC never
// widens vector code either, so the native width is chosen here per architecture:
// - aarch64 (NEON) and others: an L8 is two 16-byte vectors (lanes 0-3 in a, 4-7 in b);
// - x86_64: an L8 is one 32-byte vector: one register in the avx2 and avx512f copies, two SSE
//   registers in the default copy. With 16-byte vectors the AVX copies used half their width.
// Every operation is lane by lane and the lanes are added in one fixed order, so the width changes
// the speed only, never a result.
typedef float f4v __attribute__((vector_size(16)));

#if defined(__x86_64__)
typedef float f8v __attribute__((vector_size(32)));
typedef std::int32_t i8v __attribute__((vector_size(32)));
// The same vector at any float address. g++ 8 copies a memcpy into a 32-byte vector through the
// stack; a load through this type is one unaligned load. GCC vector types alias their element type.
typedef float f8u __attribute__((vector_size(32), aligned(4)));

struct L8 {
    f8v a;
};

inline __attribute__((always_inline)) L8 zero8()
{
    const f8v z = {0, 0, 0, 0, 0, 0, 0, 0};
    return L8{z};
}

inline __attribute__((always_inline)) L8 load8(const float *p) { return L8{*reinterpret_cast<const f8u *>(p)}; }

inline __attribute__((always_inline)) L8 add8(const L8 &x, const L8 &y) { return L8{x.a + y.a}; }

// Lanes 0-3 plus lanes 4-7.
inline __attribute__((always_inline)) f4v fold8(const L8 &v)
{
    f4v lo, hi;
    std::memcpy(&lo, &v.a, sizeof(lo));
    std::memcpy(&hi, reinterpret_cast<const char *>(&v.a) + sizeof(lo), sizeof(hi));
    return lo + hi;
}
#else
typedef std::int32_t i4v __attribute__((vector_size(16)));

struct L8 {
    f4v a, b;
};

inline __attribute__((always_inline)) L8 zero8()
{
    const f4v z = {0, 0, 0, 0};
    return L8{z, z};
}

inline __attribute__((always_inline)) f4v load4(const float *p)
{
    f4v v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline __attribute__((always_inline)) L8 load8(const float *p) { return L8{load4(p), load4(p + 4)}; }

inline __attribute__((always_inline)) L8 add8(const L8 &x, const L8 &y) { return L8{x.a + y.a, x.b + y.b}; }

// Lanes 0-3 plus lanes 4-7.
inline __attribute__((always_inline)) f4v fold8(const L8 &v) { return v.a + v.b; }
#endif

// Lanes 0-7 in lo, 8-15 in hi.
struct V16 {
    L8 lo, hi;
};

inline __attribute__((always_inline)) V16 zero16() { return V16{zero8(), zero8()}; }

inline __attribute__((always_inline)) V16 load(const float *p) { return V16{load8(p), load8(p + 8)}; }

// Lanes added in one fixed order: j + (j + 8), then j + (j + 4), j + (j + 2), 0 + 1.
inline __attribute__((always_inline)) float hsum(const V16 &v)
{
    const f4v s4 = fold8(add8(v.lo, v.hi));            // j + (j + 8), then j + (j + 4)
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

enum Kind { DOT, L2SQ, L1 };

// acc += the term of x and q. Vectors and L8 go by reference: g++ 8 warns (-Wpsabi) when a 32-byte
// vector is passed by value in the default (SSE) copy, although this is always inlined.
template <Kind K, typename F, typename I> inline __attribute__((always_inline)) void step_v(F &acc, const F &x, const F &q)
{
    if (K == DOT) { acc += x * q; return; }
    const F d = x - q;
    if (K == L2SQ) { acc += d * d; return; }
    acc += (F)((I)d & 0x7fffffff);                   // |d|: clear the sign bit
}

#if defined(__x86_64__)
template <Kind K> inline __attribute__((always_inline)) L8 step8(const L8 &acc, const L8 &x, const L8 &q)
{
    L8 r = acc;
    step_v<K, f8v, i8v>(r.a, x.a, q.a);
    return r;
}
#else
template <Kind K> inline __attribute__((always_inline)) L8 step8(const L8 &acc, const L8 &x, const L8 &q)
{
    L8 r = acc;
    step_v<K, f4v, i4v>(r.a, x.a, q.a);
    step_v<K, f4v, i4v>(r.b, x.b, q.b);
    return r;
}
#endif

template <Kind K> inline __attribute__((always_inline)) V16 step(const V16 &acc, const V16 &x, const V16 &q)
{
    return V16{step8<K>(acc.lo, x.lo, q.lo), step8<K>(acc.hi, x.hi, q.hi)};
}

template <Kind K> inline __attribute__((always_inline)) float pair(const float *a, const float *b, std::uint32_t stride)
{
    V16 acc = zero16();
    for (std::uint32_t j = 0; j < stride; j += LANES) acc = step<K>(acc, load(a + j), load(b + j));
    return hsum(acc);
}

// dot and cosine keys are minus the dot product.
template <Kind K> inline __attribute__((always_inline)) float key_of(float v) { return K == DOT ? -v : v; }

template <Kind K> inline __attribute__((always_inline)) void rows_1q(const float *rows, std::uint64_t n, std::uint32_t stride, const float *q, float *keys)
{
    for (std::uint64_t i = 0; i < n; ++i) {
        const float *r = rows + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        keys[i] = key_of<K>(pair<K>(r, q, stride));
    }
}

// Prefetches a whole row, one cache line at a time.
inline __attribute__((always_inline)) void prefetch_row(const float *r, std::uint32_t stride)
{
    for (std::uint32_t j = 0; j < stride; j += 16) __builtin_prefetch(r + j);
}

template <Kind K> inline __attribute__((always_inline)) void rows_gather(const float *rows, std::uint32_t stride, const std::uint32_t *pos,
                                          std::uint32_t n, const float *q, float *keys)
{
    const std::uint32_t ahead = 2;
    for (std::uint32_t i = 0; i < n && i < ahead; ++i) prefetch_row(rows + std::uint64_t(pos[i]) * stride, stride);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i + ahead < n) prefetch_row(rows + std::uint64_t(pos[i + ahead]) * stride, stride);
        keys[i] = key_of<K>(pair<K>(rows + std::uint64_t(pos[i]) * stride, q, stride));
    }
}

// Four queries against one row. Four 16-lane accumulators do not fit in the 32 NEON registers
// next to the row, so each row is done in two halves: lanes 0-7 (lo) over the whole row, then
// lanes 8-15 (hi). Every lane sees exactly the operations of pair(), in the same order.
template <Kind K> inline __attribute__((always_inline)) void rows_4q(const float *rows, std::uint64_t n, std::uint32_t stride, const float *const q[4],
                                      float *keys)
{
    const float *q0 = q[0], *q1 = q[1], *q2 = q[2], *q3 = q[3];
    for (std::uint64_t i = 0; i < n; ++i) {
        const float *r = rows + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        V16 a0, a1, a2, a3;
        for (int half = 0; half < 2; ++half) {
            const std::uint32_t o = half * 8;
            L8 x = zero8(), y = zero8(), u = zero8(), w = zero8();
            for (std::uint32_t j = o; j < stride; j += LANES) {
                const L8 rj = load8(r + j);
                x = step8<K>(x, rj, load8(q0 + j));
                y = step8<K>(y, rj, load8(q1 + j));
                u = step8<K>(u, rj, load8(q2 + j));
                w = step8<K>(w, rj, load8(q3 + j));
            }
            if (half == 0) { a0.lo = x; a1.lo = y; a2.lo = u; a3.lo = w; }
            else { a0.hi = x; a1.hi = y; a2.hi = u; a3.hi = w; }
        }
        keys[i] = key_of<K>(hsum(a0));
        keys[n + i] = key_of<K>(hsum(a1));
        keys[2 * n + i] = key_of<K>(hsum(a2));
        keys[3 * n + i] = key_of<K>(hsum(a3));
    }
}

// sq8: an integer sum over one pair of code rows (a: the row, b: the query). Plain loops: integer
// sums do not depend on the order of the additions, so the compiler may vectorise them as it likes.
// The loops are written in the forms GCC turns into one multiply-add of 16-bit pairs (pmaddwd): with
// 32-bit products g++ 8.5 used 32-bit multiplies at a third of the speed. l2: the difference in 16
// bits, squared. dot: g++ uses pmaddwd only when a factor is signed, so the row code is shifted by
// 128 and the sum corrected exactly: sum a x b = sum (a - 128) x b + 128 x sum b, where sum b (over
// the stride, padding 0) is the query's, computed once per query (bsum). l1: the sum of absolute
// differences (psadbw). Every sum fits: 255^2 x 32768 < 2^31, and the dot parts are below 2^30 each.
template <Kind K> inline __attribute__((always_inline)) std::uint32_t isum(const std::uint8_t *a, const std::uint8_t *b,
                                                                           std::uint32_t stride, std::uint32_t bsum)
{
    std::int32_t s = 0;
    for (std::uint32_t j = 0; j < stride; ++j) {
        if (K == L1) {
            const std::int32_t d = std::int32_t(a[j]) - std::int32_t(b[j]);
            s += d < 0 ? -d : d;
        } else if (K == L2SQ) {
            const std::int16_t d = static_cast<std::int16_t>(std::int16_t(a[j]) - std::int16_t(b[j]));
            s += std::int32_t(d) * std::int32_t(d);
        } else {
            const std::int16_t x = static_cast<std::int16_t>(std::int16_t(a[j]) - 128), y = b[j];
            s += std::int32_t(x) * std::int32_t(y);
        }
    }
    if (K == DOT) return static_cast<std::uint32_t>(std::int64_t(s) + 128 * std::int64_t(bsum));
    return static_cast<std::uint32_t>(s);
}

// The sum of the codes of a query over the whole stride (dot only; the padding is 0).
template <Kind K> inline __attribute__((always_inline)) std::uint32_t qsum(const std::uint8_t *q, std::uint32_t stride)
{
    std::uint32_t t = 0;
    if (K == DOT)
        for (std::uint32_t j = 0; j < stride; ++j) t += q[j];
    return t;
}

template <Kind K> inline __attribute__((always_inline)) void codes_1q(const std::uint8_t *codes, std::uint64_t n,
                                                                    std::uint32_t stride, const std::uint8_t *q,
                                                                    std::uint32_t *sums)
{
    const std::uint32_t qs = qsum<K>(q, stride);
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint8_t *r = codes + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        sums[i] = isum<K>(r, q, stride, qs);
    }
}

// Four queries against each row: the row is read once. Integer sums, so each equals isum's.
template <Kind K> inline __attribute__((always_inline)) void codes_4q(const std::uint8_t *codes, std::uint64_t n,
                                                                    std::uint32_t stride, const std::uint8_t *const q[4],
                                                                    std::uint32_t *sums)
{
    const std::uint8_t *q0 = q[0], *q1 = q[1], *q2 = q[2], *q3 = q[3];
    const std::uint32_t s0 = qsum<K>(q0, stride), s1 = qsum<K>(q1, stride), s2 = qsum<K>(q2, stride), s3 = qsum<K>(q3, stride);
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint8_t *r = codes + i * stride;
        __builtin_prefetch(r + 8 * static_cast<std::uint64_t>(stride));
        sums[i] = isum<K>(r, q0, stride, s0);
        sums[n + i] = isum<K>(r, q1, stride, s1);
        sums[2 * n + i] = isum<K>(r, q2, stride, s2);
        sums[3 * n + i] = isum<K>(r, q3, stride, s3);
    }
}

// Prefetches a code row, one cache line at a time.
inline __attribute__((always_inline)) void prefetch_codes(const std::uint8_t *r, std::uint32_t stride)
{
    for (std::uint32_t j = 0; j < stride; j += 64) __builtin_prefetch(r + j);
}

template <Kind K> inline __attribute__((always_inline)) void codes_gather(const std::uint8_t *codes, std::uint32_t stride, const std::uint32_t *pos,
                                           std::uint32_t n, const std::uint8_t *q, std::uint32_t *sums)
{
    const std::uint32_t qs = qsum<K>(q, stride);
    const std::uint32_t ahead = 2;
    for (std::uint32_t i = 0; i < n && i < ahead; ++i) prefetch_codes(codes + std::uint64_t(pos[i]) * stride, stride);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i + ahead < n) prefetch_codes(codes + std::uint64_t(pos[i + ahead]) * stride, stride);
        sums[i] = isum<K>(codes + std::uint64_t(pos[i]) * stride, q, stride, qs);
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
// .so runs on every node. The scores do not depend on the copy (see kernels.h). Only the marked
// function is copied: a helper it calls without inlining is compiled once, for the default target
// (g++ 8.5 stopped inlining rows_4q, and the AVX copies of keys_4q ran on SSE at half the speed).
// So every helper below the entry points is always_inline.
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

VV_CLONES void keys_gather(Metric m, const float *rows, std::uint32_t stride, const std::uint32_t *pos, std::uint32_t n,
                           const float *q, float *keys)
{
    switch (kind_of(m)) {
    case L2SQ: rows_gather<L2SQ>(rows, stride, pos, n, q, keys); break;
    case L1: rows_gather<L1>(rows, stride, pos, n, q, keys); break;
    default: rows_gather<DOT>(rows, stride, pos, n, q, keys); break;
    }
}

VV_CLONES void sq8_sums_1q(Metric m, const std::uint8_t *codes, std::uint64_t n, std::uint32_t stride,
                           const std::uint8_t *q, std::uint32_t *sums)
{
    switch (kind_of(m)) {
    case L2SQ: codes_1q<L2SQ>(codes, n, stride, q, sums); break;
    case L1: codes_1q<L1>(codes, n, stride, q, sums); break;
    default: codes_1q<DOT>(codes, n, stride, q, sums); break;
    }
}

VV_CLONES void sq8_sums_4q(Metric m, const std::uint8_t *codes, std::uint64_t n, std::uint32_t stride,
                           const std::uint8_t *const q[4], std::uint32_t *sums)
{
    switch (kind_of(m)) {
    case L2SQ: codes_4q<L2SQ>(codes, n, stride, q, sums); break;
    case L1: codes_4q<L1>(codes, n, stride, q, sums); break;
    default: codes_4q<DOT>(codes, n, stride, q, sums); break;
    }
}

VV_CLONES void sq8_sums_gather(Metric m, const std::uint8_t *codes, std::uint32_t stride, const std::uint32_t *pos,
                               std::uint32_t n, const std::uint8_t *q, std::uint32_t *sums)
{
    switch (kind_of(m)) {
    case L2SQ: codes_gather<L2SQ>(codes, stride, pos, n, q, sums); break;
    case L1: codes_gather<L1>(codes, stride, pos, n, q, sums); break;
    default: codes_gather<DOT>(codes, stride, pos, n, q, sums); break;
    }
}

VV_CLONES std::uint32_t sq8_sum(Metric m, const std::uint8_t *a, const std::uint8_t *b, std::uint32_t stride)
{
    switch (kind_of(m)) {
    case L2SQ: return isum<L2SQ>(a, b, stride, 0);
    case L1: return isum<L1>(a, b, stride, 0);
    default: return isum<DOT>(a, b, stride, qsum<DOT>(b, stride));
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
