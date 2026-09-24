// vvector engine: distance kernels. Portable SIMD through GCC vector extensions, no intrinsics.
//
// Every kernel works on rows of `stride` floats, a multiple of 16, zero-padded after dims. A row is
// processed 16 elements at a time into one 16-lane accumulator (lane j holds elements j, j+16, ...),
// and the lanes are added in one fixed order. With -ffp-contract=off (no fused multiply-add) every
// result is bit-identical on aarch64 and on x86_64 with SSE, AVX2 or AVX-512, for any number of
// threads and any batch tiling. The batch kernels give exactly the scores of the single kernels.
//
// Keys order the candidates: smaller is closer. l2: squared distance. l1: distance.
// dot and cosine: minus the dot product (cosine rows and queries have unit length).
#ifndef VVECTOR_ENGINE_KERNELS_H
#define VVECTOR_ENGINE_KERNELS_H

#include "snapshot.h"

#include <cmath>
#include <cstdint>

namespace vvector {

float dot(const float *a, const float *b, std::uint32_t stride);
float l2sq(const float *a, const float *b, std::uint32_t stride);
float l1(const float *a, const float *b, std::uint32_t stride);

// The key of one pair.
float distance_key(Metric m, const float *a, const float *b, std::uint32_t stride);

// The keys of n rows (stride floats apart) against one query: keys[i].
void keys_1q(Metric m, const float *rows, std::uint64_t n, std::uint32_t stride, const float *q, float *keys);

// The keys of n rows against 4 queries: keys[j * n + i] for query j. Each row is read once.
void keys_4q(Metric m, const float *rows, std::uint64_t n, std::uint32_t stride, const float *const q[4],
             float *keys);

// The keys of n rows at scattered positions against one query: keys[i] for the row
// rows + pos[i] * stride. For graph search: the next rows are prefetched while one is scored.
// Each key equals distance_key of the same pair bit for bit.
void keys_gather(Metric m, const float *rows, std::uint32_t stride, const std::uint32_t *pos, std::uint32_t n,
                 const float *q, float *keys);

// sq8 codes (sq8.h): integer sums of code rows of `stride` bytes (padding 0) against one query's
// codes: sum (a - b)^2 for l2, sum abs(a - b) for l1, sum a x b for dot and cosine. Integer sums
// are exact, so every CPU and every vector width gives the same result.
// n consecutive rows: sums[i] for the row codes + i * stride.
void sq8_sums_1q(Metric m, const std::uint8_t *codes, std::uint64_t n, std::uint32_t stride, const std::uint8_t *q,
                 std::uint32_t *sums);
// n consecutive rows against 4 queries: sums[j * n + i] for query j. Each row is read once.
void sq8_sums_4q(Metric m, const std::uint8_t *codes, std::uint64_t n, std::uint32_t stride, const std::uint8_t *const q[4],
                 std::uint32_t *sums);
// n rows at scattered positions: sums[i] for the row codes + pos[i] * stride (the graph search).
void sq8_sums_gather(Metric m, const std::uint8_t *codes, std::uint32_t stride, const std::uint32_t *pos,
                     std::uint32_t n, const std::uint8_t *q, std::uint32_t *sums);
// One pair.
std::uint32_t sq8_sum(Metric m, const std::uint8_t *a, const std::uint8_t *b, std::uint32_t stride);

// The value the built-in function of the metric returns: VECTOR_L2, COSINE_SIMILARITY, DOT_PRODUCT,
// or the Manhattan distance.
inline float key_to_score(Metric m, float key)
{
    switch (m) {
    case Metric::L2: return std::sqrt(key);
    case Metric::L1: return key;
    default: return -key;
    }
}

// Scales the first dims floats of v to unit length (computed in double, one fixed order).
// A zero vector stays zero: its cosine with anything is 0, as COSINE_SIMILARITY returns.
void normalize(float *v, std::uint32_t dims);

// Which code the x86_64 dispatcher picked ("avx512f", "avx2", "default"), or "neon" on aarch64,
// or "generic". For vversion and the benchmark.
const char *kernel_target();

} // namespace vvector

#endif
