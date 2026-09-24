// vvector engine: scalar quantisation (sq8). Every element of every row is also stored as one byte,
// so a search reads a quarter of the bytes while it ranks candidates, then computes the exact
// scores of the best candidates from the float rows (rescoring). docs/format.md "sq8 section".
//
// Section, every part on a 64-byte boundary from the section start:
//   header  64 bytes, Sq8Header
//   codes   uint8[count x code_stride]: code_stride = row_stride; padding codes are 0
//   sums    uint32[count]: the sum of the codes of each row
//
// One range for the whole index: code = min(255, max(0, floor((x - offset) / scale + 0.5))) in
// float32. The keys computed from the codes are integer sums, so they are the same on every CPU.
#ifndef VVECTOR_ENGINE_SQ8_H
#define VVECTOR_ENGINE_SQ8_H

#include "snapshot.h"

#include <cstdint>
#include <functional>

namespace vvector {

constexpr std::uint64_t SQ8_HEADER_BYTES = 64;
constexpr std::uint32_t SQ8_SAMPLE = 100000;     // elements the range is trained on (about)

struct Sq8Header {
    float scale;                   // (hi - lo) / 255, or 1 when hi = lo
    float offset;                  // lo
    std::uint32_t code_stride;     // bytes per code row: the snapshot's row_stride
    std::uint32_t sample;          // elements the range was trained on
    std::uint64_t count;           // positions, as in the snapshot header
    std::uint64_t reserved[5];
};
static_assert(sizeof(Sq8Header) == SQ8_HEADER_BYTES, "sq8 header must be 64 bytes");

// The range of the codes: element x has code round((x - offset) / scale), clipped to 0 .. 255.
struct Sq8Range {
    float scale = 1.0f;
    float offset = 0.0f;
    std::uint32_t sample = 0;
};

// The sq8 section of a snapshot, opened for reading.
struct Sq8Codes {
    Sq8Range range;
    std::uint32_t dims = 0;        // elements per vector
    std::uint32_t stride = 0;      // bytes per code row
    std::uint64_t count = 0;
    const std::uint8_t *codes = nullptr;
    const std::uint32_t *sums = nullptr;

    const std::uint8_t *row(std::uint64_t i) const { return codes + i * stride; }
};

// Bytes of the sq8 section for count rows of row_stride elements.
std::uint64_t sq8_section_bytes(std::uint64_t count, std::uint32_t row_stride);

// Trains the range on the float rows of s: lo and hi are the 0.001 and 0.999 quantiles of all
// elements of about SQ8_SAMPLE / dims evenly spaced rows (in position order). The same rows give
// the same range.
Sq8Range sq8_train(const VectorSet &s);

// Codes the first dims floats of v into code (stride bytes, the padding already 0) and returns the
// sum of the codes. A NaN element gets code 0.
std::uint32_t sq8_encode(const Sq8Range &r, const float *v, std::uint32_t dims, std::uint8_t *code);

// Writes the header and the codes and sums of positions first .. s.count - 1 into section
// (sq8_section_bytes bytes, zero-filled; the parts of the positions before first are already there:
// an incremental build copies them from its base).
void sq8_fill(const VectorSet &s, const Sq8Range &r, std::uint8_t *section, std::uint64_t first = 0);

// The sq8 section for SnapshotBuilder::finish: trains the range on the finished rows, then fills.
CodeSection sq8_code_section();

// Incremental build (delta.h): writes the sq8 section of s (sq8_section_bytes bytes, zero-filled):
// the codes and sums of the base's positions copied, the positions base.count .. s.count - 1
// coded with the base's range. s is the new snapshot: the base's positions first, same order.
void sq8_extend(const Sq8Codes &base, const VectorSet &s, std::uint8_t *section);

// Opens the sq8 section of a snapshot with FLAG_SQ8. Throws std::runtime_error with the cause.
// verify reads every row (vload): padding codes 0, sums right; without it only the header is checked.
Sq8Codes sq8_open(const VectorSet &s, bool verify);

// The key of a code sum, as the float keys (kernels.h): smaller is closer, and key_to_score gives the
// approximate value of the metric's built-in (what vsearch reports with rescore=false). sum is
// sum (a - b)^2 for l2, sum abs(a - b) for l1, sum a x b for dot and cosine; row_sum and query_sum
// are the code sums of the row and the query (dot and cosine only). Computed in double in one fixed
// order from exact integers, so it is the same on every CPU:
//   l2: scale^2 x sum;  l1: scale x sum;
//   dot, cosine: -(scale^2 x sum + scale x offset x (row_sum + query_sum) + dims x offset^2).
inline float sq8_key(Metric m, const Sq8Range &r, std::uint32_t sum, std::uint32_t row_sum, std::uint32_t query_sum,
                     std::uint32_t dims)
{
    const double s = r.scale, o = r.offset;
    switch (m) {
    case Metric::L2: return static_cast<float>(s * s * sum);
    case Metric::L1: return static_cast<float>(s * sum);
    default:
        return static_cast<float>(-(s * s * sum + s * o * (double(row_sum) + double(query_sum)) + double(dims) * o * o));
    }
}

} // namespace vvector

#endif
