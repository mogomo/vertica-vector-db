#include "sq8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vvector {

namespace {

std::uint64_t align64(std::uint64_t v) { return (v + 63) / 64 * 64; }

struct Sq8Layout {
    std::uint64_t codes, sums, total;
};

// The parts are sized by the layout's capacity (the snapshot's, FLAG_CAPACITY), so appended rows
// do not move the sums.
Sq8Layout sq8_layout(std::uint64_t capacity, std::uint32_t row_stride)
{
    Sq8Layout l;
    l.codes = SQ8_HEADER_BYTES;
    l.sums = align64(l.codes + capacity * row_stride);
    l.total = align64(l.sums + capacity * 4);
    return l;
}

void sq8_fail(const std::string &why) { throw std::runtime_error("bad snapshot sq8 section: " + why); }

} // namespace

std::uint64_t sq8_section_bytes(std::uint64_t capacity, std::uint32_t row_stride)
{
    return sq8_layout(capacity, row_stride).total;
}

Sq8Range sq8_train(const VectorSet &s)
{
    Sq8Range r;
    if (s.count == 0 || s.dims == 0) return r;
    const std::uint64_t rows = std::min<std::uint64_t>(s.count, (SQ8_SAMPLE + s.dims - 1) / s.dims);
    std::vector<float> v;
    v.reserve(rows * s.dims);
    for (std::uint64_t j = 0; j < rows; ++j) {
        const float *row = s.vector(j * s.count / rows);
        for (std::uint32_t d = 0; d < s.dims; ++d)
            if (row[d] == row[d]) v.push_back(row[d]);      // NaN never counts
    }
    r.sample = static_cast<std::uint32_t>(v.size());
    if (v.empty()) return r;
    const std::size_t n = v.size();
    const std::size_t ilo = static_cast<std::size_t>(std::floor(0.001 * double(n - 1)));
    const std::size_t ihi = static_cast<std::size_t>(std::ceil(0.999 * double(n - 1)));
    std::nth_element(v.begin(), v.begin() + ilo, v.end());
    const float lo = v[ilo];
    std::nth_element(v.begin(), v.begin() + ihi, v.end());
    const float hi = v[ihi];
    r.offset = lo;
    r.scale = hi > lo ? (hi - lo) / 255.0f : 1.0f;
    if (!(r.scale > 0) || !std::isfinite(r.scale)) r.scale = 1.0f;   // a range beyond the float limit
    return r;
}

std::uint32_t sq8_encode(const Sq8Range &r, const float *v, std::uint32_t dims, std::uint8_t *code)
{
    std::uint32_t sum = 0;
    for (std::uint32_t d = 0; d < dims; ++d) {
        const float t = (v[d] - r.offset) / r.scale + 0.5f;
        const std::uint8_t c = !(t > 0.0f) ? 0 : t >= 255.0f ? 255 : static_cast<std::uint8_t>(std::floor(t));
        code[d] = c;
        sum += c;
    }
    return sum;
}

void sq8_fill(const VectorSet &s, const Sq8Range &r, std::uint8_t *section, std::uint64_t first)
{
    const Sq8Layout l = sq8_layout(s.capacity, s.row_stride);
    Sq8Header h{};
    h.scale = r.scale;
    h.offset = r.offset;
    h.code_stride = s.row_stride;
    h.sample = r.sample;
    h.count = s.count;
    std::memcpy(section, &h, sizeof(h));
    std::uint8_t *codes = section + l.codes;
    std::uint32_t *sums = reinterpret_cast<std::uint32_t *>(section + l.sums);
    for (std::uint64_t i = first; i < s.count; ++i)
        sums[i] = sq8_encode(r, s.vector(i), s.dims, codes + i * s.row_stride);
}

CodeSection sq8_code_section()
{
    CodeSection c;
    c.bytes = [](std::uint64_t capacity, std::uint32_t row_stride) { return sq8_section_bytes(capacity, row_stride); };
    c.fill = [](const VectorSet &s, std::uint8_t *section) { sq8_fill(s, sq8_train(s), section, 0); };
    return c;
}

void sq8_extend(const Sq8Codes &base, const VectorSet &s, std::uint8_t *section, bool in_place)
{
    if (base.stride != s.row_stride || base.count > s.count) throw std::logic_error("sq8_extend: the snapshot does not extend the base");
    if (!in_place) {
        const Sq8Layout l = sq8_layout(s.capacity, s.row_stride);
        std::memcpy(section + l.codes, base.codes, base.count * base.stride);
        std::memcpy(section + l.sums, base.sums, base.count * 4);
    }
    sq8_fill(s, base.range, section, base.count);
}

Sq8Codes sq8_open(const VectorSet &s, bool verify)
{
    if (!(s.flags & FLAG_SQ8) || !s.sq8) sq8_fail("the snapshot has no sq8 section");
    const Sq8Layout l = sq8_layout(s.capacity, s.row_stride);
    if (s.sq8_bytes != l.total) sq8_fail("size does not match the vectors");
    Sq8Header h;
    std::memcpy(&h, s.sq8, sizeof(h));
    if (h.count != s.count) sq8_fail("count does not match the snapshot");
    if (h.code_stride != s.row_stride) sq8_fail("code_stride does not match the snapshot");
    if (!(h.scale > 0) || !std::isfinite(h.scale) || !std::isfinite(h.offset)) sq8_fail("scale or offset out of range");
    for (std::uint64_t w : h.reserved)
        if (w) sq8_fail("reserved header fields are not 0");

    Sq8Codes c;
    c.range.scale = h.scale;
    c.range.offset = h.offset;
    c.range.sample = h.sample;
    c.dims = s.dims;
    c.stride = h.code_stride;
    c.count = h.count;
    c.codes = s.sq8 + l.codes;
    c.sums = reinterpret_cast<const std::uint32_t *>(s.sq8 + l.sums);
    if (verify) {
        for (std::uint64_t i = 0; i < c.count; ++i) {
            const std::uint8_t *row = c.row(i);
            std::uint32_t sum = 0;
            for (std::uint32_t d = 0; d < s.dims; ++d) sum += row[d];
            for (std::uint32_t d = s.dims; d < c.stride; ++d)
                if (row[d]) sq8_fail("padding codes of position " + std::to_string(i) + " are not 0");
            if (sum != c.sums[i]) sq8_fail("the code sum of position " + std::to_string(i) + " is wrong");
        }
    }
    return c;
}

} // namespace vvector
