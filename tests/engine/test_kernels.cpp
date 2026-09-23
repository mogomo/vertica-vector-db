// Distance kernels: agreement with a plain double loop, bit-identical batch, gather and single kernels,
// hand-made cases, zero vectors, and a fingerprint of the result bits that must be the same on
// every machine (aarch64, x86_64 with any instruction set).
#include "check.h"

#include "../../src/engine/kernels.h"

#include <cmath>
#include <cstring>

using namespace vvector;

// Rows of `stride` floats, 64-byte aligned, zero padding after dims.
struct Rows {
    std::uint32_t dims, stride;
    std::uint64_t n;
    std::vector<float> data;
    Rows(std::uint64_t n, std::uint32_t dims, std::uint64_t seed) : dims(dims), stride(row_stride_for(dims)), n(n)
    {
        data.assign(n * stride + 16, 0.0f);
        for (std::uint64_t i = 0; i < n; ++i) test_vector(i, dims, seed, true, row(i));
    }
    float *row(std::uint64_t i) { return data.data() + i * stride; }
};

static double ref_key(Metric m, const float *a, const float *b, std::uint32_t dims)
{
    double s = 0;
    for (std::uint32_t d = 0; d < dims; ++d) {
        const double x = a[d], y = b[d];
        if (m == Metric::L2) s += (x - y) * (x - y);
        else if (m == Metric::L1) s += std::fabs(x - y);
        else s += x * y;
    }
    return m == Metric::L2 || m == Metric::L1 ? s : -s;
}

static std::uint64_t fnv(std::uint64_t h, float f)
{
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    for (int i = 0; i < 4; ++i) { h ^= (bits >> (8 * i)) & 0xFF; h *= 0x100000001B3ull; }
    return h;
}

int main()
{
    const Metric metrics[4] = {Metric::L2, Metric::Cosine, Metric::Dot, Metric::L1};
    const std::uint32_t dims_list[7] = {1, 15, 16, 17, 100, 768, 1536};
    std::uint64_t fingerprint = 0xCBF29CE484222325ull;

    for (std::uint32_t dims : dims_list) {
        Rows rows(37, dims, dims), queries(4, dims, dims + 1000);
        for (Metric m : metrics) {
            if (m == Metric::Cosine) {
                for (std::uint64_t i = 0; i < rows.n; ++i) normalize(rows.row(i), dims);
                for (std::uint64_t i = 0; i < queries.n; ++i) normalize(queries.row(i), dims);
            }
            const float *q[4] = {queries.row(0), queries.row(1), queries.row(2), queries.row(3)};
            std::vector<float> k1(rows.n), k4(4 * rows.n);
            keys_4q(m, rows.data.data(), rows.n, rows.stride, q, k4.data());
            bool close = true, same = true;
            // Scattered positions (graph search): every row once, in a mixed order.
            std::vector<std::uint32_t> pos(rows.n);
            for (std::uint64_t i = 0; i < rows.n; ++i) pos[i] = static_cast<std::uint32_t>((i * 17) % rows.n);
            std::vector<float> kg(rows.n);
            for (int j = 0; j < 4; ++j) {
                keys_1q(m, rows.data.data(), rows.n, rows.stride, q[j], k1.data());
                keys_gather(m, rows.data.data(), rows.stride, pos.data(), static_cast<std::uint32_t>(rows.n), q[j], kg.data());
                for (std::uint64_t i = 0; i < rows.n; ++i) same = same && kg[i] == k1[pos[i]];
                for (std::uint64_t i = 0; i < rows.n; ++i) {
                    const double ref = ref_key(m, rows.row(i), q[j], dims);
                    // Error bound of a float32 sum over 16 lanes: (terms per lane + 8) units of
                    // rounding, relative to the sum of the magnitudes of the terms.
                    double scale = 0;
                    for (std::uint32_t d = 0; d < dims; ++d) {
                        const double x = rows.row(i)[d], y = q[j][d];
                        scale += m == Metric::L2 ? (x - y) * (x - y) : m == Metric::L1 ? std::fabs(x - y) : std::fabs(x * y);
                    }
                    const double bound = static_cast<double>(rows.stride / 16 + 8) * std::ldexp(1.0, -24) * scale;
                    close = close && std::fabs(k1[i] - ref) <= bound;
                    same = same && k1[i] == k4[j * rows.n + i] && k1[i] == distance_key(m, rows.row(i), q[j], rows.stride);
                    fingerprint = fnv(fingerprint, k1[i]);
                }
            }
            CHECK(close);
            CHECK(same);
            if (!close || !same) std::printf("  dims %u metric %s\n", dims, metric_name(m));
        }
    }
    std::printf("kernel fingerprint %016llx (%s)\n", (unsigned long long)fingerprint, kernel_target());
    // The same bits on every machine. If this fails on a new machine or compiler, the kernels are
    // no longer bit-identical there: check the compiler flags (-ffp-contract=off, no -ffast-math).
    CHECK(fingerprint == 0xe7324c3e28cb2b21ull);

    // Hand-made cases, exact in float.
    {
        alignas(64) float a[16] = {1, 2, 3}, b[16] = {4, -6, 3};
        CHECK(dot(a, b, 16) == 1 * 4 - 2 * 6 + 9);
        CHECK(l2sq(a, b, 16) == 9 + 64 + 0);
        CHECK(l1(a, b, 16) == 3 + 8 + 0);
        CHECK(distance_key(Metric::Dot, a, b, 16) == -1.0f && key_to_score(Metric::Dot, -1.0f) == 1.0f);
        CHECK(distance_key(Metric::L2, a, b, 16) == 73.0f && key_to_score(Metric::L2, 73.0f) == std::sqrt(73.0f));
        CHECK(key_to_score(Metric::L1, 11.0f) == 11.0f && key_to_score(Metric::Cosine, -0.5f) == 0.5f);
    }
    // Zero vectors: cosine 0 (as COSINE_SIMILARITY), distances to themselves 0.
    {
        alignas(64) float z[32] = {}, v[32] = {3, 4};
        normalize(z, 20);
        CHECK(z[0] == 0.0f && z[19] == 0.0f);
        normalize(v, 20);
        CHECK(v[0] == 0.6f && v[1] == 0.8f);
        CHECK(key_to_score(Metric::Cosine, distance_key(Metric::Cosine, z, v, 32)) == 0.0f);
        CHECK(l2sq(v, v, 32) == 0.0f && l1(z, z, 32) == 0.0f);
    }
    // Large values do not turn into NaN: an overflowing distance is +inf, which still orders.
    {
        alignas(64) float a[16] = {3e38f}, b[16] = {-3e38f};
        CHECK(std::isinf(l2sq(a, b, 16)) && l2sq(a, b, 16) > 0);
    }
    std::printf("kernel target: %s\n", kernel_target());
    return finish("test_kernels");
}
