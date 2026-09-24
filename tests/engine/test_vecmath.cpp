// The arithmetic of the vector functions (milestone M5): element operations, scaling, normalising,
// l1 and squared l2 distance against a plain loop, Hamming and Jaccard on packed bits, and the
// element sum and average of many vectors.
#include "check.h"

#include "../../src/engine/vecmath.h"

#include <cmath>

using namespace vvector;

static bool near(double a, double b) { return std::fabs(a - b) <= 1e-12 * std::max(1.0, std::fabs(b)); }

static void test_elementwise()
{
    const double a[] = {1.5, -2, 0, 1e10}, b[] = {0.25, 4, -3, 2};
    double r[4];
    vec_elementwise(ElementOp::Add, a, b, 4, r);
    CHECK(r[0] == 1.75 && r[1] == 2 && r[2] == -3 && r[3] == 1e10 + 2);
    vec_elementwise(ElementOp::Sub, a, b, 4, r);
    CHECK(r[0] == 1.25 && r[1] == -6 && r[2] == 3 && r[3] == 1e10 - 2);
    vec_elementwise(ElementOp::Mul, a, b, 4, r);
    CHECK(r[0] == 0.375 && r[1] == -8 && r[2] == 0 && r[3] == 2e10);
    vec_scale(-2, a, 4, r);
    CHECK(r[0] == -3 && r[1] == 4 && r[2] == 0 && r[3] == -2e10);
    double in_place[] = {1, 2};
    vec_elementwise(ElementOp::Add, in_place, in_place, 2, in_place);
    CHECK(in_place[0] == 2 && in_place[1] == 4);
    vec_elementwise(ElementOp::Add, a, b, 0, r);       // empty vectors: nothing to do
}

static void test_normalize()
{
    const double a[] = {3, 4};
    double r[2];
    vec_normalize(a, 2, r);
    CHECK(near(r[0], 0.6) && near(r[1], 0.8));
    const double z[] = {0, 0, 0};
    double rz[3] = {1, 1, 1};
    vec_normalize(z, 3, rz);
    CHECK(rz[0] == 0 && rz[1] == 0 && rz[2] == 0);
}

static void test_distances()
{
    Rng rng(3);
    for (std::uint64_t n : {1u, 7u, 128u, 1536u}) {
        std::vector<double> a(n), b(n);
        for (std::uint64_t i = 0; i < n; ++i) { a[i] = rng.sym() * 100; b[i] = rng.sym() * 100; }
        double l1 = 0, l2 = 0;
        for (std::uint64_t i = 0; i < n; ++i) { l1 += std::fabs(a[i] - b[i]); l2 += (a[i] - b[i]) * (a[i] - b[i]); }
        CHECK(near(vec_l1(a.data(), b.data(), n), l1));
        CHECK(near(vec_l2sq(a.data(), b.data(), n), l2));
        CHECK(vec_l1(a.data(), a.data(), n) == 0 && vec_l2sq(a.data(), a.data(), n) == 0);
    }
}

static void test_bits()
{
    // Elements 0 and 1: positions that differ, and |both| / |either|.
    const std::int64_t x[] = {1, 0, 1, 1, 0}, y[] = {1, 1, 0, 1, 0};
    CHECK(vec_hamming(x, y, 5) == 2);
    CHECK(near(vec_jaccard(x, y, 5), 2.0 / 4.0));
    // Packed 64-bit words, including the sign bit.
    const std::int64_t p[] = {-1, 0x0F}, q[] = {0, 0xFF};
    CHECK(vec_hamming(p, q, 2) == 64 + 4);
    CHECK(near(vec_jaccard(p, q, 2), 4.0 / (64 + 8)));
    const std::int64_t zero[] = {0, 0};
    CHECK(vec_hamming(zero, zero, 2) == 0 && vec_jaccard(zero, zero, 2) == 1.0);
}

static void test_sum()
{
    VectorSum s;
    CHECK(s.count() == 0 && s.average().empty());
    const double a[] = {1, 2, 3}, b[] = {3, 4, 5}, c[] = {1, 2};
    s.add(a, 3);
    s.add(b, 3);
    CHECK(s.count() == 2 && s.sum()[0] == 4 && s.sum()[1] == 6 && s.sum()[2] == 8);
    CHECK(s.average()[0] == 2 && s.average()[1] == 3 && s.average()[2] == 4);
    CHECK(throws([&] { s.add(c, 2); }, "different lengths: 3 and 2 elements"));
    VectorSum e;
    e.add(a, 0);                                         // empty vectors are vectors too
    CHECK(e.count() == 1 && e.sum().empty());
}

int main()
{
    test_elementwise();
    test_normalize();
    test_distances();
    test_bits();
    test_sum();
    return finish("test_vecmath");
}
