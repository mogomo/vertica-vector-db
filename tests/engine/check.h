// Small helpers shared by the engine unit tests.
#ifndef VVECTOR_TESTS_CHECK_H
#define VVECTOR_TESTS_CHECK_H

#include "../../src/engine/snapshot.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

inline int finish(const char *name)
{
    if (failures == 0) std::printf("%s: OK\n", name);
    else std::printf("%s: %d FAILURES\n", name, failures);
    return failures == 0 ? 0 : 1;
}

// True if f throws std::runtime_error with a message that contains `containing`.
template <class F> bool throws(F f, const char *containing = "")
{
    try { f(); } catch (const std::runtime_error &e) { return std::strstr(e.what(), containing) != nullptr; }
    return false;
}

// Deterministic pseudo random numbers, same on every platform.
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::int64_t below(std::int64_t n) { return static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(n)); }
    float unit() { return static_cast<float>(next() >> 40) / static_cast<float>(1ull << 24); }     // [0, 1)
    float sym() { return unit() * 2.0f - 1.0f; }                                                   // [-1, 1)
};

// A built snapshot together with the bytes it points into.
struct TestSet {
    vvector::SnapshotBuffer buffer;
    vvector::VectorSet set;
};

enum class Order { Ascending, Reversed, Shuffled };

// The vector of the n-th test row: elements in [n, n + 1) for the snapshot tests, or in [-1, 1)
// when centered. Depends on n and the seed only, not on the order rows are added in.
inline void test_vector(std::uint64_t n, std::uint32_t dims, std::uint64_t seed, bool centered, float *v)
{
    Rng rng(seed * 1000003 + n + 1);
    for (std::uint32_t d = 0; d < dims; ++d) v[d] = centered ? rng.sym() : static_cast<float>(n) + rng.unit();
}

// count test vectors with ids 10, 20, 30, ... added in the given order.
inline void build(TestSet &t, std::uint64_t count, std::uint32_t dims, Order order = Order::Ascending,
                  vvector::Metric metric = vvector::Metric::L2, std::int64_t max_ver = 0, std::uint64_t seed = 1,
                  bool centered = false)
{
    std::vector<std::uint64_t> seq(count);
    for (std::uint64_t i = 0; i < count; ++i) seq[i] = order == Order::Reversed ? count - 1 - i : i;
    if (order == Order::Shuffled) {
        Rng rng(seed + 99);
        for (std::uint64_t i = count; i > 1; --i) std::swap(seq[i - 1], seq[rng.below(static_cast<std::int64_t>(i))]);
    }
    vvector::SnapshotBuilder b(metric);
    std::vector<float> v(dims);
    for (std::uint64_t n : seq) {
        test_vector(n, dims, seed, centered, v.data());
        b.add(static_cast<std::int64_t>((n + 1) * 10), v.data(), dims);
    }
    b.finish(max_ver, t.buffer);
    t.set = vvector::snapshot_open(t.buffer.data(), t.buffer.size(), true);
}

// Reads a .fvecs or .ivecs file (SIFT1M): n vectors of dims elements.
template <class T> bool read_vecs(const std::string &path, std::vector<T> &out, std::uint32_t &dims, std::uint64_t &n)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    n = 0;
    std::int32_t d;
    while (std::fread(&d, 4, 1, f) == 1) {
        dims = static_cast<std::uint32_t>(d);
        const std::size_t at = out.size();
        out.resize(at + d);
        if (std::fread(out.data() + at, 4, d, f) != static_cast<std::size_t>(d)) { std::fclose(f); return false; }
        ++n;
    }
    std::fclose(f);
    return true;
}

#endif
