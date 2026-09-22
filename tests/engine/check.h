// Small helpers shared by the engine unit tests.
#ifndef VVECTOR_TESTS_CHECK_H
#define VVECTOR_TESTS_CHECK_H

#include "../../src/engine/snapshot.h"

#include <cstdint>
#include <cstdio>
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

// Deterministic pseudo random numbers, same on every platform.
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::int64_t below(std::int64_t n) { return static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(n)); }
    float unit() { return static_cast<float>(next() >> 40) / static_cast<float>(1ull << 24); }     // [0, 1)
};

// A built snapshot together with the bytes it points into.
struct TestSet {
    vvector::SnapshotBuffer buffer;
    vvector::VectorSet set;
};

// count random vectors with ids 10, 20, 30, ... added in the given order.
inline void build(TestSet &t, std::uint64_t count, std::uint32_t dims, bool reversed = false,
                  vvector::Metric metric = vvector::Metric::L2, std::int64_t max_ver = 0, std::uint64_t seed = 1)
{
    vvector::SnapshotBuilder b(metric);
    std::vector<float> v(dims);
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t n = reversed ? count - 1 - i : i;
        Rng rng(seed * 1000003 + n + 1);     // the values of a vector depend on its id only
        for (std::uint32_t d = 0; d < dims; ++d) v[d] = static_cast<float>(n) + rng.unit();
        b.add(static_cast<std::int64_t>((n + 1) * 10), v.data(), dims);
    }
    b.finish(max_ver, t.buffer);
    t.set = vvector::snapshot_open(t.buffer.data(), t.buffer.size(), true);
}

#endif
