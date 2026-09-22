// Snapshot format: layout, builder, validation, checksum, and the HNSW stub.
#include "check.h"

#include "../../src/engine/hnsw.h"

#include <cstring>
#include <stdexcept>
#include <string>

using namespace vvector;

template <class F> static bool throws(F f, const char *containing = "")
{
    try { f(); } catch (const std::runtime_error &e) { return std::strstr(e.what(), containing) != nullptr; }
    return false;
}

int main()
{
    // Metric names.
    CHECK(parse_metric("l2") == Metric::L2 && parse_metric("cosine") == Metric::Cosine && parse_metric("dot") == Metric::Dot);
    CHECK(throws([] { parse_metric("L2"); }, "metric must be"));
    CHECK(std::string(metric_name(Metric::Cosine)) == "cosine");

    // Layout: header, ids, vectors; every section 8-byte aligned.
    SnapshotHeader h;
    std::memset(&h, 0, sizeof(h));
    h.count = 3;
    h.dims = 3;
    snapshot_layout(h);
    CHECK(h.off_ids == 128 && h.off_vectors == 128 + 24 && h.off_graph == 0);
    CHECK(h.total_bytes == 128 + 24 + 40);      // 36 bytes of vectors, padded to 40
    CHECK(h.total_bytes % 8 == 0);

    // Build in id order and in reverse order: the same bytes.
    TestSet a, b;
    build(a, 1000, 7, false, Metric::Cosine, 4711);
    build(b, 1000, 7, true, Metric::Cosine, 4711);
    CHECK(a.buffer.size() == b.buffer.size() && std::memcmp(a.buffer.data(), b.buffer.data(), a.buffer.size()) == 0);
    CHECK(a.set.count == 1000 && a.set.dims == 7 && a.set.metric == Metric::Cosine && a.set.max_ver == 4711);
    CHECK(!a.set.has_graph);
    bool sorted = true;
    for (std::uint64_t i = 1; i < a.set.count; ++i) sorted = sorted && a.set.ids[i - 1] < a.set.ids[i];
    CHECK(sorted);
    // Row i belongs to ids[i]: vector of id (n + 1) * 10 has elements in [n, n + 1).
    CHECK(a.set.ids[5] == 60 && a.set.vector(5)[0] >= 5.0f && a.set.vector(5)[6] < 6.0f);

    // Builder errors.
    CHECK(throws([] { SnapshotBuilder x(Metric::L2); SnapshotBuffer o; x.finish(0, o); }, "no vectors"));
    CHECK(throws([] {
        SnapshotBuilder x(Metric::L2); const float v[2] = {1, 2};
        x.add(1, v, 2); x.add(2, v, 1); }, "has 1 elements"));
    CHECK(throws([] {
        SnapshotBuilder x(Metric::L2); const float v[2] = {1, 2}; SnapshotBuffer o;
        x.add(2, v, 2); x.add(1, v, 2); x.add(2, v, 2); x.finish(0, o); }, "appears twice"));
    CHECK(throws([] { SnapshotBuilder x(Metric::L2); const float v[1] = {1}; x.add(1, v, 0); }, "no elements"));

    // Damage is found.
    CHECK(throws([&] { snapshot_open(a.buffer.data(), a.buffer.size() - 8, false); }, "header says"));
    {
        TestSet c;
        build(c, 10, 4);
        c.buffer.data()[200] ^= 1;
        CHECK(throws([&] { snapshot_open(c.buffer.data(), c.buffer.size(), true); }, "checksum mismatch"));
        CHECK(!throws([&] { snapshot_open(c.buffer.data(), c.buffer.size(), false); }));
        c.buffer.data()[0] = 'X';
        CHECK(throws([&] { snapshot_open(c.buffer.data(), c.buffer.size(), false); }, "wrong magic"));
    }

    // Checksum of parts, in any order, equals the checksum of the file.
    {
        const std::uint8_t *d = a.buffer.data();
        const std::uint64_t n = a.buffer.size(), cut = 8 * 101, at = offsetof(SnapshotHeader, checksum);
        std::uint64_t parts = snapshot_checksum_part(d + cut, n - cut, cut / 8) ^ snapshot_checksum_part(d, at, 0) ^
                              snapshot_checksum_part(d + at + 8, cut - at - 8, at / 8 + 1);
        CHECK(parts == snapshot_checksum(d, n));
    }

    // HNSW is a stub.
    CHECK(throws([&] { hnsw_build(a.set, HnswParams()); }, "not implemented"));
    CHECK(throws([&] { hnsw_search(a.set, a.set.vector(0), 5, 50); }, "not implemented"));

    return finish("test_snapshot");
}
