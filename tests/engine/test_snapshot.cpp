// Snapshot format version 2: layout, padding, builder, validation, checksum, id lookup, HNSW stub.
#include "check.h"

#include "../../src/engine/hnsw.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

using namespace vvector;

// Rewrites the checksum after a deliberate change of the bytes.
static void reseal(SnapshotBuffer &b)
{
    const std::uint64_t sum = snapshot_checksum(b.data(), b.size());
    std::memcpy(b.data() + offsetof(SnapshotHeader, checksum), &sum, 8);
}

int main()
{
    // Metric names.
    CHECK(parse_metric("l2") == Metric::L2 && parse_metric("cosine") == Metric::Cosine &&
          parse_metric("dot") == Metric::Dot && parse_metric("l1") == Metric::L1);
    CHECK(throws([] { parse_metric("L2"); }, "metric must be l2, cosine, dot or l1"));
    CHECK(std::string(metric_name(Metric::Cosine)) == "cosine" && std::string(metric_name(Metric::L1)) == "l1");

    // Row stride: dims rounded up to 16.
    CHECK(row_stride_for(1) == 16 && row_stride_for(15) == 16 && row_stride_for(16) == 16 &&
          row_stride_for(17) == 32 && row_stride_for(768) == 768);

    // Layout: header 256 bytes, vectors first, every section on a 64-byte boundary.
    {
        SnapshotHeader h;
        std::memset(&h, 0, sizeof(h));
        h.count = 3;
        h.dims = 3;
        h.row_stride = 16;
        snapshot_layout(h);
        CHECK(h.off_vectors == 256 && h.off_ids == 256 + 192 && h.total_bytes == 512);
        CHECK(h.off_id_index == 0 && h.off_tombstones == 0 && h.off_sq8 == 0 && h.off_graph == 0);
        h.flags = FLAG_ID_INDEX | FLAG_TOMBSTONES | FLAG_SQ8 | FLAG_HNSW;
        h.sq8_bytes = 100;
        h.graph_bytes = 8;
        snapshot_layout(h);
        CHECK(h.off_id_index == 512 && h.off_tombstones == 576 && h.off_sq8 == 640 && h.off_graph == 768 &&
              h.total_bytes == 832);
    }

    // Built in id order, in reverse and shuffled: the same bytes.
    TestSet a, b, c;
    build(a, 1000, 7, Order::Ascending, Metric::L2, 4711);
    build(b, 1000, 7, Order::Reversed, Metric::L2, 4711);
    build(c, 1000, 7, Order::Shuffled, Metric::L2, 4711);
    CHECK(a.buffer.size() == b.buffer.size() && std::memcmp(a.buffer.data(), b.buffer.data(), a.buffer.size()) == 0);
    CHECK(a.buffer.size() == c.buffer.size() && std::memcmp(a.buffer.data(), c.buffer.data(), a.buffer.size()) == 0);
    CHECK(a.set.count == 1000 && a.set.dims == 7 && a.set.row_stride == 16 && a.set.metric == Metric::L2);
    CHECK(a.set.max_ver == 4711 && (a.set.flags & ~FLAG_CAPACITY) == 0 && !a.set.has_graph() && !a.set.normalised());
    CHECK(a.set.id_index == nullptr && a.set.tombstone_bits == nullptr && a.set.base_snapshot == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(a.set.vectors) % 64 == 0);
    bool sorted = true, padded = true;
    for (std::uint64_t i = 1; i < a.set.count; ++i) sorted = sorted && a.set.ids[i - 1] < a.set.ids[i];
    for (std::uint64_t i = 0; i < a.set.count; ++i)
        for (std::uint32_t d = 7; d < 16; ++d) padded = padded && a.set.vector(i)[d] == 0.0f;
    CHECK(sorted && padded);
    // Row i belongs to ids[i]: the vector of id (n + 1) * 10 has elements in [n, n + 1).
    CHECK(a.set.ids[5] == 60 && a.set.vector(5)[0] >= 5.0f && a.set.vector(5)[6] < 6.0f);

    // A larger shuffled build: the buffer grows many times and the rows are moved in place.
    {
        TestSet x, y;
        build(x, 50000, 33, Order::Ascending, Metric::Dot, 1, 7);
        build(y, 50000, 33, Order::Shuffled, Metric::Dot, 1, 7);
        CHECK(x.buffer.size() == y.buffer.size() && std::memcmp(x.buffer.data(), y.buffer.data(), x.buffer.size()) == 0);
        CHECK(x.set.row_stride == 48 && x.buffer.size() % 64 == 0);
    }

    // Id lookup, also through an id_index.
    CHECK(a.set.find(10) == 0 && a.set.find(10000) == 999 && a.set.find(500) == 49 && a.set.find(505) == -1);
    CHECK(a.set.find(5) == -1 && a.set.find(15) == -1 && a.set.find(10010) == -1);
    {
        const std::int64_t ids[4] = {30, 10, 40, 20};
        const std::uint32_t index[4] = {1, 3, 0, 2};
        VectorSet s;
        s.count = 4;
        s.ids = ids;
        s.id_index = index;
        CHECK(s.find(10) == 1 && s.find(20) == 3 && s.find(30) == 0 && s.find(40) == 2);
        CHECK(s.find(25) == -1 && s.find(5) == -1 && s.find(50) == -1);
    }

    // Cosine: rows are stored with unit length; a zero vector stays zero.
    {
        SnapshotBuilder x(Metric::Cosine);
        const float v1[3] = {3, 0, 4}, v2[3] = {0, 0, 0};
        x.add(2, v1, 3);
        x.add(1, v2, 3);
        TestSet t;
        x.finish(0, t.buffer);
        t.set = snapshot_open(t.buffer.data(), t.buffer.size(), true);
        CHECK(t.set.normalised() && (t.set.flags & ~FLAG_CAPACITY) == FLAG_NORMALISED);
        CHECK(t.set.ids[0] == 1 && t.set.vector(0)[0] == 0.0f && t.set.vector(0)[2] == 0.0f);
        CHECK(std::fabs(t.set.vector(1)[0] - 0.6f) < 1e-7f && std::fabs(t.set.vector(1)[2] - 0.8f) < 1e-7f);
    }

    // Builder errors.
    CHECK(throws([] { SnapshotBuilder x(Metric::L2); SnapshotBuffer o; x.finish(0, o); }, "no vectors"));
    CHECK(throws([] {
        SnapshotBuilder x(Metric::L2); const float v[2] = {1, 2};
        x.add(1, v, 2); x.add(2, v, 1); }, "has 1 elements"));
    CHECK(throws([] {
        SnapshotBuilder x(Metric::L2); const float v[2] = {1, 2}; SnapshotBuffer o;
        x.add(2, v, 2); x.add(1, v, 2); x.add(2, v, 2); x.finish(0, o); }, "id 2 appears twice"));
    CHECK(throws([] { SnapshotBuilder x(Metric::L2); const float v[1] = {1}; x.add(1, v, 0); }, "no elements"));
    CHECK(throws([] { SnapshotBuilder x(Metric::L2); x.begin_row(1, MAX_DIMS + 1); }, "at most 32768"));

    // Damage and foreign files are found.
    CHECK(throws([&] { snapshot_open(a.buffer.data(), a.buffer.size() - 64, false); }, "header says"));
    {
        TestSet t;
        build(t, 10, 4);
        t.buffer.data()[300] ^= 1;
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), true); }, "checksum mismatch"));
        CHECK(!throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); }));
        t.buffer.data()[0] = 'X';
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); }, "wrong magic"));
    }
    {
        // A version 1 file (milestone M0) is refused with what to do.
        TestSet t;
        build(t, 10, 4);
        const std::uint32_t v1 = 1;
        std::memcpy(t.buffer.data() + offsetof(SnapshotHeader, format_version), &v1, 4);
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); },
                     "format version 1, this library reads version 2: refresh the index"));
    }
    {
        // Ids out of order are found by the full check (vload), with a valid checksum.
        TestSet t;
        build(t, 10, 4);
        std::int64_t *ids = const_cast<std::int64_t *>(t.set.ids);        // where the header put them
        std::swap(ids[3], ids[4]);
        reseal(t.buffer);
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), true); }, "not unique and ascending"));
    }
    {
        // Flags and sizes must agree.
        TestSet t;
        build(t, 10, 4);
        SnapshotHeader h;
        std::memcpy(&h, t.buffer.data(), sizeof(h));
        h.flags = FLAG_NORMALISED;
        std::memcpy(t.buffer.data(), &h, sizeof(h));
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); }, "must be normalised"));
        h.flags = 64;
        std::memcpy(t.buffer.data(), &h, sizeof(h));
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); }, "unknown flags"));
        h.flags = 0;
        h.row_stride = 4;
        std::memcpy(t.buffer.data(), &h, sizeof(h));
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), false); }, "row_stride"));
    }
    // A section size that would wrap the layout arithmetic around, and reserved fields.
    {
        TestSet u;
        build(u, 10, 4);
        SnapshotHeader h, bad;
        std::memcpy(&h, u.buffer.data(), sizeof(h));
        CHECK(!throws([&] { snapshot_open(u.buffer.data(), u.buffer.size(), false); }));
        bad = h;
        bad.flags |= FLAG_HNSW;
        bad.graph_bytes = ~0ull - 1000;
        std::memcpy(u.buffer.data(), &bad, sizeof(bad));
        CHECK(throws([&] { snapshot_open(u.buffer.data(), u.buffer.size(), false); }, "a section is larger than the file"));
        bad = h;
        bad.reserved[3] = 1;
        std::memcpy(u.buffer.data(), &bad, sizeof(bad));
        CHECK(throws([&] { snapshot_open(u.buffer.data(), u.buffer.size(), false); }, "reserved header fields are not zero"));
    }

    // Checksum of parts, in any order, equals the checksum of the file.
    {
        const std::uint8_t *d = a.buffer.data();
        const std::uint64_t n = a.buffer.size(), cut = 8 * 101, at = offsetof(SnapshotHeader, checksum);
        std::uint64_t parts = snapshot_checksum_part(d + cut, n - cut, cut / 8) ^ snapshot_checksum_part(d, at, 0) ^
                              snapshot_checksum_part(d + at + 8, cut - at - 8, at / 8 + 1);
        CHECK(parts == snapshot_checksum(d, n));
    }

    // A graph section is the last section, built in place, and covered by the checksum.
    {
        SnapshotBuilder b(Metric::L2);
        std::vector<float> v(20);
        for (int i = 0; i < 300; ++i) { test_vector(i, 20, 4, true, v.data()); b.add(300 - i, v.data(), 20); }
        HnswParams p;
        p.m = 4;
        const GraphSection g = hnsw_graph_section(p);
        TestSet t;
        b.finish(9, t.buffer, &g);
        t.set = snapshot_open(t.buffer.data(), t.buffer.size(), true);
        SnapshotHeader h;
        std::memcpy(&h, t.buffer.data(), sizeof(h));
        CHECK((h.flags & ~FLAG_CAPACITY) == FLAG_HNSW && h.off_graph > h.off_ids && h.off_graph % 64 == 0);
        CHECK(h.graph_bytes == hnsw_section_bytes(t.set.ids, 300, 4, t.set.capacity));
        CHECK(h.total_bytes == h.off_graph + h.graph_bytes);
        CHECK(t.set.ids[0] == 1 && t.set.ids[299] == 300);
        t.buffer.data()[h.off_graph + h.graph_bytes / 2] ^= 1;
        CHECK(throws([&] { snapshot_open(t.buffer.data(), t.buffer.size(), true); }, "checksum"));
    }

    // A build in a file (SnapshotBuilder::build_in_file): the same bytes as a build in memory,
    // with a graph and codes, in any order of arrival; the file is gone right away.
    {
        char tmpl[] = "/tmp/vvector_build_XXXXXX";
        const std::string dir = mkdtemp(tmpl);
        auto make = [&](bool in_file, SnapshotBuffer &out) {
            SnapshotBuilder b(Metric::Cosine);
            if (in_file) b.build_in_file(dir);
            std::vector<float> v(40);
            for (std::uint64_t i = 0; i < 30000; ++i) {
                const std::uint64_t k = (i * 7919) % 30000;
                test_vector(k, 40, 5, true, v.data());
                b.add(static_cast<std::int64_t>(10 + 3 * k), v.data(), 40);
            }
            HnswParams p;
            p.threads = 1;
            const GraphSection g = hnsw_graph_section(p);
            b.finish(9, out, &g);
        };
        SnapshotBuffer ram, file;
        make(false, ram);
        make(true, file);
#if defined(__linux__)
        CHECK(file.file_backed() && !ram.file_backed());
#endif
        CHECK(ram.size() == file.size() && std::memcmp(ram.data(), file.data(), ram.size()) == 0);
        CHECK(!throws([&] { snapshot_open(file.data(), file.size(), true); }));
        CHECK(std::system(("test -z \"$(ls -A " + dir + ")\"").c_str()) == 0);     // nothing left in the directory
        file.clear();
        std::system(("rm -rf " + dir).c_str());
    }

    return finish("test_snapshot");
}
