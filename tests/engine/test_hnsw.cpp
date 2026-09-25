// HNSW: the graph is valid, a search with ef = count finds exactly what the flat search finds
// (every metric, masks, journal rows, radius, ties), search results do not depend on the number of
// threads, a one-thread build depends on the data only, damaged graphs are refused, cancel works.
// With --dir=DIR (SIFT1M: DIR/sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs) it also
// builds the 1M index and asserts recall@10 >= 0.95 at ef_search = 100; without, that part is
// skipped with a message.
#include "check.h"

#include "../../src/engine/delta.h"
#include "../../src/engine/flat.h"
#include "../../src/engine/hnsw.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/reach.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>

using namespace vvector;

// count vectors of dims elements in [-1, 1), ids 7, 12, 17, ...; clustered = every 10 rows share
// a centre (gives ties and near duplicates); same = every row equal; dups = clusters of 50 rows, half
// of them equal to the centre, half within 1e-4 of it (the parallel insert race, review item 1).
enum class Shape { Random, Clustered, Same, Dups };

static void build_graph(TestSet &t, std::uint64_t count, std::uint32_t dims, Metric metric, std::uint32_t m,
                        std::uint32_t efc, int threads, Shape shape = Shape::Random, std::uint64_t seed = 3)
{
    SnapshotBuilder b(metric);
    std::vector<float> v(dims), c(dims);
    for (std::uint64_t i = 0; i < count; ++i) {
        if (shape == Shape::Random) test_vector(i, dims, seed, true, v.data());
        else if (shape == Shape::Same) test_vector(0, dims, seed, true, v.data());
        else if (shape == Shape::Dups) {
            test_vector(i / 50, dims, seed, true, c.data());
            test_vector(i, dims, seed + 1, true, v.data());
            const float spread = i % 2 == 0 ? 0.0f : 1e-4f;
            for (std::uint32_t d = 0; d < dims; ++d) v[d] = c[d] + spread * v[d];
        } else {
            test_vector(i / 10, dims, seed, true, c.data());
            test_vector(i, dims, seed + 1, true, v.data());
            for (std::uint32_t d = 0; d < dims; ++d) v[d] = c[d] + 0.01f * v[d];
        }
        b.add(static_cast<std::int64_t>(7 + 5 * i), v.data(), dims);
    }
    HnswParams p;
    p.m = m;
    p.ef_construction = efc;
    p.threads = threads;
    const GraphSection g = hnsw_graph_section(p);
    b.finish(0, t.buffer, &g);
    t.set = snapshot_open(t.buffer.data(), t.buffer.size(), true);
}

static std::vector<float> queries(std::uint64_t n, const VectorSet &s, std::uint64_t seed)
{
    std::vector<float> q(n * s.row_stride, 0.0f);
    for (std::uint64_t i = 0; i < n; ++i) {
        test_vector(i, s.dims, seed, true, q.data() + i * s.row_stride);
        if (s.metric == Metric::Cosine) normalize(q.data() + i * s.row_stride, s.dims);
    }
    return q;
}

static bool same(const std::vector<Neighbor> &a, const std::vector<std::uint32_t> &ca, const std::vector<Neighbor> &b,
                 const std::vector<std::uint32_t> &cb, std::uint32_t k)
{
    if (ca != cb) return false;
    for (std::size_t q = 0; q < ca.size(); ++q)
        for (std::uint32_t i = 0; i < ca[q]; ++i) {
            const Neighbor &x = a[q * k + i], &y = b[q * k + i];
            if (x.id != y.id || std::memcmp(&x.key, &y.key, 4) != 0) return false;
        }
    return true;
}

// Positions reachable on level 0 from the entry point.
static std::uint64_t reachable(const HnswGraph &g)
{
    std::vector<char> seen(g.count, 0);
    std::vector<std::uint32_t> todo(1, g.entry_point);
    seen[g.entry_point] = 1;
    std::uint64_t n = 1;
    while (!todo.empty()) {
        const std::uint32_t p = todo.back();
        todo.pop_back();
        const std::uint32_t *l = g.links(p, 0);
        for (std::uint32_t i = 1; i <= l[0]; ++i)
            if (!seen[l[i]]) { seen[l[i]] = 1; ++n; todo.push_back(l[i]); }
    }
    return n;
}

// No link list on any level holds its own node or one position twice. hnsw_open(verify) checks
// the same, but this reports what it found.
static std::uint64_t bad_links(const HnswGraph &g)
{
    std::uint64_t bad = 0;
    for (std::uint32_t p = 0; p < g.count; ++p)
        for (std::uint32_t lv = 0; lv <= g.levels[p]; ++lv) {
            const std::uint32_t *l = g.links(p, lv);
            for (std::uint32_t i = 1; i <= l[0]; ++i) {
                bad += l[i] == p;
                for (std::uint32_t j = 1; j < i; ++j) bad += l[j] == l[i];
            }
        }
    return bad;
}

// Parallel builds of many equal and near-equal vectors (review item 1 of session 11: a node could
// find itself through a neighbour that another thread had linked to it, and link to itself). Every
// build must give a clean graph: no self link, no duplicate link, every node reachable.
static void test_parallel_duplicates()
{
    const auto t0 = std::chrono::steady_clock::now();
    for (int round = 0; round < 5; ++round) {
        TestSet t;
        build_graph(t, 200000, 16, Metric::L2, 8, 32, 8, Shape::Dups, 40 + round);
        const HnswGraph g = hnsw_open(t.set, false);
        const std::uint64_t bad = bad_links(g), reached = reachable(g);
        if (bad != 0 || reached != t.set.count)
            std::printf("  duplicates round %d: %llu bad links, %llu of %llu reachable\n", round,
                        (unsigned long long)bad, (unsigned long long)reached, (unsigned long long)t.set.count);
        CHECK(bad == 0);
        CHECK(reached == t.set.count);
        CHECK(!throws([&] { hnsw_open(t.set, true); }));
    }
    std::printf("  5 parallel builds of 200000 x 16 with duplicates: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// ef = count on a connected graph visits every node: the result is the exact one.
static void exact_at_full_ef(const char *what, Metric metric, Shape shape)
{
    TestSet t;
    build_graph(t, 1500, 20, metric, 8, 64, 4, shape);
    const HnswGraph g = hnsw_open(t.set, true);
    if (reachable(g) != t.set.count) {
        std::printf("  %s: %llu of %llu reachable\n", what, (unsigned long long)reachable(g), (unsigned long long)t.set.count);
        CHECK(false);
    }
    const std::vector<float> q = queries(20, t.set, 11);
    FlatSearch s;
    s.metric = metric;
    s.stride = t.set.row_stride;
    s.queries = q.data();
    s.n_queries = 20;
    s.threads = 3;
    for (std::uint32_t k : {1u, 10u, 100u}) {
        s.k = k;
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        const RowBlock all{t.set.vectors, t.set.ids, t.set.count, nullptr};
        flat_search(s, &all, 1, ref, rc);
        hnsw_search(s, t.set, g, static_cast<std::uint32_t>(t.set.count), nullptr, nullptr, got, gc);
        if (!same(ref, rc, got, gc, k)) { std::printf("  %s k=%u differs from flat\n", what, k); CHECK(false); }
    }
}

static void test_layout_and_validity()
{
    TestSet t;
    build_graph(t, 3000, 33, Metric::L2, 6, 40, 4);
    CHECK(t.set.has_graph());
    const HnswGraph g = hnsw_open(t.set, true);
    CHECK(g.m == 6 && g.m0 == 12 && g.count == 3000);
    CHECK(g.levels[g.entry_point] == g.max_level);
    std::uint64_t blocks = 0;
    for (std::uint64_t i = 0; i < t.set.count; ++i) {
        CHECK(g.levels[i] == hnsw_level(t.set.ids[i], 6));
        blocks += g.levels[i];
    }
    CHECK(blocks == g.upper_blocks);
    CHECK(t.set.graph_bytes == hnsw_section_bytes(t.set.ids, t.set.count, 6, t.set.capacity));
    // Level distribution: P(level >= 1) = 1 / m.
    std::uint64_t upper = 0;
    for (std::uint64_t i = 0; i < t.set.count; ++i) upper += g.levels[i] >= 1;
    CHECK(upper > 3000 / 6 / 2 && upper < 3000 / 6 * 2);
    // Every node is linked on level 0.
    std::uint64_t empty = 0;
    for (std::uint64_t i = 0; i < t.set.count; ++i) empty += g.links(static_cast<std::uint32_t>(i), 0)[0] == 0;
    CHECK(empty == 0);
    CHECK(reachable(g) == t.set.count);

    // Damaged graphs: the full check (vload) refuses bad links; the header check (every query)
    // refuses a bad header.
    SnapshotBuffer copy;
    copy.allocate(t.buffer.size());
    auto restore = [&] { std::memcpy(copy.data(), t.buffer.data(), t.buffer.size()); };
    auto open_copy = [&](bool verify) { hnsw_open(snapshot_open(copy.data(), copy.size(), false), verify); };
    std::uint32_t *link = reinterpret_cast<std::uint32_t *>(copy.data() + (reinterpret_cast<const std::uint8_t *>(g.level0) - t.buffer.data()));
    HnswHeader *h = reinterpret_cast<HnswHeader *>(copy.data() + (t.set.graph - t.buffer.data()));
    restore();
    CHECK(!throws([&] { open_copy(true); }));
    link[1] = 3000;                                     // a position outside the vectors
    CHECK(throws([&] { open_copy(true); }, "a link points to a wrong position"));
    CHECK(!throws([&] { open_copy(false); }));
    restore();
    link[1] = 0;                                        // node 0 linked to itself
    CHECK(throws([&] { open_copy(true); }, "wrong position"));
    restore();
    link[0] = 13;                                       // longer than m0
    CHECK(throws([&] { open_copy(true); }, "too long"));
    restore();
    CHECK(link[0] >= 2);
    link[2] = link[1];                                  // one position twice in a list
    CHECK(throws([&] { open_copy(true); }, "holds a position twice"));
    restore();
    h->entry_point = 3000;
    CHECK(throws([&] { open_copy(false); }, "entry point"));
    restore();
    h->m = 7;
    h->m0 = 14;
    CHECK(throws([&] { open_copy(false); }, "size does not match"));

    // A flat snapshot has no graph.
    TestSet f;
    build(f, 10, 4);
    CHECK(throws([&] { hnsw_open(f.set, false); }, "no graph section"));
    CHECK(throws([] { hnsw_section_bytes(nullptr, 0, 1, 0); }, "m must be 2 to 256"));
    CHECK(throws([] { hnsw_section_bytes(nullptr, 0, 257, 0); }, "m must be 2 to 256"));
}

// The unreachable count (M7 E): a fresh build stores 0; a graph cut by hand (every link to one
// node removed) counts that node on any number of threads, and equals what the test walk finds;
// tombstoned nodes are never counted; reachability off stores "not counted"; the count is valid
// after an incremental build (delta.h) too.
static void test_unreachable()
{
    TestSet t;
    build_graph(t, 3000, 33, Metric::L2, 6, 40, 4);
    HnswGraph g = hnsw_open(t.set, true);
    CHECK(g.counted && g.unreachable == 0);

    // Cut node 1234 out of every level-0 link list: nothing reaches it any more.
    const std::uint32_t victim = 1234;
    CHECK(g.entry_point != victim);
    std::uint32_t *level0 = reinterpret_cast<std::uint32_t *>(t.buffer.data() + (reinterpret_cast<const std::uint8_t *>(g.level0) - t.buffer.data()));
    for (std::uint32_t p = 0; p < g.count; ++p) {
        std::uint32_t *l = level0 + std::uint64_t(p) * (g.m0 + 1);
        std::uint32_t n = 0;
        for (std::uint32_t i = 1; i <= l[0]; ++i)
            if (l[i] != victim) l[1 + n++] = l[i];
        for (std::uint32_t i = n + 1; i <= l[0]; ++i) l[i] = 0;
        l[0] = n;
    }
    const VectorSet cut = snapshot_open(t.buffer.data(), t.buffer.size(), false);
    const HnswGraph gc = hnsw_open(cut, true);
    CHECK(reachable(gc) == cut.count - 1);
    for (int threads : {1, 3, 8}) CHECK(hnsw_unreachable(cut, gc, threads) == 1);
    // A tombstoned victim is not counted.
    {
        SnapshotBuilder b(Metric::L2);
        std::vector<float> v(33);
        for (std::uint64_t i = 0; i < 3000; ++i) {
            test_vector(i, 33, 3, true, v.data());
            b.add(static_cast<std::int64_t>(7 + 5 * i), v.data(), 33);
        }
        // The same graph, then the victim deleted through an incremental build with reachability on.
        HnswParams p;
        p.m = 6;
        p.ef_construction = 40;
        p.threads = 4;
        p.reachability = Reachability::Off;
        const GraphSection gs = hnsw_graph_section(p);
        SnapshotBuffer buf;
        b.finish(0, buf, &gs);
        const VectorSet off = snapshot_open(buf.data(), buf.size(), true);
        const HnswGraph go = hnsw_open(off, true);
        CHECK(!go.counted && go.unreachable == 0);
        p.reachability = Reachability::On;
        IncrementalBuilder inc(off);
        inc.remove(static_cast<std::int64_t>(7 + 5 * victim));
        SnapshotBuffer out;
        CHECK(inc.finish(1, 1, out, &p));
        const VectorSet next = snapshot_open(out.data(), out.size(), true);
        CHECK(next.dead(victim));
        const HnswGraph gn = hnsw_open(next, true);
        CHECK(gn.counted && gn.unreachable == 0);
        CHECK(hnsw_unreachable(next, gn, 2) == 0);
    }
}

static void test_small()
{
    for (std::uint64_t n : {1u, 2u, 3u, 17u}) {
        TestSet t;
        build_graph(t, n, 5, Metric::L2, 2, 1, 8);
        const HnswGraph g = hnsw_open(t.set, true);
        CHECK(reachable(g) == n);
        const std::vector<float> q = queries(3, t.set, 5);
        FlatSearch s;
        s.metric = Metric::L2;
        s.stride = t.set.row_stride;
        s.queries = q.data();
        s.n_queries = 3;
        s.k = 5;
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        const RowBlock all{t.set.vectors, t.set.ids, t.set.count, nullptr};
        flat_search(s, &all, 1, ref, rc);
        hnsw_search(s, t.set, g, 32, nullptr, nullptr, got, gc);
        CHECK(same(ref, rc, got, gc, 5));
    }
}

static void test_masks_journal_radius()
{
    TestSet t;
    build_graph(t, 2000, 24, Metric::Dot, 8, 64, 4);
    const HnswGraph g = hnsw_open(t.set, true);
    CHECK(reachable(g) == t.set.count);
    const std::vector<float> q = queries(16, t.set, 21);
    // Mask every third position; add journal rows with new ids and with ids of masked positions.
    std::vector<std::uint64_t> skip((t.set.count + 63) / 64, 0);
    for (std::uint64_t i = 0; i < t.set.count; i += 3) skip[i >> 6] |= 1ull << (i & 63);
    skip[g.entry_point >> 6] |= 1ull << (g.entry_point & 63);          // the entry point is masked too
    const std::uint32_t stride = t.set.row_stride;
    std::vector<float> jrows(300 * stride, 0.0f);
    std::vector<std::int64_t> jids(300);
    for (std::uint64_t i = 0; i < 300; ++i) {
        test_vector(i, t.set.dims, 77, true, jrows.data() + i * stride);
        jids[i] = i < 150 ? t.set.ids[i * 3] : 100000 + static_cast<std::int64_t>(i);
    }
    const RowBlock blocks[2] = {RowBlock{t.set.vectors, t.set.ids, t.set.count, skip.data()},
                                RowBlock{jrows.data(), jids.data(), jids.size(), nullptr}};
    FlatSearch s;
    s.metric = Metric::Dot;
    s.stride = stride;
    s.queries = q.data();
    s.n_queries = 16;
    s.k = 25;
    s.threads = 4;
    for (int radius = 0; radius < 2; ++radius) {
        s.has_radius = radius == 1;
        s.radius = 0.5;
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        flat_search(s, blocks, 2, ref, rc);
        hnsw_search(s, t.set, g, static_cast<std::uint32_t>(t.set.count), skip.data(), &blocks[1], got, gc);
        CHECK(same(ref, rc, got, gc, s.k));
        // Approximate search: a masked position never comes back.
        hnsw_search(s, t.set, g, 30, skip.data(), &blocks[1], got, gc);
        for (std::uint64_t qi = 0; qi < 16; ++qi)
            for (std::uint32_t i = 0; i < gc[qi]; ++i) {
                const std::int64_t id = got[qi * s.k + i].id;
                const std::int64_t pos = t.set.find(id);
                const bool from_journal = std::find(jids.begin(), jids.end(), id) != jids.end();
                CHECK(from_journal || (pos >= 0 && !(skip[pos >> 6] >> (pos & 63) & 1)));
                if (s.has_radius) CHECK(within_radius(s, got[qi * s.k + i].key));
            }
    }
    // Everything masked: only journal rows come back.
    std::vector<std::uint64_t> all_masked((t.set.count + 63) / 64, ~0ull);
    std::vector<Neighbor> got;
    std::vector<std::uint32_t> gc;
    s.has_radius = false;
    hnsw_search(s, t.set, g, 50, all_masked.data(), nullptr, got, gc);
    CHECK(gc[0] == 0);
    hnsw_search(s, t.set, g, 50, all_masked.data(), &blocks[1], got, gc);
    CHECK(gc[0] == 25);
}

static void test_threads_and_determinism()
{
    TestSet a, b;
    build_graph(a, 4000, 40, Metric::L2, 12, 80, 1);
    build_graph(b, 4000, 40, Metric::L2, 12, 80, 1);
    CHECK(a.buffer.size() == b.buffer.size() && std::memcmp(a.buffer.data(), b.buffer.data(), a.buffer.size()) == 0);

    TestSet t;
    build_graph(t, 4000, 40, Metric::L2, 12, 80, 8);        // a parallel build is valid
    const HnswGraph g = hnsw_open(t.set, true);
    const std::vector<float> q = queries(200, t.set, 31);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = t.set.row_stride;
    s.queries = q.data();
    s.n_queries = 200;
    s.k = 10;
    std::vector<Neighbor> first, got, ref;
    std::vector<std::uint32_t> fc, gc, rc;
    s.threads = 1;
    hnsw_search(s, t.set, g, 40, nullptr, nullptr, first, fc);
    for (int threads : {2, 3, 8, 64}) {
        s.threads = threads;
        hnsw_search(s, t.set, g, 40, nullptr, nullptr, got, gc);
        CHECK(same(first, fc, got, gc, 10));
    }
    // One query at a time gives the same as the batch.
    s.threads = 4;
    for (std::uint64_t i = 0; i < 200; i += 37) {
        FlatSearch one = s;
        one.queries = q.data() + i * s.stride;
        one.n_queries = 1;
        hnsw_search(one, t.set, g, 40, nullptr, nullptr, got, gc);
        CHECK(gc[0] == fc[i]);
        for (std::uint32_t j = 0; j < gc[0]; ++j) CHECK(got[j].id == first[i * 10 + j].id);
    }
    // Recall against the exact search: random data has no structure, so this is only a sanity check.
    const RowBlock all{t.set.vectors, t.set.ids, t.set.count, nullptr};
    flat_search(s, &all, 1, ref, rc);
    std::uint64_t hit = 0;
    for (std::uint64_t i = 0; i < 200; ++i)
        for (std::uint32_t j = 0; j < 10; ++j)
            for (std::uint32_t r = 0; r < 10; ++r) hit += first[i * 10 + j].id == ref[i * 10 + r].id;
    std::printf("  random 4000 x 40, m 12, ef 40: recall@10 %.3f\n", hit / 2000.0);
    CHECK(hit > 2000 * 0.8);

    // Cancel: poll says stop.
    s.threads = 2;
    bool cancelled = false;
    try {
        hnsw_search(s, t.set, g, 40, nullptr, nullptr, got, gc, [] { return true; });
    } catch (const Cancelled &) {
        cancelled = true;
    }
    CHECK(cancelled);
    cancelled = false;
    try {
        TestSet c;
        SnapshotBuilder bb(Metric::L2);
        std::vector<float> v(8);
        for (int i = 0; i < 5000; ++i) { test_vector(i, 8, 1, true, v.data()); bb.add(i, v.data(), 8); }
        HnswParams p;
        p.threads = 2;
        const GraphSection gs = hnsw_graph_section(p, [] { return true; });
        bb.finish(0, c.buffer, &gs);
    } catch (const Cancelled &) {
        cancelled = true;
    }
    CHECK(cancelled);
}

static void test_sift(const std::string &dir)
{
    if (dir.empty()) {
        std::printf("  SIFT1M recall: skipped (make test DATA_DIR=<dir with sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs>)\n");
        return;
    }
    std::vector<float> base, query;
    std::vector<std::int32_t> truth;
    std::uint32_t dims = 0, qdims = 0, tdims = 0;
    std::uint64_t n = 0, nq = 0, nt = 0;
    if (!read_vecs(dir + "/sift_base.fvecs", base, dims, n) || !read_vecs(dir + "/sift_query.fvecs", query, qdims, nq) ||
        !read_vecs(dir + "/sift_groundtruth.ivecs", truth, tdims, nt)) {
        std::printf("  SIFT1M files not found in %s\n", dir.c_str());
        CHECK(false);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    SnapshotBuilder b(Metric::L2);
    for (std::uint64_t i = 0; i < n; ++i) b.add(static_cast<std::int64_t>(i), base.data() + i * dims, dims);
    std::vector<float>().swap(base);
    HnswParams p;
    p.threads = resolve_threads(0);
    const GraphSection g = hnsw_graph_section(p);
    TestSet t;
    b.finish(0, t.buffer, &g);
    t.set = snapshot_open(t.buffer.data(), t.buffer.size(), false);
    const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const HnswGraph graph = hnsw_open(t.set, true);
    const std::uint32_t stride = t.set.row_stride;
    std::vector<float> qs(nq * stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) std::memcpy(qs.data() + i * stride, query.data() + i * dims, dims * 4);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = 10;
    s.threads = p.threads;
    std::printf("  SIFT1M: %llu x %u, m 16, ef_construction 200, %d threads: build %.1f s, unreachable %llu (counted by the build)\n",
                static_cast<unsigned long long>(n), dims, p.threads, build_s, static_cast<unsigned long long>(graph.unreachable));
    CHECK(graph.counted);
    const auto t1 = std::chrono::steady_clock::now();
    const std::uint64_t again = hnsw_unreachable(t.set, graph, p.threads);
    std::printf("  SIFT1M: the walk again: %llu, %.2f s on %d threads\n", static_cast<unsigned long long>(again),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(), p.threads);
    CHECK(again == graph.unreachable);
    for (std::uint32_t ef : {16u, 32u, 64u, 100u, 200u}) {
        std::vector<Neighbor> got;
        std::vector<std::uint32_t> gc;
        hnsw_search(s, t.set, graph, ef, nullptr, nullptr, got, gc);
        std::uint64_t hit = 0;
        for (std::uint64_t q = 0; q < nq; ++q)
            for (std::uint32_t i = 0; i < gc[q]; ++i)
                for (std::uint32_t r = 0; r < 10; ++r) hit += got[q * 10 + i].id == truth[q * tdims + r];
        const double recall = hit / (10.0 * nq);
        std::printf("  SIFT1M ef_search %3u: recall@10 %.4f\n", ef, recall);
        if (ef == 100) CHECK(recall >= 0.95);
    }
}

int main(int argc, char **argv)
{
    std::string dir;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--dir=", 6) == 0) dir = argv[i] + 6;
    test_layout_and_validity();
    test_unreachable();
    test_small();
    exact_at_full_ef("l2", Metric::L2, Shape::Random);
    exact_at_full_ef("cosine", Metric::Cosine, Shape::Random);
    exact_at_full_ef("dot", Metric::Dot, Shape::Random);
    exact_at_full_ef("l1", Metric::L1, Shape::Random);
    exact_at_full_ef("l2 clustered", Metric::L2, Shape::Clustered);
    exact_at_full_ef("l2 all equal", Metric::L2, Shape::Same);
    test_masks_journal_radius();
    test_threads_and_determinism();
    test_parallel_duplicates();
    test_sift(dir);
    return finish("test_hnsw");
}
