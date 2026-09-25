// Incremental build (src/engine/delta.h). After every round of random changes (deletes, changed
// vectors, re-adds of deleted ids, new ids above and between the old ones, unchanged re-adds,
// deletes of ids that are not there) the new snapshot passes the full verification, holds exactly
// the live vectors, and a flat search over it gives bit for bit what a full build of the same
// vectors gives. On an HNSW index a graph search with ef = count does too, and every live node stays
// reachable. Also: changes that change nothing, the flags, and the errors.
// With --dir=DIR (SIFT1M files) also: a 900k HNSW base, then 100 rounds of 1000 adds and 500
// deletes; recall@10 at ef_search 100 must be at least 0.95 against the exact answer of the live
// set, and is printed next to the recall of a full build of the same live set.
#include "check.h"

#include "../../src/engine/delta.h"
#include "../../src/engine/flat.h"
#include "../../src/engine/hnsw.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/cache.h"
#include "../../src/engine/sq8.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <unistd.h>

using namespace vvector;

using Live = std::map<std::int64_t, std::vector<float>>;

struct Change {
    std::int64_t id;
    bool del;
    std::vector<float> vec;
};

// A full build of the live vectors, with an HNSW graph when p is given.
static void full_build(TestSet &t, const Live &live, Metric metric, const HnswParams *p)
{
    SnapshotBuilder b(metric);
    for (const auto &x : live) b.add(x.first, x.second.data(), static_cast<std::uint32_t>(x.second.size()));
    if (p) {
        const GraphSection g = hnsw_graph_section(*p);
        b.finish(0, t.buffer, &g);
    } else {
        b.finish(0, t.buffer);
    }
    t.set = snapshot_open(t.buffer.data(), t.buffer.size(), true);
}

// The changes applied to base by the incremental builder; out is opened with the full verification.
static bool incremental(TestSet &out, const TestSet &base, const std::vector<Change> &changes, const HnswParams *p,
                        std::int64_t base_id, DeltaStats *stats = nullptr)
{
    IncrementalBuilder b(base.set);
    for (const Change &c : changes) {
        if (c.del) {
            b.remove(c.id);
        } else {
            float *row = b.begin_add(c.id, static_cast<std::uint32_t>(c.vec.size()));
            std::memcpy(row, c.vec.data(), c.vec.size() * 4);
            b.end_add();
        }
    }
    const bool changed = b.finish(77, base_id, out.buffer, p);
    if (stats) *stats = b.stats();
    if (changed) {
        out.set = snapshot_open(out.buffer.data(), out.buffer.size(), true);
        if (p) hnsw_open(out.set, true);
    }
    return changed;
}

static void take(TestSet &to, TestSet &from)
{
    to.buffer.swap(from.buffer);
    to.set = from.set;
    from.buffer.clear();
}

// Every live position holds the vector the full build holds for its id, find gives it, and there
// is one live position per live id.
static bool same_live(const TestSet &inc, const TestSet &full, const Live &live)
{
    std::uint64_t alive = 0;
    for (std::uint64_t p = 0; p < inc.set.count; ++p) {
        if (inc.set.dead(p)) continue;
        ++alive;
        const std::int64_t id = inc.set.ids[p];
        const std::int64_t f = full.set.find(id);
        if (f < 0 || inc.set.find(id) != static_cast<std::int64_t>(p)) return false;
        if (std::memcmp(inc.set.vector(p), full.set.vector(f), inc.set.row_stride * 4) != 0) return false;
    }
    return alive == live.size() && inc.set.count - inc.set.tombstones == live.size();
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

// Live positions reachable on level 0 from the entry point, and all live positions.
static void reachable_live(const VectorSet &s, const HnswGraph &g, std::uint64_t &reached, std::uint64_t &alive)
{
    std::vector<char> seen(g.count, 0);
    std::vector<std::uint32_t> todo(1, g.entry_point);
    seen[g.entry_point] = 1;
    while (!todo.empty()) {
        const std::uint32_t p = todo.back();
        todo.pop_back();
        const std::uint32_t *l = g.links(p, 0);
        for (std::uint32_t i = 1; i <= l[0]; ++i)
            if (!seen[l[i]]) { seen[l[i]] = 1; todo.push_back(l[i]); }
    }
    reached = alive = 0;
    for (std::uint64_t p = 0; p < g.count; ++p)
        if (!s.dead(p)) { ++alive; reached += seen[p]; }
}

// Flat search over inc (tombstones skipped) and, with a graph, a graph search with ef = count, both
// equal to the flat search over the full build.
static bool same_results(const TestSet &inc, const TestSet &full, bool graph)
{
    const std::uint32_t stride = full.set.row_stride;
    const std::uint64_t nq = 12;
    std::vector<float> q(nq * stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) {
        test_vector(i, full.set.dims, 91, true, q.data() + i * stride);
        if (full.set.metric == Metric::Cosine) normalize(q.data() + i * stride, full.set.dims);
    }
    FlatSearch s;
    s.metric = full.set.metric;
    s.stride = stride;
    s.queries = q.data();
    s.n_queries = nq;
    s.threads = 3;
    bool ok = true;
    for (std::uint32_t k : {1u, 10u, 50u}) {
        s.k = k;
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        const RowBlock all{full.set.vectors, full.set.ids, full.set.count, nullptr};
        flat_search(s, &all, 1, ref, rc);
        const RowBlock mine{inc.set.vectors, inc.set.ids, inc.set.count, inc.set.tombstone_bits};
        flat_search(s, &mine, 1, got, gc);
        ok = ok && same(ref, rc, got, gc, k);
        if (graph) {
            hnsw_search(s, inc.set, hnsw_open(inc.set, false), static_cast<std::uint32_t>(inc.set.count),
                        inc.set.tombstone_bits, nullptr, got, gc);
            ok = ok && same(ref, rc, got, gc, k);
        }
    }
    return ok;
}

static std::vector<float> random_vector(Rng &rng, std::uint32_t dims)
{
    std::vector<float> v(dims);
    for (float &x : v) x = rng.sym();
    return v;
}

// Rounds of random changes on a base of 2000 vectors (ids 10, 20, ...).
static void rounds(const char *what, Metric metric, bool graph)
{
    const std::uint32_t dims = 20;
    HnswParams p;
    p.m = 8;
    p.ef_construction = 64;
    p.threads = 4;
    const HnswParams *hp = graph ? &p : nullptr;
    Rng rng(metric == Metric::Cosine ? 5 : 6);

    Live live;
    for (std::int64_t i = 1; i <= 2000; ++i) live[10 * i] = random_vector(rng, dims);
    std::vector<std::int64_t> gone;
    std::int64_t next_id = 20010;
    TestSet cur;
    full_build(cur, live, metric, hp);

    for (int round = 1; round <= 8; ++round) {
        std::vector<Change> ch;
        std::set<std::int64_t> touched;
        std::vector<std::int64_t> keys;
        for (const auto &x : live) keys.push_back(x.first);
        auto pick_live = [&]() {
            for (;;) {
                const std::int64_t id = keys[rng.below(static_cast<std::int64_t>(keys.size()))];
                if (touched.insert(id).second) return id;
            }
        };
        DeltaStats want;
        // id 20 changes in every round: it ends up at many positions, all but the last dead.
        touched.insert(20);
        ch.push_back(Change{20, false, random_vector(rng, dims)});
        ++want.tombstoned;
        ++want.appended;
        for (int i = 0; i < 60; ++i) { ch.push_back(Change{pick_live(), true, {}}); ++want.tombstoned; }
        for (int i = 0; i < 60; ++i) { ch.push_back(Change{pick_live(), false, random_vector(rng, dims)}); ++want.tombstoned; ++want.appended; }
        for (int i = 0; i < 20; ++i) { const std::int64_t id = pick_live(); ch.push_back(Change{id, false, live[id]}); ++want.unchanged; }
        for (int i = 0; i < 20 && !gone.empty(); ++i) {
            const std::int64_t id = gone[rng.below(static_cast<std::int64_t>(gone.size()))];
            if (!touched.insert(id).second) continue;
            ch.push_back(Change{id, false, random_vector(rng, dims)});
            ++want.appended;
        }
        for (int i = 0; i < 40; ++i) {                              // new ids between the old ones
            const std::int64_t id = 10 * (1 + rng.below(2000)) + 1 + rng.below(9);
            if (live.count(id) || !touched.insert(id).second) continue;
            ch.push_back(Change{id, false, random_vector(rng, dims)});
            ++want.appended;
        }
        for (int i = 0; i < 40; ++i) { ch.push_back(Change{next_id, false, random_vector(rng, dims)}); touched.insert(next_id); next_id += 10; ++want.appended; }
        for (int i = 0; i < 10; ++i) {                              // deletes of ids that are not there
            const std::int64_t id = -1 - 1000 * round - i;
            ch.push_back(Change{id, true, {}});
            ++want.absent;
        }
        // The order of the changes must not matter.
        for (std::size_t i = ch.size(); i > 1; --i) std::swap(ch[i - 1], ch[rng.below(static_cast<std::int64_t>(i))]);

        for (const Change &c : ch) {
            if (c.del) {
                if (live.erase(c.id)) gone.push_back(c.id);
            } else {
                live[c.id] = c.vec;
            }
        }
        gone.erase(std::remove_if(gone.begin(), gone.end(), [&](std::int64_t id) { return live.count(id) != 0; }), gone.end());

        TestSet next, full;
        DeltaStats got;
        CHECK(incremental(next, cur, ch, hp, 100 + round, &got));
        full_build(full, live, metric, hp);
        const bool stats_ok = got.appended == want.appended && got.tombstoned == want.tombstoned &&
                              got.unchanged == want.unchanged && got.absent == want.absent;
        if (!stats_ok)
            std::printf("  %s round %d: appended %llu/%llu tombstoned %llu/%llu unchanged %llu/%llu absent %llu/%llu\n", what, round,
                        (unsigned long long)got.appended, (unsigned long long)want.appended, (unsigned long long)got.tombstoned,
                        (unsigned long long)want.tombstoned, (unsigned long long)got.unchanged, (unsigned long long)want.unchanged,
                        (unsigned long long)got.absent, (unsigned long long)want.absent);
        CHECK(stats_ok);
        CHECK(next.set.base_snapshot == 100 + round && next.set.max_ver == 77);
        CHECK(next.set.id_index != nullptr && next.set.tombstone_bits != nullptr);
        CHECK(next.set.count == cur.set.count + got.appended && next.set.tombstones == cur.set.tombstones + got.tombstoned);
        if (!same_live(next, full, live)) { std::printf("  %s round %d: live vectors differ\n", what, round); CHECK(false); }
        for (const std::int64_t id : gone) CHECK(next.set.find(id) == -1);
        {   // find_sorted gives what find gives: live, gone, never seen and repeated-position ids
            std::set<std::int64_t> ask(gone.begin(), gone.end());
            for (const auto &x : live) ask.insert(x.first);
            for (std::int64_t i = -3; i <= 30000; i += 97) ask.insert(i);
            const std::vector<std::int64_t> want(ask.begin(), ask.end());
            std::vector<std::int64_t> got_pos(want.size());
            next.set.find_sorted(want.data(), want.size(), got_pos.data());
            bool same_pos = true;
            for (std::size_t i = 0; i < want.size(); ++i) same_pos &= got_pos[i] == next.set.find(want[i]);
            CHECK(same_pos);
        }
        if (!same_results(next, full, graph)) { std::printf("  %s round %d: search differs from a full build\n", what, round); CHECK(false); }
        if (graph) {
            std::uint64_t reached, alive;
            reachable_live(next.set, hnsw_open(next.set, true), reached, alive);
            if (reached != alive) { std::printf("  %s round %d: %llu of %llu live nodes reachable\n", what, round, (unsigned long long)reached, (unsigned long long)alive); CHECK(false); }
        }
        take(cur, next);
    }
    // id 20 is at 9 positions now, the last one live.
    std::uint64_t at = 0;
    for (std::uint64_t p = 0; p < cur.set.count; ++p) at += cur.set.ids[p] == 20;
    CHECK(at == 9 && cur.set.find(20) >= 0 && static_cast<std::uint64_t>(cur.set.find(20)) >= 2000);
}

static void test_flags_and_nothing()
{
    TestSet base;
    build(base, 100, 5, Order::Ascending, Metric::L2);        // ids 10 .. 1000
    std::vector<float> v(5, 0.5f), row(base.set.vector(3), base.set.vector(3) + 5);

    // Only new ids above the largest: ids stay ascending, no id_index, no tombstones.
    TestSet a;
    CHECK(incremental(a, base, {Change{2000, false, v}, Change{1500, false, v}}, nullptr, 9));
    CHECK(a.set.count == 102 && (a.set.flags & ~FLAG_CAPACITY) == 0 && a.set.id_index == nullptr && a.set.tombstone_bits == nullptr);
    CHECK(a.set.ids[100] == 1500 && a.set.ids[101] == 2000 && a.set.find(1500) == 100 && a.set.find(40) == 3);
    // A delete only: tombstones, no id_index.
    TestSet b;
    CHECK(incremental(b, base, {Change{40, true, {}}}, nullptr, 9));
    CHECK(b.set.count == 100 && b.set.tombstones == 1 && (b.set.flags & ~FLAG_CAPACITY) == FLAG_TOMBSTONES && b.set.find(40) == -1);
    // A new id between: id_index.
    TestSet c;
    CHECK(incremental(c, base, {Change{45, false, v}}, nullptr, 9));
    CHECK((c.set.flags & ~FLAG_CAPACITY) == FLAG_ID_INDEX && c.set.find(45) == 100 && c.set.find(50) == 4);

    // Nothing changes: the same vector again, a delete of an absent id. out is left alone.
    TestSet d;
    DeltaStats st;
    CHECK(!incremental(d, base, {Change{40, false, row}, Change{45, true, {}}}, nullptr, 9, &st));
    CHECK(d.buffer.size() == 0 && st.unchanged == 1 && st.absent == 1 && st.appended == 0 && st.tombstoned == 0);
    // Every id deleted: a snapshot of dead positions only, still valid and searchable.
    std::vector<Change> all;
    for (std::int64_t id = 10; id <= 1000; id += 10) all.push_back(Change{id, true, {}});
    TestSet e;
    CHECK(incremental(e, base, all, nullptr, 9));
    CHECK(e.set.count == 100 && e.set.tombstones == 100);
    TestSet f;
    CHECK(incremental(f, e, {Change{40, false, v}}, nullptr, 10));
    CHECK(f.set.find(40) == 100 && f.set.count - f.set.tombstones == 1);

    // Errors.
    CHECK(throws([&] { TestSet x; incremental(x, base, {Change{2000, false, v}, Change{2000, false, v}}, nullptr, 9); }, "id 2000 appears twice"));
    CHECK(throws([&] { TestSet x; incremental(x, base, {Change{40, true, {}}, Change{40, false, v}}, nullptr, 9); }, "id 40 appears twice"));
    CHECK(throws([&] { TestSet x; incremental(x, base, {Change{30, true, {}}, Change{30, true, {}}}, nullptr, 9); }, "id 30 appears twice"));
    CHECK(throws([&] { TestSet x; incremental(x, base, {Change{7, false, std::vector<float>(6, 1.0f)}}, nullptr, 9); }, "has 6 elements, the index has 5"));
    HnswParams p;
    CHECK(throws([&] { TestSet x; incremental(x, base, {Change{7, false, v}}, &p, 9); }, "is a flat index, the build is hnsw"));
    TestSet g;
    {
        SnapshotBuilder sb(Metric::L2);
        for (std::int64_t id = 1; id <= 50; ++id) sb.add(id, v.data(), 5);
        p.m = 6;
        const GraphSection gs = hnsw_graph_section(p);
        sb.finish(0, g.buffer, &gs);
        g.set = snapshot_open(g.buffer.data(), g.buffer.size(), true);
    }
    CHECK(throws([&] { TestSet x; incremental(x, g, {Change{70, false, v}}, nullptr, 9); }, "is an HNSW index, the build is flat"));
    p.m = 16;
    CHECK(throws([&] { TestSet x; incremental(x, g, {Change{70, false, v}}, &p, 9); }, "base snapshot was built with m 6"));

    // The verification refuses an id that is live at two positions.
    TestSet h;
    CHECK(incremental(h, base, {Change{40, false, v}}, nullptr, 9));   // 40 at positions 100 (live) and 3 (dead)
    SnapshotHeader hh;
    std::memcpy(&hh, h.buffer.data(), sizeof(hh));
    std::uint64_t *bits = reinterpret_cast<std::uint64_t *>(h.buffer.data() + hh.off_tombstones);
    bits[0] &= ~(1ull << 3);
    hh.tombstones = 0;
    std::memcpy(h.buffer.data(), &hh, sizeof(hh));
    const std::uint64_t sum = snapshot_checksum(h.buffer.data(), h.buffer.size());
    std::memcpy(h.buffer.data() + offsetof(SnapshotHeader, checksum), &sum, 8);
    CHECK(throws([&] { snapshot_open(h.buffer.data(), h.buffer.size(), true); }, "an id is live at two positions"));
}

static void test_sift(const std::string &dir)
{
    if (dir.empty()) {
        std::printf("  SIFT1M incremental rounds: skipped (make test DATA_DIR=<dir with sift_base.fvecs, sift_query.fvecs>)\n");
        return;
    }
    std::vector<float> base, query;
    std::uint32_t dims = 0, qdims = 0;
    std::uint64_t n = 0, nq = 0;
    if (!read_vecs(dir + "/sift_base.fvecs", base, dims, n) || !read_vecs(dir + "/sift_query.fvecs", query, qdims, nq)) {
        std::printf("  SIFT1M files not found in %s\n", dir.c_str());
        CHECK(false);
        return;
    }
    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const std::uint64_t n0 = 900000, rounds = 100, adds = 1000, dels = 500;
    HnswParams p;
    p.threads = resolve_threads(0);
    TestSet cur;
    {
        SnapshotBuilder b(Metric::L2);
        for (std::uint64_t i = 0; i < n0; ++i) b.add(static_cast<std::int64_t>(i), base.data() + i * dims, dims);
        const GraphSection g = hnsw_graph_section(p);
        b.finish(0, cur.buffer, &g);
        cur.set = snapshot_open(cur.buffer.data(), cur.buffer.size(), false);
    }
    std::vector<std::int64_t> live_ids(n0);
    for (std::uint64_t i = 0; i < n0; ++i) live_ids[i] = static_cast<std::int64_t>(i);
    Rng rng(17);
    double total = 0, slowest = 0;
    std::uint64_t added = n0;
    for (std::uint64_t r = 0; r < rounds; ++r) {
        const Clock::time_point t0 = Clock::now();
        IncrementalBuilder ib(cur.set);
        for (std::uint64_t i = 0; i < adds; ++i, ++added) {
            float *row = ib.begin_add(static_cast<std::int64_t>(added), dims);
            std::memcpy(row, base.data() + added * dims, dims * 4);
            ib.end_add();
        }
        for (std::uint64_t i = 0; i < dels; ++i) {
            const std::uint64_t at = static_cast<std::uint64_t>(rng.below(static_cast<std::int64_t>(live_ids.size())));
            ib.remove(live_ids[at]);
            live_ids[at] = live_ids.back();
            live_ids.pop_back();
        }
        for (std::uint64_t i = added - adds; i < added; ++i) live_ids.push_back(static_cast<std::int64_t>(i));
        TestSet next;
        CHECK(ib.finish(0, static_cast<std::int64_t>(r + 1), next.buffer, &p));
        next.set = snapshot_open(next.buffer.data(), next.buffer.size(), false);
        const double s = secs(t0);
        total += s;
        slowest = std::max(slowest, s);
        take(cur, next);
    }
    const HnswGraph graph = hnsw_open(cur.set, true);
    CHECK(snapshot_open(cur.buffer.data(), cur.buffer.size(), true).tombstones == rounds * dels);
    std::uint64_t reached, alive;
    reachable_live(cur.set, graph, reached, alive);
    CHECK(reached == alive && alive == live_ids.size());
    std::printf("  SIFT1M: %llu base, %llu rounds of %llu adds and %llu deletes: %.3f s per round (slowest %.3f s), %llu positions, %llu tombstones\n",
                (unsigned long long)n0, (unsigned long long)rounds, (unsigned long long)adds, (unsigned long long)dels,
                total / rounds, slowest, (unsigned long long)cur.set.count, (unsigned long long)cur.set.tombstones);

    // The exact answer of the live set, the incremental graph, and a full build of the same live set.
    const std::uint32_t stride = cur.set.row_stride, k = 10;
    std::vector<float> qs(nq * stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) std::memcpy(qs.data() + i * stride, query.data() + i * dims, dims * 4);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = k;
    s.threads = p.threads;
    std::vector<Neighbor> ref;
    std::vector<std::uint32_t> rc;
    const RowBlock all{cur.set.vectors, cur.set.ids, cur.set.count, cur.set.tombstone_bits};
    flat_search(s, &all, 1, ref, rc);
    auto recall = [&](const std::vector<Neighbor> &got, const std::vector<std::uint32_t> &gc) {
        std::uint64_t hit = 0;
        for (std::uint64_t q = 0; q < nq; ++q)
            for (std::uint32_t i = 0; i < gc[q]; ++i)
                for (std::uint32_t j = 0; j < rc[q]; ++j) hit += got[q * k + i].id == ref[q * k + j].id;
        return hit / (double(k) * nq);
    };
    TestSet full;
    {
        std::sort(live_ids.begin(), live_ids.end());
        const Clock::time_point t0 = Clock::now();
        SnapshotBuilder b(Metric::L2);
        for (const std::int64_t id : live_ids) b.add(id, base.data() + std::uint64_t(id) * dims, dims);
        const GraphSection g = hnsw_graph_section(p);
        b.finish(0, full.buffer, &g);
        full.set = snapshot_open(full.buffer.data(), full.buffer.size(), false);
        std::printf("  SIFT1M: full build of the same %llu live vectors: %.1f s\n", (unsigned long long)live_ids.size(), secs(t0));
    }
    const HnswGraph full_graph = hnsw_open(full.set, true);
    for (std::uint32_t ef : {32u, 64u, 100u, 200u}) {
        std::vector<Neighbor> got, fgot;
        std::vector<std::uint32_t> gc, fgc;
        hnsw_search(s, cur.set, graph, ef, cur.set.tombstone_bits, nullptr, got, gc);
        hnsw_search(s, full.set, full_graph, ef, nullptr, nullptr, fgot, fgc);
        const double r = recall(got, gc), fr = recall(fgot, fgc);
        std::printf("  SIFT1M ef_search %3u: recall@10 %.4f after the incremental rounds, %.4f full build\n", ef, r, fr);
        if (ef == 100) CHECK(r >= 0.95);
    }
}

// Vector i of clusters of 50: even i equal to the centre of its cluster, odd i within 1e-4 of it.
static std::vector<float> dup_vector(std::int64_t i, std::uint32_t dims, std::uint64_t seed)
{
    std::vector<float> c(dims), v(dims);
    test_vector(static_cast<std::uint64_t>(i / 50), dims, seed, true, c.data());
    test_vector(static_cast<std::uint64_t>(i), dims, seed + 1, true, v.data());
    const float spread = i % 2 == 0 ? 0.0f : 1e-4f;
    for (std::uint32_t d = 0; d < dims; ++d) v[d] = c[d] + spread * v[d];
    return v;
}

// Parallel inserts of equal and near-equal vectors into an extended graph with tombstones (review
// item 1 of session 11, the incremental path): every round must pass the full verification (no self
// link, no position twice in a list) and keep every live node reachable.
static void test_parallel_duplicates()
{
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t dims = 16;
    HnswParams hp;
    hp.m = 8;
    hp.ef_construction = 32;
    hp.threads = 8;
    Live live;
    for (std::int64_t i = 0; i < 100000; ++i) live[i] = dup_vector(i, dims, 5);
    TestSet cur;
    full_build(cur, live, Metric::L2, &hp);
    Rng rng(17);
    std::int64_t next_id = 100000;
    for (int round = 1; round <= 3; ++round) {
        std::vector<Change> ch;
        std::set<std::int64_t> touched;
        for (int i = 0; i < 5000; ++i) {                             // deletes
            const std::int64_t id = rng.below(next_id);
            if (live.count(id) && touched.insert(id).second) ch.push_back(Change{id, true, {}});
        }
        for (int i = 0; i < 30000; ++i) {                            // new ids in the old clusters
            const std::int64_t id = next_id++;
            ch.push_back(Change{id, false, dup_vector(id % 100000, dims, 5)});
        }
        for (const Change &c : ch) {
            if (c.del) live.erase(c.id);
            else live[c.id] = c.vec;
        }
        TestSet next;
        bool ok = false;
        const bool threw = throws([&] { ok = incremental(next, cur, ch, &hp, 200 + round); });
        CHECK(!threw && ok);
        if (threw || !ok) return;
        std::uint64_t reached = 0, alive = 0;
        reachable_live(next.set, hnsw_open(next.set, true), reached, alive);
        if (reached != alive || alive != live.size())
            std::printf("  duplicates round %d: %llu of %llu live reachable, %llu live expected\n", round,
                        (unsigned long long)reached, (unsigned long long)alive, (unsigned long long)live.size());
        CHECK(reached == alive && alive == live.size());
        take(cur, next);
    }
    std::printf("  3 incremental rounds of 30000 duplicates on 8 threads: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// An incremental build into a file-backed buffer (vbuild build_in='file') gives the same bytes.
static void test_incremental_in_file()
{
    Live live;
    for (std::int64_t i = 0; i < 3000; ++i) live[i] = dup_vector(i, 12, 9);
    HnswParams hp;
    hp.m = 8;
    hp.threads = 1;
    TestSet base;
    full_build(base, live, Metric::L2, &hp);
    std::vector<Change> ch;
    for (std::int64_t i = 0; i < 200; ++i) ch.push_back(Change{i * 7, true, {}});
    for (std::int64_t i = 0; i < 300; ++i) ch.push_back(Change{5000 + i, false, dup_vector(i, 12, 9)});
    auto run = [&](SnapshotBuffer &out) {
        IncrementalBuilder b(base.set);
        for (const Change &c : ch) {
            if (c.del) { b.remove(c.id); continue; }
            float *row = b.begin_add(c.id, 12);
            std::memcpy(row, c.vec.data(), 12 * 4);
            b.end_add();
        }
        return b.finish(77, 5, out, &hp);
    };
    char tmpl[] = "/tmp/vvector_inc_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    SnapshotBuffer ram, file;
    file.back_with_file(dir);
    CHECK(run(ram) && run(file));
    CHECK(ram.size() == file.size() && std::memcmp(ram.data(), file.data(), ram.size()) == 0);
#if defined(__linux__)
    CHECK(file.file_backed());
#endif
    file.clear();
    std::system(("rm -rf " + dir).c_str());
}

// Milestone M7: the build in place on a copy-on-write mapping of the base file gives byte for byte
// the snapshot of the copying build; snapshot_diff finds exactly the changed blocks; the runs
// written over a copy of the base (what vload does) give the same file; seal_in_place gives the
// checksum without reading the file.
static void in_place_case(const std::string &dir, const char *name, Metric metric, bool graph, bool codes, bool ascending_ids,
                          std::uint32_t growth)
{
    Live live;
    for (std::int64_t i = 0; i < 4000; ++i) live[i * 10] = dup_vector(i, 12, 21);
    HnswParams hp;
    hp.m = 8;
    hp.threads = 1;
    TestSet base;
    {
        SnapshotBuilder b(metric);
        b.set_growth(growth);
        for (const auto &x : live) b.add(x.first, x.second.data(), 12);
        const CodeSection cs = sq8_code_section();
        const GraphSection gs = hnsw_graph_section(hp);
        b.finish(0, base.buffer, graph ? &gs : nullptr, codes ? &cs : nullptr);
        base.set = snapshot_open(base.buffer.data(), base.buffer.size(), true);
    }
    // The base in a cache directory, as on a node.
    const std::string index = std::string("ip_") + name;
    {
        CacheWriter w;
        w.begin(dir, index, 1);
        for (std::uint64_t off = 0; off < base.buffer.size(); off += CHUNK_BYTES)
            w.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(base.buffer.data()) + off,
                       std::min<std::uint64_t>(CHUNK_BYTES, base.buffer.size() - off));
        w.commit();
    }
    std::vector<Change> ch;
    for (std::int64_t i = 0; i < 150; ++i) ch.push_back(Change{(i + 1) * 70, true, {}});                        // deletes
    for (std::int64_t i = 0; i < 100; ++i) ch.push_back(Change{20000 + i * 30, false, dup_vector(9000 + i, 12, 21)}); // changes
    for (std::int64_t i = 0; i < 300; ++i)                                                                // adds
        ch.push_back(Change{ascending_ids ? 100000 + i : 5 + i * 20, false, dup_vector(7000 + i, 12, 21)});
    auto feed = [&](IncrementalBuilder &b) {
        for (const Change &c : ch) {
            if (c.del) { b.remove(c.id); continue; }
            float *row = b.begin_add(c.id, 12);
            std::memcpy(row, c.vec.data(), 12 * 4);
            b.end_add();
        }
    };
    const std::string path = snapshot_path(dir, index, 1);
    MappedSnapshot shared, copy;
    shared.open(path, true);
    copy.open(path, false, false, false, true);
    CHECK(copy.writable() != nullptr && copy.size() == base.buffer.size());

    // The copying build (the base's layout is kept when it has room).
    TestSet by_copy;
    {
        IncrementalBuilder b(shared.vectors());
        feed(b);
        CHECK(b.finish(77, 1, by_copy.buffer, graph ? &hp : nullptr));
        by_copy.set = snapshot_open(by_copy.buffer.data(), by_copy.buffer.size(), true);
    }
    // The build in place.
    IncrementalBuilder b(shared.vectors());
    feed(b);
    std::vector<ByteRange> cand;
    const InPlace r = b.finish_in_place(77, 1, copy.writable(), copy.size(), graph ? &hp : nullptr, cand);
    if (growth == 0) {
        CHECK(r == InPlace::NoRoom && cand.empty());
        CHECK(by_copy.set.has_capacity() && by_copy.set.capacity > by_copy.set.count);   // laid out anew, with room
        std::system(("rm -rf " + dir + "/" + index).c_str());
        return;
    }
    CHECK(r == InPlace::Built);
    CHECK(by_copy.buffer.size() == copy.size());
    const std::vector<ByteRange> runs = snapshot_diff(shared.data(), copy.data(), copy.size(), cand, CHUNK_BYTES);
    CHECK(!runs.empty() && runs.front().offset == 0);
    seal_in_place(shared.data(), copy.writable(), copy.size(), runs);
    CHECK(std::memcmp(by_copy.buffer.data(), copy.data(), copy.size()) == 0);
    // The runs hold every changed block and only changed blocks (after the seal the header run differs too).
    std::uint64_t sent = 0;
    bool minimal = true;
    for (const ByteRange &x : runs) {
        sent += x.bytes;
        minimal = minimal && x.offset % DIFF_BLOCK == 0 && x.bytes <= CHUNK_BYTES && x.offset / CHUNK_BYTES == (x.offset + x.bytes - 1) / CHUNK_BYTES;
        for (std::uint64_t blk = x.offset; blk < x.offset + x.bytes; blk += DIFF_BLOCK)
            minimal = minimal && std::memcmp(shared.data() + blk, copy.data() + blk, std::min(DIFF_BLOCK, x.offset + x.bytes - blk)) != 0;
    }
    CHECK(minimal && sent < copy.size() / 2);
    std::printf("  in place %-12s %5zu runs, %8llu of %8llu bytes\n", name, runs.size(),
                static_cast<unsigned long long>(sent), static_cast<unsigned long long>(copy.size()));
    // What vbuild and vload do: the runs packed into rows (small rows here, so there are several),
    // unpacked over a copy of the base, give the new file, verified in full.
    {
        CacheWriter w;
        w.begin(dir, index, 2, 1);
        std::uint64_t rows = 0, packed = 0, unpacked = 0;
        pack_runs(copy.data(), runs, 4096, [&](std::uint64_t first, const std::uint8_t *row, std::uint64_t len) {
            CHECK(len <= 4096 && (rows == 0 || first > 0));
            ++rows;
            packed += len;
            unpack_runs(row, len, [&](std::uint64_t off, const std::uint8_t *b, std::uint64_t n) {
                w.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(b), n);
                unpacked += n;
            });
        });
        CHECK(unpacked == sent && packed >= sent && rows >= (sent + 4095) / 4096);
        CHECK(w.commit() == copy.size());
        MappedSnapshot loaded;
        loaded.open(snapshot_path(dir, index, 2), true);
        CHECK(loaded.size() == copy.size() && std::memcmp(loaded.data(), copy.data(), copy.size()) == 0);
        CHECK(loaded.vectors().base_snapshot == 1 && loaded.vectors().count == by_copy.set.count);
        if (graph) hnsw_open(loaded.vectors(), true);
    }
    // The live vectors are those of a full build of the changed set.
    Live after = live;
    for (const Change &c : ch) { if (c.del) after.erase(c.id); else after[c.id] = c.vec; }
    TestSet full;
    full_build(full, after, metric, graph ? &hp : nullptr);
    CHECK(same_live(by_copy, full, after));
    // No change at all: Unchanged, the copy untouched.
    {
        MappedSnapshot again;
        again.open(path, false, false, false, true);
        IncrementalBuilder none(shared.vectors());
        none.remove(123456789);
        std::vector<ByteRange> c2;
        CHECK(none.finish_in_place(78, 1, again.writable(), again.size(), graph ? &hp : nullptr, c2) == InPlace::Unchanged);
        CHECK(std::memcmp(again.data(), shared.data(), shared.size()) == 0);
    }
    std::system(("rm -rf " + dir + "/" + index).c_str());
}

static void test_in_place()
{
    // snapshot_diff on small arrays: blocks, joins, chunk borders, overlapping candidates.
    {
        std::vector<std::uint8_t> a(DIFF_BLOCK * 20 + 64, 1), b = a;
        b[DIFF_BLOCK * 2 + 5] = 2;                 // block 2
        b[DIFF_BLOCK * 3] = 2;                     // block 3: joined with 2
        b[DIFF_BLOCK * 7 + 100] = 2;               // block 7
        b[DIFF_BLOCK * 8 + 1] = 2;                 // block 8: a chunk border between 7 and 8 with chunk = 8 blocks
        b[DIFF_BLOCK * 20 + 10] = 2;               // the last, short block
        std::vector<ByteRange> cand = {{DIFF_BLOCK * 2 + 10, DIFF_BLOCK}, {0, DIFF_BLOCK * 4}, {DIFF_BLOCK * 6, 100000}, {DIFF_BLOCK * 3, 10}};
        const std::vector<ByteRange> runs = snapshot_diff(a.data(), b.data(), a.size(), cand, DIFF_BLOCK * 8);
        CHECK(runs.size() == 4);
        CHECK(runs.size() == 4 && runs[0].offset == DIFF_BLOCK * 2 && runs[0].bytes == DIFF_BLOCK * 2);
        CHECK(runs.size() == 4 && runs[1].offset == DIFF_BLOCK * 7 && runs[1].bytes == DIFF_BLOCK);
        CHECK(runs.size() == 4 && runs[2].offset == DIFF_BLOCK * 8 && runs[2].bytes == DIFF_BLOCK);
        CHECK(runs.size() == 4 && runs[3].offset == DIFF_BLOCK * 20 && runs[3].bytes == 64);
        // Outside the candidates nothing is seen (block 5 is in none of them).
        b[DIFF_BLOCK * 5] = 2;
        CHECK(snapshot_diff(a.data(), b.data(), a.size(), cand, DIFF_BLOCK * 8).size() == 4);
        CHECK(snapshot_diff(a.data(), b.data(), a.size(), {{0, a.size()}}, DIFF_BLOCK * 8).size() == 5);
    }
    char tmpl[] = "/tmp/vvector_inplace_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    in_place_case(dir, "flat", Metric::L2, false, false, true, 5);
    in_place_case(dir, "flat_rnd", Metric::L2, false, false, false, 5);
    in_place_case(dir, "hnsw", Metric::L2, true, false, true, 5);
    in_place_case(dir, "hnsw_rnd", Metric::Cosine, true, false, false, 5);
    in_place_case(dir, "sq8", Metric::L2, false, true, true, 5);
    in_place_case(dir, "hnsw_sq8", Metric::Dot, true, true, false, 5);
    in_place_case(dir, "no_room", Metric::L2, true, false, true, 0);
    std::system(("rm -rf " + dir).c_str());
}

// pack_runs / unpack_runs (the rows a patch travels in): every byte of every run comes back at its
// offset, in order, for row capacities from one record to more than the patch; a run longer than a
// row is split; malformed rows are refused.
static void test_pack_runs()
{
    std::vector<std::uint8_t> data(100000);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i * 7 + 3);
    const std::vector<ByteRange> runs = {{0, 512}, {1024, 1}, {2048, 5000}, {90000, 10000}, {8192, 512}};
    std::uint64_t total = 0;
    for (const ByteRange &r : runs) total += r.bytes;
    for (std::uint64_t capacity : {13ull, 100ull, 600ull, 4096ull, 1048576ull}) {
        std::uint64_t at_run = 0, at_off = runs[0].offset, got = 0, rows = 0;
        bool ok = true;
        pack_runs(data.data(), runs, capacity, [&](std::uint64_t first, const std::uint8_t *row, std::uint64_t len) {
            ok = ok && len <= capacity && first == at_off;
            ++rows;
            unpack_runs(row, len, [&](std::uint64_t off, const std::uint8_t *b, std::uint64_t n) {
                ok = ok && off == at_off && n > 0 && std::memcmp(b, data.data() + off, n) == 0;
                got += n;
                at_off += n;
                if (at_off == runs[at_run].offset + runs[at_run].bytes && at_run + 1 < runs.size()) at_off = runs[++at_run].offset;
            });
        });
        CHECK(ok && got == total);
        CHECK(capacity < total ? rows > 1 : rows == 1);
    }
    CHECK(throws([] { std::uint8_t r[8] = {0}; unpack_runs(r, 8, [](std::uint64_t, const std::uint8_t *, std::uint64_t) {}); }, "record header"));
    CHECK(throws([] { std::uint8_t r[16] = {0}; r[8] = 9; unpack_runs(r, 16, [](std::uint64_t, const std::uint8_t *, std::uint64_t) {}); }, "inside a record"));
    CHECK(throws([] { std::uint8_t r[1] = {0}; unpack_runs(r, 0, [](std::uint64_t, const std::uint8_t *, std::uint64_t) {}); }, "empty"));
}

int main(int argc, char **argv)
{
    std::string dir;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--dir=", 6) == 0) dir = argv[i] + 6;
    test_pack_runs();
    test_flags_and_nothing();
    rounds("flat l2", Metric::L2, false);
    rounds("flat cosine", Metric::Cosine, false);
    rounds("hnsw l2", Metric::L2, true);
    rounds("hnsw cosine", Metric::Cosine, true);
    rounds("hnsw dot", Metric::Dot, true);
    test_parallel_duplicates();
    test_incremental_in_file();
    test_in_place();
    test_sift(dir);
    return finish("test_delta");
}
