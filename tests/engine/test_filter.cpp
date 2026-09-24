// Filtered search and range search (milestone M5).
// Filtered: both paths (the allowed rows searched exactly, and the search with the rest masked) give
// exactly what the flat search over the allowed rows gives, with journal rows and masks, for every
// metric, flat and HNSW, float rows and sq8 codes; an allow-list in any order with repeats; an empty
// allow-list returns only journal rows; a position beyond the count is refused.
// Range: on HNSW with a radius and a large k the walk grows while its candidates are within the
// radius: the results are within it, with the recall of the flat search, independent of threads.
// With --dir=DIR (SIFT1M) it also measures filtered recall and time at selectivities 0.01%, 1% and
// 50% on both paths and asserts recall@10 >= 0.95 at ef_search 100.
#include "check.h"

#include "../../src/engine/flat.h"
#include "../../src/engine/hnsw.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/search.h"
#include "../../src/engine/sq8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

using namespace vvector;

// count vectors of dims elements in [-1, 1), ids 7, 12, 17, ..., flat or HNSW, with or without codes.
static void build_set(TestSet &t, std::uint64_t count, std::uint32_t dims, Metric metric, bool graph, bool codes,
                      std::uint64_t seed = 3)
{
    SnapshotBuilder b(metric);
    std::vector<float> v(dims);
    for (std::uint64_t i = 0; i < count; ++i) {
        test_vector(i, dims, seed, true, v.data());
        b.add(static_cast<std::int64_t>(7 + 5 * i), v.data(), dims);
    }
    HnswParams p;
    p.m = 12;
    p.ef_construction = 60;
    p.threads = 2;
    const GraphSection g = hnsw_graph_section(p);
    const CodeSection c = sq8_code_section();
    b.finish(0, t.buffer, graph ? &g : nullptr, codes ? &c : nullptr);
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

static bool is_set(const std::vector<std::uint64_t> &bits, std::uint64_t pos) { return (bits[pos >> 6] >> (pos & 63) & 1u) != 0; }

// The filtered search against the flat search over the allowed rows, on both paths.
static void filtered_exact(Metric metric, bool graph, bool codes)
{
    const std::string what = std::string(metric_name(metric)) + (graph ? " hnsw" : " flat") + (codes ? " sq8" : "");
    TestSet t;
    build_set(t, 3000, 24, metric, graph, codes);
    const std::uint64_t nq = 12, n = t.set.count;
    const std::vector<float> qs = queries(nq, t.set, 11);
    FlatSearch s;
    s.metric = metric;
    s.stride = t.set.row_stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = 10;
    s.threads = 3;

    // Journal rows (the caller passes only allowed ones) and a mask of every 7th position.
    std::vector<float> jrows(4 * s.stride, 0.0f);
    std::vector<std::int64_t> jids = {1, 2, 3, 99999};
    for (int j = 0; j < 4; ++j) {
        test_vector(5000 + j, t.set.dims, 3, true, jrows.data() + j * s.stride);
        if (metric == Metric::Cosine) normalize(jrows.data() + j * s.stride, t.set.dims);
    }
    const RowBlock journal{jrows.data(), jids.data(), jids.size(), nullptr};
    std::vector<std::uint64_t> skip((n + 63) / 64, 0);
    for (std::uint64_t p = 0; p < n; p += 7) skip[p >> 6] |= 1ull << (p & 63);

    Rng rng(17);
    for (std::uint64_t allowed : {3ull, 30ull, 1500ull}) {
        // Random positions, unsorted, with repeats.
        std::vector<std::uint32_t> allow;
        for (std::uint64_t i = 0; i < allowed; ++i) allow.push_back(static_cast<std::uint32_t>(rng.below(static_cast<std::int64_t>(n))));
        allow.push_back(allow[0]);
        // The reference: every position outside the list, or in skip, is masked.
        std::vector<std::uint64_t> ref_skip((n + 63) / 64, ~0ull);
        for (std::uint32_t p : allow) ref_skip[p >> 6] &= ~(1ull << (p & 63));
        for (std::size_t w = 0; w < ref_skip.size(); ++w) ref_skip[w] |= skip[w];
        const RowBlock ref_blocks[2] = {RowBlock{t.set.vectors, t.set.ids, n, ref_skip.data()}, journal};
        for (int radius = 0; radius < 2; ++radius) {
            std::vector<Neighbor> ref, got;
            std::vector<std::uint32_t> rc, gc;
            s.has_radius = false;
            flat_search(s, ref_blocks, 2, ref, rc);
            if (radius) {                 // a radius that cuts the list of the first query
                s.has_radius = true;
                s.radius = key_to_score(metric, ref[std::min<std::uint32_t>(rc[0], 5) - 1].key);
                flat_search(s, ref_blocks, 2, ref, rc);
            }
            SearchPlan p;
            p.graph = graph;
            p.codes = codes;
            p.filtered = true;
            p.allow = allow.data();
            p.n_allow = allow.size();
            p.ef = 32;
            p.exact_below = -1;           // 3000 positions: always below the limit, the rows exactly
            index_search(s, t.set, p, skip.data(), &journal, got, gc);
            CHECK(same(ref, rc, got, gc, s.k));
            if (!same(ref, rc, got, gc, s.k)) std::printf("    exact path %s, %llu allowed, radius %d\n", what.c_str(), (unsigned long long)allowed, radius);
            // The masked path: exact with every candidate (ef = count, every code candidate rescored).
            p.exact_below = 0;
            p.ef = static_cast<std::uint32_t>(n);
            p.oversampling = double(n);
            index_search(s, t.set, p, skip.data(), &journal, got, gc);
            CHECK(same(ref, rc, got, gc, s.k));
            if (!same(ref, rc, got, gc, s.k)) std::printf("    masked path %s, %llu allowed, radius %d\n", what.c_str(), (unsigned long long)allowed, radius);
            // The masked path with a small ef: only allowed positions and journal rows come back.
            p.ef = 16;
            p.oversampling = 2;
            index_search(s, t.set, p, skip.data(), &journal, got, gc);
            for (std::uint64_t q = 0; q < nq; ++q)
                for (std::uint32_t i = 0; i < gc[q]; ++i) {
                    const std::int64_t id = got[q * s.k + i].id;
                    const std::int64_t pos = t.set.find(id);
                    CHECK(std::find(jids.begin(), jids.end(), id) != jids.end() || (pos >= 0 && !is_set(ref_skip, pos)));
                }
        }
    }
    s.has_radius = false;

    // An empty allow-list: only the journal's rows.
    SearchPlan p;
    p.graph = graph;
    p.codes = codes;
    p.filtered = true;
    std::vector<Neighbor> got;
    std::vector<std::uint32_t> gc;
    index_search(s, t.set, p, nullptr, nullptr, got, gc);
    CHECK(gc[0] == 0 && gc[nq - 1] == 0);
    index_search(s, t.set, p, nullptr, &journal, got, gc);
    CHECK(gc[0] == 4);
    p.exact_below = 0;
    index_search(s, t.set, p, nullptr, &journal, got, gc);
    CHECK(gc[0] == 4);
    // A position beyond the count.
    const std::uint32_t bad = static_cast<std::uint32_t>(n);
    p.allow = &bad;
    p.n_allow = 1;
    CHECK(throws([&] { index_search(s, t.set, p, nullptr, nullptr, got, gc); }, "not in the snapshot"));
}

// find_sorted on a snapshot without an id_index (ids ascending): what find gives, for ids in the
// snapshot, between its ids, below and above them.
static void test_find_sorted()
{
    TestSet t;
    build_set(t, 3000, 8, Metric::L2, false, false);
    std::vector<std::int64_t> want;
    for (std::int64_t id = -10; id < 7 + 5 * 3000 + 50; id += 3) want.push_back(id);
    std::vector<std::int64_t> got(want.size());
    t.set.find_sorted(want.data(), want.size(), got.data());
    bool same_pos = true;
    for (std::size_t i = 0; i < want.size(); ++i) same_pos &= got[i] == t.set.find(want[i]);
    CHECK(same_pos);
    t.set.find_sorted(want.data(), 0, got.data());       // nothing to find
}

static void test_limit()
{
    CHECK(filter_exact_limit(100, 1000) == 10000);
    CHECK(filter_exact_limit(100, 1000000) == 80000);
    CHECK(filter_exact_limit(400, 1000000) == 160000);
    CHECK(filter_exact_limit(100, 100000000) == 800000);
}

// Range search on HNSW: radius with a k far above ef.
static void range_search(Metric metric, bool codes)
{
    const std::string what = std::string(metric_name(metric)) + (codes ? " sq8" : "");
    TestSet t;
    build_set(t, 3000, 16, metric, true, codes, 5);
    const std::uint64_t nq = 20, n = t.set.count;
    const std::vector<float> qs = queries(nq, t.set, 23);
    FlatSearch s;
    s.metric = metric;
    s.stride = t.set.row_stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = static_cast<std::uint32_t>(n);
    s.threads = 4;
    const RowBlock all{t.set.vectors, t.set.ids, n, nullptr};

    // The radius of the 40th neighbour of query 0: tens of vectors within it for most queries.
    std::vector<Neighbor> ref, got, got1;
    std::vector<std::uint32_t> rc, gc, gc1;
    flat_search(s, &all, 1, ref, rc);
    s.has_radius = true;
    s.radius = key_to_score(metric, ref[39].key);
    flat_search(s, &all, 1, ref, rc);

    SearchPlan p;
    p.graph = true;
    p.codes = codes;
    p.ef = 16;
    p.oversampling = 2;
    index_search(s, t.set, p, nullptr, nullptr, got, gc);
    std::uint64_t hit = 0, total = 0;
    for (std::uint64_t q = 0; q < nq; ++q) {
        total += rc[q];
        for (std::uint32_t i = 0; i < gc[q]; ++i) {
            CHECK(within_radius(s, got[q * s.k + i].key));
            for (std::uint32_t r = 0; r < rc[q]; ++r) hit += got[q * s.k + i].id == ref[q * s.k + r].id;
        }
    }
    const double recall = total ? double(hit) / double(total) : 1.0;
    std::printf("  range search %s: %llu vectors within the radius over %llu queries, recall %.4f\n", what.c_str(),
                (unsigned long long)total, (unsigned long long)nq, recall);
    CHECK(total > nq * 10);
    CHECK(recall >= 0.95);
    // Threads change nothing.
    s.threads = 1;
    index_search(s, t.set, p, nullptr, nullptr, got1, gc1);
    CHECK(same(got, gc, got1, gc1, s.k));
    s.threads = 4;

    // A radius around everything: the walk grows to k = count and finds exactly the flat result.
    s.radius = metric == Metric::L2 || metric == Metric::L1 ? 1e30 : -1e30;
    flat_search(s, &all, 1, ref, rc);
    p.oversampling = 1;               // k = count candidates: every position rescored
    index_search(s, t.set, p, nullptr, nullptr, got, gc);
    CHECK(rc[0] == n);
    CHECK(same(ref, rc, got, gc, s.k));
}

// SIFT1M: filtered recall and time at three selectivities, both paths.
static void test_sift(const std::string &dir)
{
    if (dir.empty()) {
        std::printf("  SIFT1M filtered search: skipped (make test DATA_DIR=<dir with sift_base.fvecs, sift_query.fvecs>)\n");
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
    nq = std::min<std::uint64_t>(nq, 200);        // the masked graph at 0.01% is slow by nature
    SnapshotBuilder b(Metric::L2);
    for (std::uint64_t i = 0; i < n; ++i) b.add(static_cast<std::int64_t>(i), base.data() + i * dims, dims);
    std::vector<float>().swap(base);
    HnswParams hp;
    hp.threads = resolve_threads(0);
    const GraphSection g = hnsw_graph_section(hp);
    const CodeSection c = sq8_code_section();
    TestSet t;
    b.finish(0, t.buffer, &g, &c);
    t.set = snapshot_open(t.buffer.data(), t.buffer.size(), false);
    const std::uint32_t stride = t.set.row_stride;
    std::vector<float> qs(nq * stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) std::memcpy(qs.data() + i * stride, query.data() + i * dims, dims * 4);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = 10;
    s.threads = hp.threads;

    Rng rng(29);
    for (double sel : {0.0001, 0.01, 0.5}) {
        std::vector<std::uint32_t> allow;
        for (std::uint64_t i = 0; i < n; ++i)
            if (rng.unit() < sel) allow.push_back(static_cast<std::uint32_t>(i));
        SearchPlan exact;
        exact.filtered = true;
        exact.allow = allow.data();
        exact.n_allow = allow.size();
        exact.exact_below = static_cast<std::int64_t>(n) + 1;
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        auto timed = [&](const SearchPlan &p, std::vector<Neighbor> &out, std::vector<std::uint32_t> &cnt) {
            const auto a = std::chrono::steady_clock::now();
            index_search(s, t.set, p, nullptr, nullptr, out, cnt);
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        };
        const double t_exact = timed(exact, ref, rc);
        for (bool codes : {false, true}) {
            SearchPlan p = exact;
            p.graph = true;
            p.codes = codes;
            p.ef = 100;
            p.oversampling = 2;
            p.exact_below = 0;
            const double t_masked = timed(p, got, gc);
            std::uint64_t hit = 0, total = 0;
            for (std::uint64_t q = 0; q < nq; ++q) {
                total += rc[q];
                for (std::uint32_t i = 0; i < gc[q]; ++i)
                    for (std::uint32_t r = 0; r < rc[q]; ++r) hit += got[q * 10 + i].id == ref[q * 10 + r].id;
            }
            const double recall = total ? double(hit) / double(total) : 1.0;
            std::printf("  SIFT1M filter %.2f%% (%zu allowed)%s: exact path %.1f q/s, masked HNSW ef 100 %.1f q/s recall@10 %.4f\n",
                        sel * 100, allow.size(), codes ? " sq8" : "", nq / t_exact, nq / t_masked, recall);
            if (allow.size() >= filter_exact_limit(100, n)) CHECK(recall >= 0.95);   // what the default rule chooses
        }
    }
}

int main(int argc, char **argv)
{
    std::string dir;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--dir=", 6) == 0) dir = argv[i] + 6;
    test_limit();
    test_find_sorted();
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::Dot, Metric::L1})
        for (bool graph : {false, true})
            for (bool codes : {false, true}) filtered_exact(m, graph, codes);
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::Dot, Metric::L1})
        for (bool codes : {false, true}) range_search(m, codes);
    test_sift(dir);
    return finish("test_filter");
}
