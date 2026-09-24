// sq8 (scalar quantisation, milestone M4): the integer kernels are exact; training, coding and the
// section follow docs/format.md; damaged sections are refused; a search that rescores every
// candidate gives exactly what the float search gives (every metric, flat and HNSW, journal rows,
// masks, radius); without rescoring the scores stay within the error the scale allows; results do
// not depend on the number of threads; an incremental build keeps the range of its base.
// With --dir=DIR (SIFT1M) it also asserts that rescoring loses at most 0.01 of recall@10 against
// the float HNSW search at ef_search 100, and prints the recall without rescoring.
#include "check.h"

#include "../../src/engine/delta.h"
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

static std::uint64_t naive_sum(Metric m, const std::uint8_t *a, const std::uint8_t *b, std::uint32_t n)
{
    std::uint64_t s = 0;
    for (std::uint32_t j = 0; j < n; ++j) {
        const std::int64_t x = a[j], y = b[j];
        if (m == Metric::L2) s += std::uint64_t((x - y) * (x - y));
        else if (m == Metric::L1) s += std::uint64_t(x > y ? x - y : y - x);
        else s += std::uint64_t(x * y);
    }
    return s;
}

static void test_kernels()
{
    const Metric metrics[] = {Metric::L2, Metric::L1, Metric::Dot};
    Rng rng(5);
    for (std::uint32_t dims : {1u, 15u, 16u, 17u, 100u, 128u, 768u, 1536u}) {
        const std::uint32_t stride = row_stride_for(dims);
        const std::uint32_t n = 37;
        std::vector<std::uint8_t> rows(std::uint64_t(n) * stride, 0), q(stride, 0);
        for (std::uint32_t i = 0; i < n; ++i)
            for (std::uint32_t d = 0; d < dims; ++d) rows[std::uint64_t(i) * stride + d] = static_cast<std::uint8_t>(rng.below(256));
        for (std::uint32_t d = 0; d < dims; ++d) q[d] = static_cast<std::uint8_t>(rng.below(256));
        std::vector<std::uint32_t> pos(n);
        for (std::uint32_t i = 0; i < n; ++i) pos[i] = (i * 17) % n;
        for (Metric m : metrics) {
            std::vector<std::uint32_t> s1(n), sg(n), s4(4 * n);
            sq8_sums_1q(m, rows.data(), n, stride, q.data(), s1.data());
            const std::uint8_t *q4[4] = {rows.data(), q.data(), rows.data() + stride, q.data()};
            sq8_sums_4q(m, rows.data(), n, stride, q4, s4.data());
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint8_t *r = rows.data() + std::uint64_t(i) * stride;
                CHECK(s4[i] == sq8_sum(m, r, q4[0], stride) && s4[n + i] == s1[i] && s4[2 * n + i] == sq8_sum(m, r, q4[2], stride) &&
                      s4[3 * n + i] == s1[i]);
            }
            sq8_sums_gather(m, rows.data(), stride, pos.data(), n, q.data(), sg.data());
            bool ok = true;
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint8_t *r = rows.data() + std::uint64_t(i) * stride;
                ok = ok && s1[i] == naive_sum(m, r, q.data(), stride) && sq8_sum(m, r, q.data(), stride) == s1[i];
                ok = ok && sg[i] == s1[pos[i]];
            }
            CHECK(ok);
        }
    }
    // The largest sums: 32768 elements of 255 against 0 (l2, l1) and 255 against 255 (dot).
    const std::uint32_t big = MAX_DIMS;
    std::vector<std::uint8_t> full(big, 255), zero(big, 0);
    CHECK(sq8_sum(Metric::L2, full.data(), zero.data(), big) == 255u * 255u * big);
    CHECK(sq8_sum(Metric::L1, zero.data(), full.data(), big) == 255u * big);
    CHECK(sq8_sum(Metric::Dot, full.data(), full.data(), big) == 255u * 255u * big);
}

static void test_coding()
{
    // Training: 1000 rows of 100 elements 0 .. 99999 in order: the quantiles are near the ends.
    TestSet t;
    {
        SnapshotBuilder b(Metric::L2);
        std::vector<float> v(100);
        for (int i = 0; i < 1000; ++i) {
            for (int d = 0; d < 100; ++d) v[d] = static_cast<float>(i * 100 + d);
            b.add(i, v.data(), 100);
        }
        b.finish(0, t.buffer);
        t.set = snapshot_open(t.buffer.data(), t.buffer.size(), true);
    }
    const Sq8Range r = sq8_train(t.set);
    CHECK(r.sample == 100000);
    CHECK(r.offset == 99.0f);                             // floor(0.001 x 99999) = 99
    CHECK(std::fabs(r.scale - (99900.0f - 99.0f) / 255.0f) < 1e-3f);   // ceil(0.999 x 99999) = 99900
    std::vector<std::uint8_t> code(16, 0);
    const float v[5] = {99.0f, 99900.0f, -5.0f, 1e9f, std::nanf("")};
    const std::uint32_t sum = sq8_encode(r, v, 5, code.data());
    CHECK(code[0] == 0 && code[1] == 255 && code[2] == 0 && code[3] == 255 && code[4] == 0);
    CHECK(sum == 510);
    for (int d = 5; d < 16; ++d) CHECK(code[d] == 0);

    // Constant data: scale 1, offset the value, every code 0.
    TestSet c;
    {
        SnapshotBuilder b(Metric::L2);
        const float x[3] = {2.5f, 2.5f, 2.5f};
        for (int i = 0; i < 10; ++i) b.add(i, x, 3);
        const CodeSection cs = sq8_code_section();
        b.finish(0, c.buffer, nullptr, &cs);
        c.set = snapshot_open(c.buffer.data(), c.buffer.size(), true);
    }
    const Sq8Codes cc = sq8_open(c.set, true);
    CHECK(cc.range.scale == 1.0f && cc.range.offset == 2.5f);
    CHECK(cc.sums[0] == 0 && cc.row(9)[0] == 0);
}

// count vectors of dims elements in [-1, 1), ids 7, 12, 17, ..., with sq8 codes, flat or HNSW.
static void build_coded(TestSet &t, std::uint64_t count, std::uint32_t dims, Metric metric, bool graph, int threads = 1,
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
    p.threads = threads;
    const GraphSection g = hnsw_graph_section(p);
    const CodeSection c = sq8_code_section();
    b.finish(0, t.buffer, graph ? &g : nullptr, &c);
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

static void test_section()
{
    TestSet t;
    build_coded(t, 500, 40, Metric::L2, true);
    CHECK((t.set.flags & FLAG_SQ8) && (t.set.flags & FLAG_HNSW));
    CHECK(t.set.sq8_bytes == sq8_section_bytes(500, 48));
    const Sq8Codes c = sq8_open(t.set, true);
    CHECK(c.count == 500 && c.stride == 48 && c.dims == 40);
    bool coded = true;
    std::vector<std::uint8_t> code(48, 0);
    for (std::uint64_t i = 0; i < 500; ++i) {
        std::fill(code.begin(), code.end(), 0);
        const std::uint32_t sum = sq8_encode(c.range, t.set.vector(i), 40, code.data());
        coded = coded && sum == c.sums[i] && std::memcmp(code.data(), c.row(i), 48) == 0;
    }
    CHECK(coded);

    // Damaged copies: a wrong sum, a padding code, a wrong count. Opened without the checksum.
    auto damaged = [&](const std::function<void(std::uint8_t *)> &hurt, const char *message) {
        SnapshotBuffer copy;
        copy.allocate(t.buffer.size());
        std::memcpy(copy.data(), t.buffer.data(), t.buffer.size());
        const VectorSet s = snapshot_open(copy.data(), copy.size(), false);
        hurt(copy.data() + (s.sq8 - copy.data()));
        CHECK(throws([&] { sq8_open(snapshot_open(copy.data(), copy.size(), false), true); }, message));
    };
    const std::uint64_t sums_at = (64 + 500 * 48 + 63) / 64 * 64;
    damaged([&](std::uint8_t *sq) { sq[sums_at] ^= 1; }, "code sum");
    damaged([](std::uint8_t *sq) { sq[64 + 45] = 1; }, "padding codes");
    damaged([](std::uint8_t *sq) { sq[16] ^= 1; }, "count does not match");
    damaged([](std::uint8_t *sq) { std::memset(sq, 0, 4); }, "scale or offset");
}

// With every row a candidate, rescoring gives exactly the float search: same ids, same key bits.
static void exact_with_all_candidates(Metric metric, bool graph)
{
    TestSet t;
    build_coded(t, 3000, 40, metric, graph, 2);
    const std::uint64_t nq = 20;
    const std::vector<float> qs = queries(nq, t.set, 9);
    FlatSearch s;
    s.metric = metric;
    s.stride = t.set.row_stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = 10;
    s.threads = 3;

    // A journal of 3 rows beside the snapshot, and a mask of 100 positions.
    std::vector<float> jrows(3 * s.stride, 0.0f);
    std::vector<std::int64_t> jids = {1, 2, 99999};
    for (int j = 0; j < 3; ++j) {
        test_vector(5000 + j, t.set.dims, 3, true, jrows.data() + j * s.stride);
        if (metric == Metric::Cosine) normalize(jrows.data() + j * s.stride, t.set.dims);
    }
    const RowBlock journal{jrows.data(), jids.data(), 3, nullptr};
    std::vector<std::uint64_t> skip((t.set.count + 63) / 64, 0);
    for (std::uint64_t p = 0; p < t.set.count; p += 30) skip[p >> 6] |= 1ull << (p & 63);

    SearchPlan exact;
    SearchPlan coded;
    coded.graph = graph;
    coded.ef = 3000;
    coded.codes = true;
    coded.rescore = true;
    coded.oversampling = 300;          // 3000 candidates: every row
    std::vector<Neighbor> a, b;
    std::vector<std::uint32_t> ca, cb;
    index_search(s, t.set, exact, skip.data(), &journal, a, ca);
    index_search(s, t.set, coded, skip.data(), &journal, b, cb);
    CHECK(same(a, ca, b, cb, s.k));

    s.has_radius = true;               // the radius acts on the exact scores
    s.radius = key_to_score(metric, a[4].key);
    index_search(s, t.set, exact, skip.data(), &journal, a, ca);
    index_search(s, t.set, coded, skip.data(), &journal, b, cb);
    CHECK(same(a, ca, b, cb, s.k));
    if (!same(a, ca, b, cb, s.k)) std::printf("    exact with all candidates failed: %s %s\n", metric_name(metric), graph ? "hnsw" : "flat");
}

// Without rescoring: the returned scores are the approximate ones, within the error of the scale.
static void approximate_scores(Metric metric)
{
    TestSet t;
    build_coded(t, 2000, 64, metric, false);
    const Sq8Codes c = sq8_open(t.set, false);
    const std::uint64_t nq = 10;
    const std::vector<float> qs = queries(nq, t.set, 11);
    FlatSearch s;
    s.metric = metric;
    s.stride = t.set.row_stride;
    s.queries = qs.data();
    s.n_queries = nq;
    s.k = 10;
    SearchPlan p;
    p.codes = true;
    p.rescore = false;
    std::vector<Neighbor> got;
    std::vector<std::uint32_t> count;
    index_search(s, t.set, p, nullptr, nullptr, got, count);
    const double h = c.range.scale / 2.0 + 0.01;       // per element: rounding, plus the clipped tails
    bool within = true;
    for (std::uint64_t q = 0; q < nq; ++q) {
        CHECK(count[q] == 10);
        for (std::uint32_t i = 0; i < count[q]; ++i) {
            const Neighbor &nb = got[q * 10 + i];
            const float *row = t.set.vector(std::uint64_t(t.set.find(nb.id)));
            const float *qv = qs.data() + q * s.stride;
            const double approx = key_to_score(metric, nb.key);
            const double exact = key_to_score(metric, distance_key(metric, qv, row, s.stride));
            double bound = 0;
            if (metric == Metric::L2) bound = 2 * h * std::sqrt(double(t.set.dims));
            else if (metric == Metric::L1) bound = 2 * h * t.set.dims;
            else {
                double sx = 0, sq = 0;
                for (std::uint32_t d = 0; d < t.set.dims; ++d) { sx += std::fabs(row[d]); sq += std::fabs(qv[d]); }
                bound = h * (sx + sq) + t.set.dims * h * h;
            }
            within = within && std::fabs(approx - exact) <= bound;
        }
    }
    CHECK(within);
    if (!within) std::printf("    approximate scores out of bound: %s\n", metric_name(metric));
}

static void test_threads()
{
    for (bool graph : {false, true}) {
        TestSet t;
        build_coded(t, 4000, 32, Metric::Dot, graph, 4);
        const std::uint64_t nq = 64;
        const std::vector<float> qs = queries(nq, t.set, 21);
        FlatSearch s;
        s.metric = Metric::Dot;
        s.stride = t.set.row_stride;
        s.queries = qs.data();
        s.n_queries = nq;
        s.k = 7;
        SearchPlan p;
        p.graph = graph;
        p.ef = 50;
        p.codes = true;
        p.oversampling = 3;
        std::vector<Neighbor> a, b;
        std::vector<std::uint32_t> ca, cb;
        s.threads = 1;
        index_search(s, t.set, p, nullptr, nullptr, a, ca);
        s.threads = 8;
        index_search(s, t.set, p, nullptr, nullptr, b, cb);
        CHECK(same(a, ca, b, cb, s.k));
        p.rescore = false;
        s.threads = 1;
        index_search(s, t.set, p, nullptr, nullptr, a, ca);
        s.threads = 8;
        index_search(s, t.set, p, nullptr, nullptr, b, cb);
        CHECK(same(a, ca, b, cb, s.k));
    }
}

// An incremental build keeps the range of its base and codes the appended rows with it.
static void test_incremental()
{
    for (bool graph : {false, true}) {
        TestSet t;
        build_coded(t, 2000, 24, Metric::Cosine, graph, 2);
        const Sq8Codes base = sq8_open(t.set, true);
        IncrementalBuilder ib(t.set);
        std::vector<float> v(24);
        for (int i = 0; i < 300; ++i) {                 // new ids, with values beyond the base's range
            test_vector(9000 + i, 24, 4, true, v.data());
            for (float &x : v) x *= 3.0f;
            float *row = ib.begin_add(100000 + i, 24);
            std::memcpy(row, v.data(), sizeof(float) * 24);
            ib.end_add();
        }
        for (int i = 0; i < 100; ++i) {                 // changed vectors
            test_vector(7000 + i, 24, 4, true, v.data());
            float *row = ib.begin_add(7 + 5 * (i * 7), 24);
            std::memcpy(row, v.data(), sizeof(float) * 24);
            ib.end_add();
        }
        for (int i = 0; i < 50; ++i) ib.remove(7 + 5 * (1000 + i));
        HnswParams p;
        p.m = 12;
        p.ef_construction = 60;
        TestSet n;
        CHECK(ib.finish(1, 5, n.buffer, graph ? &p : nullptr));
        n.set = snapshot_open(n.buffer.data(), n.buffer.size(), true);
        CHECK(n.set.flags & FLAG_SQ8);
        const Sq8Codes c = sq8_open(n.set, true);
        CHECK(c.range.scale == base.range.scale && c.range.offset == base.range.offset && c.range.sample == base.range.sample);
        CHECK(c.count == 2400);
        CHECK(std::memcmp(c.codes, base.codes, base.count * base.stride) == 0);
        bool coded = true;
        std::vector<std::uint8_t> code(c.stride, 0);
        for (std::uint64_t i = base.count; i < c.count; ++i) {
            std::fill(code.begin(), code.end(), 0);
            coded = coded && sq8_encode(base.range, n.set.vector(i), 24, code.data()) == c.sums[i] &&
                    std::memcmp(code.data(), c.row(i), c.stride) == 0;
        }
        CHECK(coded);

        // Every row a candidate: the incremental snapshot answers exactly as the float search.
        const std::vector<float> qs = queries(15, n.set, 31);
        FlatSearch s;
        s.metric = Metric::Cosine;
        s.stride = n.set.row_stride;
        s.queries = qs.data();
        s.n_queries = 15;
        s.k = 10;
        SearchPlan exact, coded_plan;
        coded_plan.graph = graph;
        coded_plan.ef = 2400;
        coded_plan.codes = true;
        coded_plan.oversampling = 240;
        std::vector<Neighbor> a, b;
        std::vector<std::uint32_t> ca, cb;
        index_search(s, n.set, exact, n.set.tombstone_bits, nullptr, a, ca);     // callers pass the tombstones
        index_search(s, n.set, coded_plan, n.set.tombstone_bits, nullptr, b, cb);
        CHECK(same(a, ca, b, cb, s.k));
    }
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
    const CodeSection c = sq8_code_section();
    TestSet t;
    b.finish(0, t.buffer, &g, &c);
    t.set = snapshot_open(t.buffer.data(), t.buffer.size(), false);
    const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Sq8Codes codes = sq8_open(t.set, true);
    std::printf("  SIFT1M with sq8: build %.1f s, range %.3f .. %.3f, codes %.1f MB\n", build_s, codes.range.offset,
                codes.range.offset + 255.0 * codes.range.scale, t.set.sq8_bytes / 1048576.0);
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
    auto recall = [&](const SearchPlan &plan, double &seconds) {
        std::vector<Neighbor> got;
        std::vector<std::uint32_t> gc;
        const auto a = std::chrono::steady_clock::now();
        index_search(s, t.set, plan, nullptr, nullptr, got, gc);
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        std::uint64_t hit = 0;
        for (std::uint64_t q = 0; q < nq; ++q)
            for (std::uint32_t i = 0; i < gc[q]; ++i)
                for (std::uint32_t r = 0; r < 10; ++r) hit += got[q * 10 + i].id == truth[q * tdims + r];
        return hit / (10.0 * nq);
    };
    for (bool graph : {true, false}) {
        SearchPlan fp;
        fp.graph = graph;
        fp.ef = 100;
        SearchPlan rs = fp, raw = fp;
        rs.codes = raw.codes = true;
        rs.oversampling = 2;
        raw.rescore = false;
        double t_fp, t_rs, t_raw;
        const double r_fp = recall(fp, t_fp), r_rs = recall(rs, t_rs), r_raw = recall(raw, t_raw);
        std::printf("  SIFT1M %s: recall@10 float %.4f (%.0f q/s), sq8 rescored x2 %.4f (%.0f q/s), sq8 without rescoring %.4f (%.0f q/s)\n",
                    graph ? "HNSW ef 100" : "flat", r_fp, nq / t_fp, r_rs, nq / t_rs, r_raw, nq / t_raw);
        CHECK(r_rs >= r_fp - 0.01);
    }
}

int main(int argc, char **argv)
{
    std::string dir;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--dir=", 6) == 0) dir = argv[i] + 6;
    test_kernels();
    test_coding();
    test_section();
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::Dot, Metric::L1}) {
        exact_with_all_candidates(m, false);
        exact_with_all_candidates(m, true);
        approximate_scores(m);
    }
    test_threads();
    test_incremental();
    test_sift(dir);
    return finish("test_sq8");
}
