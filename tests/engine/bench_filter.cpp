// Benchmark of filtered search (milestone M5) without Vertica: for allow-lists of several
// selectivities, the two paths of search.h (the allowed rows searched exactly; the graph with the
// other positions masked): queries per second for a batch of queries, the latency of a single query
// (as vsearch runs one), and the recall@10 of the masked graph against the exact path. The numbers
// decide where the filtered search switches from one path to the other (filter_exact_limit).
// Then range search (radius with k 16384) on the graph: the growing walk, the walk with ef = k, and
// the flat search, for radii that hold about 10, 100 and 1000 vectors per query.
//
//   bench_filter [--dir=DIR] [--dataset=sift] [--threads=N] [--ef_search=100] [--codes] [--range_only]
//
// With --dir, reads DIR/<dataset>_base.fvecs and _query.fvecs (SIFT1M) and builds the HNSW graph
// (m 16, ef_construction 200). Without, prints a note and exits. --codes adds sq8 codes and ranks by
// them (precision balanced: 2 x k candidates rescored).
#include "../../src/engine/hnsw.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/search.h"
#include "../../src/engine/snapshot.h"
#include "../../src/engine/sq8.h"
#include "bench_util.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace vvector;
using namespace bench;

int main(int argc, char **argv)
{
    std::string dir, dataset = "sift";
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::uint32_t ef = 100;
    bool codes = false, range_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
        else if (a.rfind("--dataset=", 0) == 0) dataset = a.substr(10);
        else if (a.rfind("--threads=", 0) == 0) threads = std::atoi(a.c_str() + 10);
        else if (a.rfind("--ef_search=", 0) == 0) ef = static_cast<std::uint32_t>(std::atoi(a.c_str() + 12));
        else if (a == "--codes") codes = true;
        else if (a == "--range_only") range_only = true;
        else { std::fprintf(stderr, "bench_filter: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (dir.empty()) {
        std::printf("bench_filter: skipped (make bench DATA_DIR=<dir with %s_base.fvecs>)\n", dataset.c_str());
        return 0;
    }
    std::vector<float> base, query;
    std::uint32_t dims = 0, qdims = 0;
    std::uint64_t n = 0, nq = 0;
    if (!read_vecs(dir + "/" + dataset + "_base.fvecs", base, dims, n) || !read_vecs(dir + "/" + dataset + "_query.fvecs", query, qdims, nq)) {
        std::fprintf(stderr, "bench_filter: cannot read the %s files in %s\n", dataset.c_str(), dir.c_str());
        return 1;
    }
    SnapshotBuilder b(Metric::L2);
    for (std::uint64_t i = 0; i < n; ++i) b.add(static_cast<std::int64_t>(i), base.data() + i * dims, dims);
    std::vector<float>().swap(base);
    HnswParams hp;
    hp.threads = threads;
    const GraphSection g = hnsw_graph_section(hp);
    const CodeSection c = sq8_code_section();
    SnapshotBuffer buffer;
    const Clock::time_point t0 = Clock::now();
    b.finish(0, buffer, &g, codes ? &c : nullptr);
    const VectorSet set = snapshot_open(buffer.data(), buffer.size(), false);
    std::printf("bench_filter: %s %llu x %u, HNSW m 16 ef_construction 200%s, build %.1f s, %d threads, ef_search %u\n",
                dataset.c_str(), (unsigned long long)n, dims, codes ? " + sq8" : "", seconds_since(t0), threads, ef);

    const std::uint64_t batch = std::min<std::uint64_t>(nq, 1000), singles = std::min<std::uint64_t>(nq, 200);
    const std::uint32_t stride = set.row_stride;
    std::vector<float> qs(nq * stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) std::memcpy(qs.data() + i * stride, query.data() + i * dims, dims * 4);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = stride;
    s.queries = qs.data();
    s.k = 10;
    s.threads = threads;

    std::printf("%10s %9s | %12s %12s %8s | %12s %12s\n", "filter", "allowed", "exact q/s", "graph q/s", "recall", "exact 1q ms",
                "graph 1q ms");
    std::uint64_t seed = 7;                      // xorshift: the same allow-lists on every machine
    auto next = [&seed] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (double sel : {0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}) {
        if (range_only) break;
        std::vector<std::uint32_t> allow;
        for (std::uint64_t i = 0; i < n; ++i)
            if ((next() >> 11) * 0x1.0p-53 < sel) allow.push_back(static_cast<std::uint32_t>(i));
        SearchPlan exact, graph;
        exact.filtered = graph.filtered = true;
        exact.allow = graph.allow = allow.data();
        exact.n_allow = graph.n_allow = allow.size();
        exact.exact_below = static_cast<std::int64_t>(n) + 1;
        graph.exact_below = 0;
        graph.graph = true;
        graph.ef = ef;
        graph.codes = codes;
        graph.oversampling = 2;
        std::vector<Neighbor> a, bb;
        std::vector<std::uint32_t> ca, cb;
        s.n_queries = batch;
        s.queries = qs.data();
        const double te = median_seconds(3, [&] { index_search(s, set, exact, nullptr, nullptr, a, ca); });
        const double tg = median_seconds(3, [&] { index_search(s, set, graph, nullptr, nullptr, bb, cb); });
        std::uint64_t hit = 0, total = 0;
        for (std::uint64_t q = 0; q < batch; ++q) {
            total += ca[q];
            for (std::uint32_t i = 0; i < cb[q]; ++i)
                for (std::uint32_t r = 0; r < ca[q]; ++r) hit += bb[q * 10 + i].id == a[q * 10 + r].id;
        }
        // Single queries, one after the other, as separate calls.
        s.n_queries = 1;
        auto one_by_one = [&](const SearchPlan &p) {
            const Clock::time_point t = Clock::now();
            for (std::uint64_t q = 0; q < singles; ++q) {
                s.queries = qs.data() + q * stride;
                index_search(s, set, p, nullptr, nullptr, a, ca);
            }
            s.queries = qs.data();
            return seconds_since(t) / double(singles);
        };
        const double se = one_by_one(exact), sg = one_by_one(graph);
        std::printf("%9.2f%% %9zu | %12.1f %12.1f %8.4f | %12.3f %12.3f\n", sel * 100, allow.size(), batch / te, batch / tg,
                    total ? double(hit) / double(total) : 1.0, se * 1000, sg * 1000);
    }

    // Range search: radius with k 16384. The walk that grows ef while the candidates are within the
    // radius (hnsw.h), against the walk with ef = k (the search before milestone M5), and the flat
    // search (exact). The radius: the median over the queries of the distance of their r-th
    // neighbour, so a typical query has about r vectors within it.
    std::printf("%10s %9s | %12s %12s %8s | %12s\n", "range", "within", "grow q/s", "ef=k q/s", "recall", "flat q/s");
    s.n_queries = batch;
    s.queries = qs.data();
    const RowBlock all{set.vectors, set.ids, set.count, nullptr};
    for (std::uint32_t r : {10u, 100u, 1000u}) {
        std::vector<Neighbor> ref, got;
        std::vector<std::uint32_t> rc, gc;
        const std::uint64_t nr = 100;              // k 16384 with ef = k is slow: 100 queries
        FlatSearch some = s;
        some.n_queries = nr;
        some.k = r;
        flat_search(some, &all, 1, ref, rc);
        std::vector<float> kth(nr);
        for (std::uint64_t q = 0; q < nr; ++q) kth[q] = ref[q * r + r - 1].key;
        std::nth_element(kth.begin(), kth.begin() + nr / 2, kth.end());
        s.k = 16384;
        s.has_radius = true;
        s.radius = key_to_score(Metric::L2, kth[nr / 2]);
        SearchPlan grow, fixed, flat;
        grow.graph = fixed.graph = true;
        grow.codes = fixed.codes = codes;
        grow.oversampling = fixed.oversampling = 2;
        grow.ef = ef;
        fixed.ef = 16384;
        s.n_queries = nr;
        const double tf = median_seconds(1, [&] { index_search(s, set, flat, nullptr, nullptr, ref, rc); });
        const double tg = median_seconds(3, [&] { index_search(s, set, grow, nullptr, nullptr, got, gc); });
        std::uint64_t hit = 0, total = 0;
        for (std::uint64_t q = 0; q < nr; ++q) {
            total += rc[q];
            std::vector<std::int64_t> want;
            for (std::uint32_t i = 0; i < rc[q]; ++i) want.push_back(ref[q * s.k + i].id);
            std::sort(want.begin(), want.end());
            for (std::uint32_t i = 0; i < gc[q]; ++i) hit += std::binary_search(want.begin(), want.end(), got[q * s.k + i].id);
        }
        const double tx = median_seconds(1, [&] { index_search(s, set, fixed, nullptr, nullptr, got, gc); });
        std::printf("%10u %9.1f | %12.1f %12.1f %8.4f | %12.1f\n", r, double(total) / double(nr), nr / tg, nr / tx,
                    total ? double(hit) / double(total) : 1.0, nr / tf);
        s.k = 10;
        s.has_radius = false;
    }
    return 0;
}
