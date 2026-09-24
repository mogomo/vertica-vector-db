// Benchmark of the engine without Vertica: memory bandwidth of the machine, flat search of one
// query and of batches, snapshot build, recall against a ground truth.
//
//   bench_flat [--dir=DIR] [--dataset=sift] [--threads=N] [--queries=N]
//
// With --dir, reads DIR/<dataset>_base.fvecs, _query.fvecs and _groundtruth.ivecs (SIFT1M).
// Without, generates 1,000,000 random vectors of 128 dimensions and prints no recall.
// Prints one line per measurement: what, value, unit. Timings are medians of repeated runs.
#include "../../src/engine/flat.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/snapshot.h"
#include "../../src/engine/version.h"
#include "check.h"
#include "bench_util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace vvector;
using namespace bench;

namespace {

// Memory bandwidth: every thread sums its part of a large buffer (reads only, as a flat scan),
// with 16 independent add chains, so the loop waits for memory, not for the adder.
double read_bandwidth(int threads, const std::vector<float> &buf)
{
    const std::uint64_t n = buf.size();
    std::vector<double> sums(threads);
    const double s = median_seconds(5, [&] {
        parallel_ranges(n, (n + threads - 1) / threads, threads, [&](int t, std::uint64_t, std::uint64_t b, std::uint64_t e) {
            typedef float f4v __attribute__((vector_size(16)));
            f4v acc[16] = {};
            std::uint64_t i = b;
            for (; i + 64 <= e; i += 64)
                for (int j = 0; j < 16; ++j) { f4v x; std::memcpy(&x, &buf[i + 4 * j], 16); acc[j] += x; }
            float r = 0;
            for (int j = 0; j < 16; ++j) r += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
            sums[t] += r;
        });
    });
    return n * 4.0 / s / 1e9;
}

} // namespace

int main(int argc, char **argv)
{
    std::string dir, dataset = "sift";
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::uint64_t n_queries = 1000;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
        else if (a.rfind("--dataset=", 0) == 0) dataset = a.substr(10);
        else if (a.rfind("--threads=", 0) == 0) threads = std::atoi(a.c_str() + 10);
        else if (a.rfind("--queries=", 0) == 0) n_queries = std::strtoull(a.c_str() + 10, nullptr, 10);
        else { std::fprintf(stderr, "usage: bench_flat [--dir=DIR] [--dataset=sift] [--threads=N] [--queries=N]\n"); return 2; }
    }

    std::vector<float> base, queries;
    std::vector<std::int32_t> gt;
    std::uint32_t dims = 128, qdims = 0, gdims = 0;
    std::uint64_t n = 1000000, nq = 10000, ngt = 0;
    bool have_gt = false;
    if (!dir.empty()) {
        if (!bench::read_vecs(dir + "/" + dataset + "_base.fvecs", base, dims, n) ||
            !bench::read_vecs(dir + "/" + dataset + "_query.fvecs", queries, qdims, nq) || qdims != dims) {
            std::fprintf(stderr, "bench_flat: cannot read the %s files in %s\n", dataset.c_str(), dir.c_str());
            return 1;
        }
        have_gt = bench::read_vecs(dir + "/" + dataset + "_groundtruth.ivecs", gt, gdims, ngt) && ngt == nq;
        std::printf("data: %s, %llu vectors of %u dimensions, %llu queries\n", dataset.c_str(), (unsigned long long)n, dims,
                    (unsigned long long)nq);
    } else {
        base.resize(n * dims);
        queries.resize(nq * dims);
        for (std::uint64_t i = 0; i < n; ++i) test_vector(i, dims, 1, true, &base[i * dims]);
        for (std::uint64_t i = 0; i < nq; ++i) test_vector(i, dims, 2, true, &queries[i * dims]);
        std::printf("data: random, %llu vectors of %u dimensions (no ground truth)\n", (unsigned long long)n, dims);
    }
    n_queries = std::min(n_queries, nq);
    std::printf("threads: %d, kernels: %s, flags: %s\n", threads, kernel_target(), BUILD_FLAGS);

    // Build.
    SnapshotBuffer buf;
    const double build_s = median_seconds(3, [&] {
        SnapshotBuilder b(Metric::L2);
        for (std::uint64_t i = 0; i < n; ++i) b.add(static_cast<std::int64_t>(i), &base[i * dims], dims);
        b.finish(0, buf);
    });
    const VectorSet set = snapshot_open(buf.data(), buf.size(), false);
    const double vec_bytes = double(set.count) * set.row_stride * 4;
    line("build: snapshot from float vectors (one thread)", build_s, "s");
    line("snapshot size", buf.size() / 1048576.0, "MB");

    // Padded queries.
    std::vector<float> q(nq * set.row_stride, 0.0f);
    for (std::uint64_t i = 0; i < nq; ++i) std::memcpy(&q[i * set.row_stride], &queries[i * dims], dims * 4);

    const double bw1 = read_bandwidth(1, base), bwn = read_bandwidth(threads, base);
    line("memory read bandwidth, 1 thread", bw1, "GB/s");
    line("memory read bandwidth, all threads", bwn, "GB/s");

    RowBlock block{set.vectors, set.ids, set.count, nullptr};
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = set.row_stride;
    s.k = 10;
    std::vector<Neighbor> out;
    std::vector<std::uint32_t> count;
    auto run = [&](std::uint64_t first, std::uint64_t nq_run, int t) {
        s.queries = &q[first * set.row_stride];
        s.n_queries = nq_run;
        s.threads = t;
        flat_search(s, &block, 1, out, count);
    };

    for (int t : {1, threads}) {
        std::uint64_t next = 0;
        const double one = median_seconds(21, [&] { run(next++ % nq, 1, t); });
        char what[100];
        std::snprintf(what, sizeof(what), "flat, 1 query, k 10, %d thread%s: latency", t, t == 1 ? "" : "s");
        line(what, one * 1000, "ms");
        std::snprintf(what, sizeof(what), "flat, 1 query, %d thread%s: vectors scanned per second", t, t == 1 ? "" : "s");
        line(what, vec_bytes / one / 1e9, "GB/s");
        std::snprintf(what, sizeof(what), "  as a share of the read bandwidth with %d thread%s", t, t == 1 ? "" : "s");
        line(what, 100.0 * (vec_bytes / one / 1e9) / (t == 1 ? bw1 : bwn), "%");
        const double b16 = median_seconds(5, [&] { run(16, 16, t); });
        std::snprintf(what, sizeof(what), "flat, batch of 16 queries, %d thread%s", t, t == 1 ? "" : "s");
        line(what, 16 / b16, "queries/s");
        std::snprintf(what, sizeof(what), "  batch of 16 against one query at a time");
        line(what, (16 / b16) / (1 / one), "x");
    }
    const double big = median_seconds(3, [&] { run(0, n_queries, threads); });
    char what[100];
    std::snprintf(what, sizeof(what), "flat, batch of %llu queries, %d threads", (unsigned long long)n_queries, threads);
    line(what, n_queries / big, "queries/s");

    if (have_gt) {
        // Recall@10 of the exact search against the published ground truth (ties can differ).
        run(0, n_queries, threads);
        std::uint64_t hit = 0;
        for (std::uint64_t i = 0; i < n_queries; ++i)
            for (std::uint32_t a = 0; a < count[i]; ++a)
                for (std::uint32_t b = 0; b < 10; ++b)
                    if (out[i * 10 + a].id == gt[i * gdims + b]) { ++hit; break; }
        line("recall@10 of the flat search against the ground truth", hit / (10.0 * n_queries), "");
    }
    return 0;
}
