// Benchmark of the HNSW engine without Vertica: graph build, then for each ef_search the recall@10
// against the ground truth, queries per second on one thread and on all threads, and the latency
// of one search call as vsearch makes it; then the same with sq8 codes (milestone M4): the walk on
// the codes with 2 x k candidates rescored (precision balanced), and without rescoring.
//
//   bench_hnsw [--dir=DIR] [--dataset=sift] [--threads=N] [--m=16] [--ef_construction=200] [--build1]
//
// With --dir, reads DIR/<dataset>_base.fvecs, _query.fvecs and _groundtruth.ivecs (SIFT1M).
// Without, prints a note and exits: recall on random data means nothing. --build1 also times a
// build on one thread. The same measurements for hnswlib: bench_hnswlib (make bench HNSWLIB_DIR=...).
#include "../../src/engine/hnsw.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"
#include "../../src/engine/search.h"
#include "../../src/engine/snapshot.h"
#include "../../src/engine/sq8.h"
#include "../../src/engine/version.h"
#include "bench_util.h"

#include <cstdlib>
#include <cstring>
#include <thread>

using namespace vvector;
using namespace bench;

int main(int argc, char **argv)
{
    std::string dir, dataset = "sift";
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    HnswParams p;
    bool build1 = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
        else if (a.rfind("--dataset=", 0) == 0) dataset = a.substr(10);
        else if (a.rfind("--threads=", 0) == 0) threads = std::atoi(a.c_str() + 10);
        else if (a.rfind("--m=", 0) == 0) p.m = static_cast<std::uint32_t>(std::atoi(a.c_str() + 4));
        else if (a.rfind("--ef_construction=", 0) == 0) p.ef_construction = static_cast<std::uint32_t>(std::atoi(a.c_str() + 18));
        else if (a == "--build1") build1 = true;
        else {
            std::fprintf(stderr, "usage: bench_hnsw [--dir=DIR] [--dataset=sift] [--threads=N] [--m=16] [--ef_construction=200] [--build1]\n");
            return 2;
        }
    }
    if (dir.empty()) {
        std::printf("bench_hnsw: skipped, it needs a data set with ground truth (make bench DATA_DIR=<dir of sift_*.fvecs>)\n");
        return 0;
    }
    Dataset d;
    if (!d.load(dir, dataset)) {
        std::fprintf(stderr, "bench_hnsw: cannot read the %s files in %s\n", dataset.c_str(), dir.c_str());
        return 1;
    }
    std::printf("data: %s, %llu vectors of %u dimensions, %llu queries; threads %d, kernels %s, flags %s\n", dataset.c_str(),
                (unsigned long long)d.n, d.dims, (unsigned long long)d.nq, threads, kernel_target(), BUILD_FLAGS);

    auto build = [&](int t, SnapshotBuffer &out) {
        SnapshotBuilder b(Metric::L2);
        for (std::uint64_t i = 0; i < d.n; ++i) b.add(static_cast<std::int64_t>(i), &d.base[i * d.dims], d.dims);
        HnswParams hp = p;
        hp.threads = t;
        const GraphSection g = hnsw_graph_section(hp);
        const CodeSection c = sq8_code_section();
        const Clock::time_point s = Clock::now();
        b.finish(0, out, &g, &c);
        return seconds_since(s);
    };
    SnapshotBuffer buf;
    char what[120];
    if (build1) {
        const double s1 = build(1, buf);
        std::snprintf(what, sizeof(what), "HNSW build, m %u, ef_construction %u, 1 thread", p.m, p.ef_construction);
        line(what, s1, "s");
    }
    const double sn = build(threads, buf);
    std::snprintf(what, sizeof(what), "HNSW build, m %u, ef_construction %u, %d threads", p.m, p.ef_construction, threads);
    line(what, sn, "s");
    const VectorSet set = snapshot_open(buf.data(), buf.size(), false);
    const HnswGraph g = hnsw_open(set, true);
    const Sq8Codes codes = sq8_open(set, true);
    // The sq8 part of the build (training and coding), timed alone: the build above includes it.
    SnapshotBuffer scratch;
    scratch.allocate(set.sq8_bytes);
    const double sq8_s = median_seconds(3, [&] { sq8_fill(set, sq8_train(set), scratch.data()); });
    line("  of it sq8 codes (training and coding, one thread)", sq8_s, "s");
    line("graph section", set.graph_bytes / 1048576.0, "MB");
    line("sq8 section (codes, sums)", set.sq8_bytes / 1048576.0, "MB");
    line("snapshot size", buf.size() / 1048576.0, "MB");

    std::vector<float> q(d.nq * set.row_stride, 0.0f);
    for (std::uint64_t i = 0; i < d.nq; ++i) std::memcpy(&q[i * set.row_stride], &d.queries[i * d.dims], d.dims * 4);
    FlatSearch s;
    s.metric = Metric::L2;
    s.stride = set.row_stride;
    s.k = 10;
    std::vector<Neighbor> out;
    std::vector<std::uint32_t> count;
    auto run = [&](std::uint64_t first, std::uint64_t n, int t, std::uint32_t ef) {
        s.queries = &q[first * set.row_stride];
        s.n_queries = n;
        s.threads = t;
        hnsw_search(s, set, g, ef, nullptr, nullptr, out, count);
    };

    std::printf("%-8s %10s %16s %16s %18s\n", "ef", "recall@10", "q/s 1 thread", "q/s all threads", "1 call, ms (p50)");
    const std::uint64_t n1 = std::min<std::uint64_t>(d.nq, 2000);
    for (std::uint32_t ef : ef_list()) {
        run(0, d.nq, threads, ef);
        std::vector<std::int64_t> ids(d.nq * 10, -1);
        for (std::uint64_t i = 0; i < d.nq; ++i)
            for (std::uint32_t j = 0; j < count[i]; ++j) ids[i * 10 + j] = out[i * 10 + j].id;
        const double recall = d.recall(ids, 10, d.nq);
        const double all = median_seconds(3, [&] { run(0, d.nq, threads, ef); });
        const double one = median_seconds(3, [&] { run(0, n1, 1, ef); });
        std::uint64_t next = 0;
        const double call = median_seconds(501, [&] { run(next++ % d.nq, 1, 1, ef); });
        std::printf("%-8u %10.4f %16.0f %16.0f %18.4f\n", ef, recall, n1 / one, d.nq / all, call * 1000);
    }

    // sq8: the walk on the codes, then 2 x k candidates rescored (precision balanced), or none.
    for (bool rescore : {true, false}) {
        std::printf("sq8 codes (range %.3f .. %.3f), %s\n", codes.range.offset, codes.range.offset + 255.0 * codes.range.scale,
                    rescore ? "2 x k candidates rescored from the float rows" : "without rescoring");
        std::printf("%-8s %10s %16s %16s %18s\n", "ef", "recall@10", "q/s 1 thread", "q/s all threads", "1 call, ms (p50)");
        auto runq = [&](std::uint64_t first, std::uint64_t n, int t, std::uint32_t ef) {
            s.queries = &q[first * set.row_stride];
            s.n_queries = n;
            s.threads = t;
            SearchPlan plan;
            plan.graph = true;
            plan.ef = ef;
            plan.codes = true;
            plan.rescore = rescore;
            plan.oversampling = 2;
            index_search(s, set, plan, nullptr, nullptr, out, count);
        };
        for (std::uint32_t ef : ef_list()) {
            runq(0, d.nq, threads, ef);
            std::vector<std::int64_t> ids(d.nq * 10, -1);
            for (std::uint64_t i = 0; i < d.nq; ++i)
                for (std::uint32_t j = 0; j < count[i]; ++j) ids[i * 10 + j] = out[i * 10 + j].id;
            const double recall = d.recall(ids, 10, d.nq);
            const double all = median_seconds(3, [&] { runq(0, d.nq, threads, ef); });
            const double one = median_seconds(3, [&] { runq(0, n1, 1, ef); });
            std::uint64_t next = 0;
            const double call = median_seconds(501, [&] { runq(next++ % d.nq, 1, 1, ef); });
            std::printf("%-8u %10.4f %16.0f %16.0f %18.4f\n", ef, recall, n1 / one, d.nq / all, call * 1000);
        }
    }
    return 0;
}
