// The measurements of bench_hnsw, made with hnswlib (github.com/nmslib/hnswlib, Apache 2.0) on
// the same data, the same parameters and the same threads: the reference for the HNSW engine.
// Built only by `make bench HNSWLIB_DIR=<clone of hnswlib>`, with the flags hnswlib's own build
// uses (-Ofast -march=native); hnswlib is not part of this repository and not in the library.
//
//   bench_hnswlib --dir=DIR [--dataset=sift] [--threads=N] [--m=16] [--ef_construction=200] [--build1]
#include "hnswlib/hnswlib.h"
#include "bench_util.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace bench;

// Runs fn(i) for i in [0, n) on `threads` threads, handing out indexes one by one.
template <class F> void parallel_for(std::uint64_t n, int threads, F fn)
{
    std::atomic<std::uint64_t> next(0);
    auto loop = [&] {
        for (std::uint64_t i; (i = next.fetch_add(1)) < n;) fn(i);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(loop);
    loop();
    for (std::thread &t : pool) t.join();
}

int main(int argc, char **argv)
{
    std::string dir, dataset = "sift";
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::size_t m = 16, efc = 200;
    bool build1 = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
        else if (a.rfind("--dataset=", 0) == 0) dataset = a.substr(10);
        else if (a.rfind("--threads=", 0) == 0) threads = std::atoi(a.c_str() + 10);
        else if (a.rfind("--m=", 0) == 0) m = std::strtoul(a.c_str() + 4, nullptr, 10);
        else if (a.rfind("--ef_construction=", 0) == 0) efc = std::strtoul(a.c_str() + 18, nullptr, 10);
        else if (a == "--build1") build1 = true;
        else {
            std::fprintf(stderr, "usage: bench_hnswlib --dir=DIR [--dataset=sift] [--threads=N] [--m=16] [--ef_construction=200] [--build1]\n");
            return 2;
        }
    }
    if (dir.empty()) {
        std::printf("bench_hnswlib: skipped, it needs a data set with ground truth (make bench DATA_DIR=... HNSWLIB_DIR=...)\n");
        return 0;
    }
    Dataset d;
    if (!d.load(dir, dataset)) {
        std::fprintf(stderr, "bench_hnswlib: cannot read the %s files in %s\n", dataset.c_str(), dir.c_str());
        return 1;
    }
    std::printf("hnswlib: %s, %llu vectors of %u dimensions, %llu queries; threads %d\n", dataset.c_str(),
                (unsigned long long)d.n, d.dims, (unsigned long long)d.nq, threads);

    hnswlib::L2Space space(d.dims);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    auto build = [&](int t) {
        index.reset(new hnswlib::HierarchicalNSW<float>(&space, d.n, m, efc, 100));
        const Clock::time_point s = Clock::now();
        index->addPoint(&d.base[0], 0);
        parallel_for(d.n - 1, t, [&](std::uint64_t i) { index->addPoint(&d.base[(i + 1) * d.dims], i + 1); });
        return seconds_since(s);
    };
    char what[120];
    if (build1) {
        std::snprintf(what, sizeof(what), "hnswlib build, m %zu, ef_construction %zu, 1 thread", m, efc);
        line(what, build(1), "s");
    }
    std::snprintf(what, sizeof(what), "hnswlib build, m %zu, ef_construction %zu, %d threads", m, efc, threads);
    line(what, build(threads), "s");

    std::vector<std::int64_t> ids(d.nq * 10, -1);
    auto search = [&](std::uint64_t q) {
        auto r = index->searchKnn(&d.queries[q * d.dims], 10);
        for (int j = static_cast<int>(r.size()) - 1; j >= 0; --j) {
            ids[q * 10 + j] = static_cast<std::int64_t>(r.top().second);
            r.pop();
        }
    };
    std::printf("%-8s %10s %16s %16s %18s\n", "ef", "recall@10", "q/s 1 thread", "q/s all threads", "1 call, ms (p50)");
    const std::uint64_t n1 = std::min<std::uint64_t>(d.nq, 2000);
    for (std::uint32_t ef : ef_list()) {
        index->setEf(std::max<std::size_t>(ef, 10));
        parallel_for(d.nq, threads, search);
        const double recall = d.recall(ids, 10, d.nq);
        const double all = median_seconds(3, [&] { parallel_for(d.nq, threads, search); });
        const double one = median_seconds(3, [&] { for (std::uint64_t q = 0; q < n1; ++q) search(q); });
        std::uint64_t next = 0;
        const double call = median_seconds(501, [&] { search(next++ % d.nq); });
        std::printf("%-8u %10.4f %16.0f %16.0f %18.4f\n", ef, recall, n1 / one, d.nq / all, call * 1000);
    }
    return 0;
}
