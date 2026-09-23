// Shared by the engine benchmarks: timing, output lines, and the TEXMEX vector files (SIFT1M).
// Header only, no engine includes: bench_hnswlib.cpp uses it without the vvector engine.
#ifndef VVECTOR_TESTS_BENCH_UTIL_H
#define VVECTOR_TESTS_BENCH_UTIL_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

template <class F> double median_seconds(int runs, F f)
{
    std::vector<double> t;
    for (int i = 0; i < runs; ++i) {
        const Clock::time_point s = Clock::now();
        f();
        t.push_back(seconds_since(s));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// One result line: what, value, unit.
inline void line(const char *what, double value, const char *unit) { std::printf("%-58s %12.3f %s\n", what, value, unit); }

// Reads a .fvecs or .ivecs file: count vectors of dims values (as float or int).
template <class T>
bool read_vecs(const std::string &path, std::vector<T> &out, std::uint32_t &dims, std::uint64_t &count)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    count = 0;
    std::int32_t d;
    while (std::fread(&d, 4, 1, f) == 1) {
        dims = static_cast<std::uint32_t>(d);
        const std::size_t at = out.size();
        out.resize(at + d);
        if (std::fread(out.data() + at, 4, d, f) != static_cast<std::size_t>(d)) { std::fclose(f); return false; }
        ++count;
    }
    std::fclose(f);
    return true;
}

// A data set with ground truth: DIR/NAME_base.fvecs, _query.fvecs, _groundtruth.ivecs.
struct Dataset {
    std::vector<float> base, queries;
    std::vector<std::int32_t> truth;
    std::uint32_t dims = 0, truth_dims = 0;
    std::uint64_t n = 0, nq = 0;

    bool load(const std::string &dir, const std::string &name)
    {
        std::uint32_t qd = 0;
        std::uint64_t nt = 0;
        return read_vecs(dir + "/" + name + "_base.fvecs", base, dims, n) &&
               read_vecs(dir + "/" + name + "_query.fvecs", queries, qd, nq) && qd == dims &&
               read_vecs(dir + "/" + name + "_groundtruth.ivecs", truth, truth_dims, nt) && nt == nq;
    }

    // Recall@k of found[q * k + i] (ids, -1 = none) against the first k true neighbours.
    double recall(const std::vector<std::int64_t> &found, std::uint32_t k, std::uint64_t queries) const
    {
        std::uint64_t hit = 0;
        for (std::uint64_t q = 0; q < queries; ++q)
            for (std::uint32_t i = 0; i < k; ++i)
                for (std::uint32_t r = 0; r < k; ++r)
                    if (found[q * k + i] == truth[q * truth_dims + r]) { ++hit; break; }
        return hit / (double(k) * queries);
    }
};

// The ef_search values every HNSW benchmark reports.
inline std::vector<std::uint32_t> ef_list() { return {10, 16, 32, 64, 100, 200, 400}; }

} // namespace bench

#endif
