// Thread helper: every range runs exactly once, results are independent of the thread count,
// errors and cancellation reach the caller.
#include "check.h"

#include "../../src/engine/parallel.h"

#include <atomic>

using namespace vvector;

int main()
{
    const std::uint64_t n = 100003;
    std::vector<std::uint64_t> expect;
    for (int threads : {1, 2, 7, 64}) {
        // Per-range sums of a float-free function, added in range order: the same for any thread count.
        std::vector<std::uint64_t> per_range((n + 999) / 1000, 0);
        std::vector<std::atomic<int>> seen(n);
        std::atomic<int> max_thread(0);
        parallel_ranges(n, 1000, threads, [&](int t, std::uint64_t r, std::uint64_t b, std::uint64_t e) {
            if (t > max_thread.load()) max_thread.store(t);
            for (std::uint64_t i = b; i < e; ++i) { per_range[r] += i * i % 7919; seen[i]++; }
        });
        bool once = true;
        for (std::uint64_t i = 0; i < n; ++i) once = once && seen[i].load() == 1;
        CHECK(once);
        CHECK(max_thread.load() < threads);
        if (expect.empty()) expect = per_range;
        CHECK(per_range == expect);
    }
    // Empty input: fn is never called.
    int calls = 0;
    parallel_ranges(0, 10, 4, [&](int, std::uint64_t, std::uint64_t, std::uint64_t) { ++calls; });
    CHECK(calls == 0);
    // Threads resolve: 0 = one per core, limits enforced.
    CHECK(resolve_threads(0) >= 1 && resolve_threads(3) == 3);
    CHECK(throws([] { resolve_threads(65); }, "threads must be") && throws([] { resolve_threads(-1); }, "threads must be"));
    // An error in a worker is rethrown in the caller after the join.
    CHECK(throws([] {
        parallel_ranges(1000, 10, 4, [](int, std::uint64_t r, std::uint64_t, std::uint64_t) {
            if (r == 57) throw std::runtime_error("range 57 failed");
        });
    }, "range 57 failed"));
    // Cancellation: poll runs on the calling thread; no new range starts after it says stop.
    {
        std::atomic<std::uint64_t> done(0);
        int polls = 0;
        bool cancelled = false;
        try {
            parallel_ranges(100000, 1, 4, [&](int, std::uint64_t, std::uint64_t, std::uint64_t) { done++; },
                            [&] { return ++polls > 10; });
        } catch (const Cancelled &) {
            cancelled = true;
        }
        CHECK(cancelled && done.load() < 100000);
    }
    return finish("test_parallel");
}
