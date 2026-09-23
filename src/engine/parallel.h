// vvector engine: run a loop over ranges on several threads.
// The threads only touch engine data and are joined before the call returns. They never call
// the Vertica SDK: only the calling thread runs `poll` (for isCanceled()).
#ifndef VVECTOR_ENGINE_PARALLEL_H
#define VVECTOR_ENGINE_PARALLEL_H

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vvector {

// Thrown by parallel_ranges when poll asked to stop. The caller returns quietly.
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

// Threads to use for a `threads` parameter: 0 = one per core; at most 64.
inline int resolve_threads(std::int64_t threads)
{
    if (threads < 0 || threads > 64) throw std::runtime_error("threads must be 0 (one per core) to 64");
    if (threads > 0) return static_cast<int>(threads);
    const unsigned cores = std::thread::hardware_concurrency();
    return cores == 0 ? 1 : static_cast<int>(cores > 64 ? 64 : cores);
}

// Calls fn(thread, range, begin, end) for every range of `size` elements of [0, n). thread is
// 0 .. threads-1 (for per-thread buffers); thread 0 is the calling thread. Ranges are handed out one
// by one in rising order, so uneven ranges balance. Range borders do not depend on the number of
// threads. Before each range the calling thread runs poll() if given; when it returns true, no new
// range starts anywhere and Cancelled is thrown after the join. An exception in a thread is
// rethrown here.
template <class F>
void parallel_ranges(std::uint64_t n, std::uint64_t size, int threads, F &&fn,
                     const std::function<bool()> &poll = std::function<bool()>())
{
    if (size == 0) size = 1;
    const std::uint64_t ranges = (n + size - 1) / size;
    if (static_cast<std::uint64_t>(threads) > ranges) threads = static_cast<int>(ranges);
    std::atomic<std::uint64_t> next(0);
    std::atomic<bool> cancelled(false);
    std::exception_ptr error;
    std::mutex error_lock;
    auto loop = [&](int thread) {
        try {
            for (;;) {
                if (thread == 0 && poll && poll()) { cancelled.store(true); next.store(ranges); }
                const std::uint64_t r = next.fetch_add(1);
                if (r >= ranges) break;
                const std::uint64_t begin = r * size;
                fn(thread, r, begin, begin + size < n ? begin + size : n);
            }
        } catch (...) {
            std::lock_guard<std::mutex> hold(error_lock);
            if (!error) error = std::current_exception();
            next.store(ranges);
        }
    };
    std::vector<std::thread> pool;
    try {
        for (int t = 1; t < threads; ++t) pool.emplace_back(loop, t);
    } catch (...) {
        // Could not start another thread: the ones that run take all ranges.
    }
    loop(0);
    for (std::thread &t : pool) t.join();
    if (error) std::rethrow_exception(error);
    if (cancelled.load()) throw Cancelled();
}

} // namespace vvector

#endif
