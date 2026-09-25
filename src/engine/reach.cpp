#include "reach.h"
#include "parallel.h"

#include <algorithm>
#include <atomic>
#include <vector>

namespace vvector {

std::uint64_t hnsw_unreachable(const VectorSet &s, const HnswGraph &g, int threads, const std::function<bool()> &poll)
{
    const std::uint64_t n = g.count;
    if (n == 0) return 0;
    threads = std::max(1, std::min(threads, 64));
    // One bit per position, set with an atomic or: two threads that reach a position in the same
    // round both see it new at worst, and then it is in the next frontier twice, which is harmless
    // (its links are read twice) and does not change the count.
    std::vector<std::atomic<std::uint64_t>> seen((n + 63) / 64);
    for (auto &w : seen) w.store(0, std::memory_order_relaxed);
    auto mark = [&seen](std::uint32_t p) {
        const std::uint64_t bit = 1ull << (p & 63);
        return (seen[p >> 6].fetch_or(bit, std::memory_order_relaxed) & bit) == 0;     // true: new
    };
    std::vector<std::uint32_t> frontier(1, g.entry_point), next;
    mark(g.entry_point);
    std::vector<std::vector<std::uint32_t>> found(static_cast<std::size_t>(threads));
    while (!frontier.empty()) {
        const std::uint64_t per_unit = std::max<std::uint64_t>(256, frontier.size() / (4 * std::uint64_t(threads)) + 1);
        parallel_ranges(frontier.size(), per_unit, threads, [&](int t, std::uint64_t, std::uint64_t a, std::uint64_t b) {
            std::vector<std::uint32_t> &out = found[t];
            for (std::uint64_t i = a; i < b; ++i) {
                const std::uint32_t *l = g.links(frontier[i], 0);
                for (std::uint32_t j = 1; j <= l[0]; ++j)
                    if (mark(l[j])) out.push_back(l[j]);
            }
        }, poll);
        next.clear();
        for (auto &out : found) {
            next.insert(next.end(), out.begin(), out.end());
            out.clear();
        }
        frontier.swap(next);
    }
    std::uint64_t unreachable = 0;
    for (std::uint64_t p = 0; p < n; ++p)
        if (!(seen[p >> 6].load(std::memory_order_relaxed) >> (p & 63) & 1u) && !s.dead(p)) ++unreachable;
    return unreachable;
}

} // namespace vvector
