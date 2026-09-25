// vvector engine: how many live positions of an HNSW graph no search can reach (milestone M7 E).
// A search walks level 0 from the entry point along links: a live position that no chain of links
// from the entry point leads to is never returned. The neighbour heuristic can prune the last
// link to a node (hnsw_build repairs what it finds); this count is the check, stored in the graph
// header and reported by vinfo and refresh_index.
#ifndef VVECTOR_ENGINE_REACH_H
#define VVECTOR_ENGINE_REACH_H

#include "hnsw.h"

#include <cstdint>
#include <functional>

namespace vvector {

// Live positions of s that a breadth-first walk over level 0 from the entry point of g does not
// visit. Tombstoned positions are walked through (a search passes them too) but never counted.
// The walk runs on `threads` threads, one frontier per round; the count does not depend on the
// number of threads. Memory: one bit per position plus the frontier. poll as parallel.h.
std::uint64_t hnsw_unreachable(const VectorSet &s, const HnswGraph &g, int threads,
                               const std::function<bool()> &poll = std::function<bool()>());

} // namespace vvector

#endif
