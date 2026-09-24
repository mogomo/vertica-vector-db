// vvector engine: one search of a snapshot and of the journal's rows beside it, as vsearch and vknn
// run it: the HNSW graph or every row, ranked by the float rows or by the sq8 codes (sq8.h).
//
// With codes the search first finds candidates by their codes: the k x oversampling best per query
// (at least k). With rescore, their exact keys are computed from the float rows and the k best of
// those are the result: the scores are exact, only the choice of candidates is approximate. Without
// rescore, the k best candidates are returned with their approximate scores. The radius is applied
// to the scores that are returned. The journal's rows are always searched exactly and merged.
//
// Filtered search (PLAN 7.5, Qdrant's cardinality rule): only the allowed positions can be results.
// Below filter_exact_limit live allowed positions, those rows are searched exactly (copied, then the
// flat kernels on the float rows): the graph walk would visit about ef / selectivity nodes to find
// them. Otherwise the search runs as without a filter with the positions outside the list added to
// the skip mask: the graph walk passes through them but never returns them.
#ifndef VVECTOR_ENGINE_SEARCH_H
#define VVECTOR_ENGINE_SEARCH_H

#include "flat.h"
#include "snapshot.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace vvector {

struct SearchPlan {
    bool graph = false;          // walk the HNSW graph (the snapshot must have one), else every row
    std::uint32_t ef = 100;      // candidate list of the graph walk (raised to the candidates)
    bool codes = false;          // rank by the sq8 codes (the snapshot must have them)
    bool rescore = true;         // codes: exact scores for the best candidates, from the float rows
    double oversampling = 1.0;   // codes with rescore: k x oversampling candidates, at least 1
    bool filtered = false;       // only positions in allow can be results (none when n_allow is 0)
    const std::uint32_t *allow = nullptr;   // filtered: snapshot positions, below the count, any order
    std::uint64_t n_allow = 0;
    std::int64_t exact_below = -1;          // filtered: brute force below this many live allowed
                                            //   positions; -1: filter_exact_limit(ef, count)
};

// Filtered search: below this many allowed positions the rows are searched exactly:
// max(10000, sqrt(64 x ef x count)). The exact path costs the allowed rows; the graph walk about
// ef x count / allowed nodes; they meet near sqrt(c x ef x count). c = 64 from SIFT1M (1M x 128, ef
// 100: the paths meet at about 68,000 allowed rows for one query and 120,000 for a batch of 1000;
// tests/engine/bench_filter.cpp, docs/design.md "Filtered search").
std::uint64_t filter_exact_limit(std::uint32_t ef, std::uint64_t count);

// The k closest rows of every query of s (s.queries, s.k, s.radius, s.threads) in the snapshot set
// (positions in skip, may be null, are never returned) and in the extra rows (the journal's live
// vectors, may be null; a filter does not apply to them: the caller passes only allowed rows), as
// the plan says. Result layout and order as flat_search: (key, id). Throws std::runtime_error when an
// allowed position is not below the count.
void index_search(const FlatSearch &s, const VectorSet &set, const SearchPlan &p, const std::uint64_t *skip,
                  const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                  const std::function<bool()> &poll = std::function<bool()>());

// The number of candidates a search with codes takes per query.
std::uint32_t code_candidates(std::uint32_t k, const SearchPlan &p);

} // namespace vvector

#endif
