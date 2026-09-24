// vvector engine: one search of a snapshot and of the journal's rows beside it, as vsearch and vknn
// run it: the HNSW graph or every row, ranked by the float rows or by the sq8 codes (sq8.h).
//
// With codes the search first finds candidates by their codes: the k x oversampling best per query
// (at least k). With rescore, their exact keys are computed from the float rows and the k best of
// those are the result: the scores are exact, only the choice of candidates is approximate. Without
// rescore, the k best candidates are returned with their approximate scores. The radius is applied
// to the scores that are returned. The journal's rows are always searched exactly and merged.
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
};

// The k closest rows of every query of s (s.queries, s.k, s.radius, s.threads) in the snapshot set
// (positions in skip, may be null, are never returned) and in the extra rows (the journal's live
// vectors, may be null), as the plan says. Result layout and order as flat_search: (key, id).
void index_search(const FlatSearch &s, const VectorSet &set, const SearchPlan &p, const std::uint64_t *skip,
                  const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                  const std::function<bool()> &poll = std::function<bool()>());

// The number of candidates a search with codes takes per query.
std::uint32_t code_candidates(std::uint32_t k, const SearchPlan &p);

} // namespace vvector

#endif
