// vvector engine: HNSW index. STUB.
// The interface is fixed so that vbuild, the snapshot format (FLAG_HNSW, off_graph) and
// vsearch can be wired now. Every call throws until the index is written (milestone M2).
#ifndef VVECTOR_ENGINE_HNSW_H
#define VVECTOR_ENGINE_HNSW_H

#include "snapshot.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vvector {

struct HnswParams {
    std::uint32_t m = 16;                  // links per node and layer (2 * m on layer 0)
    std::uint32_t ef_construction = 200;   // candidate list size while building
};

// Builds the graph section of a snapshot for the vectors in s. Returns the section bytes.
inline std::vector<std::uint8_t> hnsw_build(const VectorSet &, const HnswParams &)
{
    throw std::runtime_error("HNSW is not implemented yet");
}

// Approximate k nearest neighbours of q: (position in s, score), best first.
// ef_search >= k is the candidate list size of the search.
inline std::vector<std::pair<std::uint64_t, float>> hnsw_search(const VectorSet &, const float *, std::uint32_t,
                                                                std::uint32_t)
{
    throw std::runtime_error("HNSW is not implemented yet");
}

} // namespace vvector

#endif
