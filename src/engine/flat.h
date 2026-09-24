// vvector engine: exact (flat) k-nearest-neighbour search.
//
// Searches one or more blocks of rows (the snapshot, and the journal's live vectors beside it)
// for the k closest rows to every query. Candidates are ordered by (key, id): smaller key first,
// ties by smaller id. The ids of all blocks together must be unique among the rows that are not
// skipped; then the result depends on nothing but the data: not on the number of threads, the
// batch tiling, the CPU or the order of the queries.
#ifndef VVECTOR_ENGINE_FLAT_H
#define VVECTOR_ENGINE_FLAT_H

#include "snapshot.h"
#include "sq8.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace vvector {

// Rows to search: n rows of `stride` floats, their ids, and an optional bitset of rows that are no
// candidates (bit i % 64 of word i / 64 set: a masked or tombstoned row). ids null: the id of a row
// is its row number (the candidate search of search.cpp works with positions). With codes, the rows are
// ranked by their sq8 codes (row i of codes is row i of the block) against the query codes of the
// search, and the keys are sq8 keys (sq8.h); the float rows are then not read.
struct RowBlock {
    const float *rows = nullptr;
    const std::int64_t *ids = nullptr;
    std::uint64_t n = 0;
    const std::uint64_t *skip = nullptr;
    const Sq8Codes *codes = nullptr;
};

struct Neighbor {
    float key;          // see kernels.h: smaller is closer
    std::int64_t id;
};

inline bool closer(const Neighbor &a, const Neighbor &b) { return a.key < b.key || (a.key == b.key && a.id < b.id); }

struct FlatSearch {
    Metric metric = Metric::L2;
    std::uint32_t stride = 0;
    const float *queries = nullptr;      // n_queries x stride, zero padded; unit length for cosine
    std::uint64_t n_queries = 0;
    std::uint32_t k = 10;
    bool has_radius = false;             // only candidates whose score is within radius:
    double radius = 0;                   //   l2, l1: score <= radius; cosine, dot: score >= radius
    int threads = 1;
    const std::uint8_t *query_codes = nullptr;   // blocks with codes: n_queries rows of code_stride bytes
    const std::uint32_t *query_sums = nullptr;   //   and the code sum of each query
};

// The k closest rows of every query: neighbours of query q are out[q * k] to out[q * k + count[q] - 1],
// closest first. count[q] < k when there are fewer candidates, or fewer within the radius.
// A candidate whose key is not a number (only possible with elements near the float limit) is never
// returned. poll runs on the calling thread between work units; when it returns true the search
// stops and throws Cancelled (parallel.h).
void flat_search(const FlatSearch &s, const RowBlock *blocks, std::size_t n_blocks, std::vector<Neighbor> &out,
                 std::vector<std::uint32_t> &count, const std::function<bool()> &poll = std::function<bool()>());

// True if a key passes the radius of s.
bool within_radius(const FlatSearch &s, float key);

// Searches block extra exactly (flat_search with s) and merges its results into out and count, the
// results of the same queries of s in flat_search's layout: the k closest of both, in (key, id) order.
void merge_block(const FlatSearch &s, const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                 const std::function<bool()> &poll = std::function<bool()>());

} // namespace vvector

#endif
