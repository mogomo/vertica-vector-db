// vvector engine: HNSW graph index (Malkov and Yashunin, TPAMI 2018; the mechanics follow hnswlib).
// The graph is built in place into the graph section of a snapshot (SnapshotBuilder::finish with
// hnsw_graph_section) and searched straight from the mapped file. Vectors stay in the vectors
// section: the flat search and the graph search read the same rows.
//
// Graph section (docs/format.md), every part on a 64-byte boundary from the section start:
//   header       64 bytes, HnswHeader
//   levels       uint8[count]: the top level of every position
//   level0       count blocks of (m0 + 1) uint32: n, then n neighbour positions; m0 = 2 x m
//   upper_index  uint32[count]: the first upper block of a position, NO_UPPER when its level is 0
//   upper        for every position of level L >= 1, in position order, L blocks of (m + 1) uint32
//                (n, then n neighbours) for the levels 1 .. L
#ifndef VVECTOR_ENGINE_HNSW_H
#define VVECTOR_ENGINE_HNSW_H

#include "flat.h"
#include "snapshot.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace vvector {

constexpr std::uint32_t HNSW_MIN_M = 2;
constexpr std::uint32_t HNSW_MAX_M = 256;
constexpr std::uint32_t HNSW_MAX_LEVEL = 32;
constexpr std::uint32_t HNSW_NO_UPPER = 0xFFFFFFFFu;
constexpr std::uint64_t HNSW_HEADER_BYTES = 64;
constexpr std::uint64_t HNSW_LEVEL_SEED = 0x76766563746f7231ull;

struct HnswHeader {
    std::uint32_t m;               // links per node on the levels above 0
    std::uint32_t m0;              // links per node on level 0: 2 x m
    std::uint32_t ef_construction;
    std::uint32_t max_level;       // level of the entry point
    std::uint32_t entry_point;     // position where every search starts
    std::uint32_t reserved0;
    std::uint64_t count;           // positions, as in the snapshot header
    std::uint64_t level_seed;      // levels are a function of (level_seed, id)
    std::uint64_t upper_blocks;    // blocks in the upper part: the sum of all levels
    std::uint64_t upper_capacity;  // blocks the upper part has room for (snapshot FLAG_CAPACITY); else 0
    std::uint64_t reserved1;
};
static_assert(sizeof(HnswHeader) == HNSW_HEADER_BYTES, "graph header must be 64 bytes");

struct HnswParams {
    std::uint32_t m = 16;                  // links per node and level (2 x m on level 0)
    std::uint32_t ef_construction = 200;   // candidate list size while building
    int threads = 1;                       // build threads
};

// The level of the node with this id: floor(-ln(u) / ln(m)) for u in (0, 1] from a hash of
// (seed, id), at most HNSW_MAX_LEVEL. The same id always gets the same level.
std::uint32_t hnsw_level(std::int64_t id, std::uint32_t m, std::uint64_t seed = HNSW_LEVEL_SEED);

// The graph section of a snapshot, opened for reading.
struct HnswGraph {
    std::uint32_t m = 0, m0 = 0, ef_construction = 0, max_level = 0, entry_point = 0;
    std::uint64_t count = 0, upper_blocks = 0;
    std::uint64_t capacity = 0, upper_capacity = 0;   // what the layout has room for (= count, upper_blocks without FLAG_CAPACITY)
    const std::uint8_t *levels = nullptr;
    const std::uint32_t *level0 = nullptr;
    const std::uint32_t *upper_index = nullptr;
    const std::uint32_t *upper = nullptr;

    // The links of pos on a level: [0] = n, [1 .. n] = neighbour positions.
    const std::uint32_t *links(std::uint32_t pos, std::uint32_t level) const
    {
        return level == 0 ? level0 + std::uint64_t(pos) * (m0 + 1)
                          : upper + (std::uint64_t(upper_index[pos]) + level - 1) * (m + 1);
    }
};

// Blocks the upper part is laid out for: the blocks of the levels plus room for slack more
// positions (2 x their expected levels, at least 64 blocks); blocks when slack is 0.
std::uint64_t hnsw_upper_capacity(std::uint64_t blocks, std::uint64_t slack, std::uint32_t m);

// Bytes of the graph section for these ids (in position order) in a layout with room for capacity
// positions (n = no slack). Throws when m is out of range.
std::uint64_t hnsw_section_bytes(const std::int64_t *ids, std::uint64_t n, std::uint32_t m, std::uint64_t capacity);

// Builds the graph of the vectors of s into section (hnsw_section_bytes bytes, zero-filled).
// Nodes are inserted in parallel as in hnswlib: a lock per group of link lists, a global lock only
// while the entry point changes. The graph depends on how the threads interleave; with one thread it
// depends on the data only. poll: see parallel.h (Cancelled is thrown).
void hnsw_build(const VectorSet &s, std::uint8_t *section, const HnswParams &p,
                const std::function<bool()> &poll = std::function<bool()>());

// The graph section for SnapshotBuilder::finish.
GraphSection hnsw_graph_section(const HnswParams &p, const std::function<bool()> &poll = std::function<bool()>());

// Incremental build (delta.h): bytes of the graph section of base extended by n_new positions with
// these ids, in a layout with room for capacity positions.
std::uint64_t hnsw_extended_bytes(const HnswGraph &base, const std::int64_t *new_ids, std::uint64_t n_new,
                                  std::uint64_t capacity);
// True when the base's layout has room for n_new more positions with these ids: then the extended
// graph fits into the base's section unchanged in size and place (hnsw_extend in place).
bool hnsw_fits_in_place(const HnswGraph &base, const std::int64_t *new_ids, std::uint64_t n_new, std::uint64_t capacity);

// Incremental build: writes the base graph into section (hnsw_extended_bytes bytes, zero-filled) and
// inserts the positions base.count .. s.count - 1 of s with the insertion code of hnsw_build.
// s is the new snapshot: the base's positions first, same order. Tombstoned positions of s stay in
// the graph and are passed through, but new nodes are not linked to them. p.m must be base.m.
// in_place: section already holds the base graph in the base's layout, which is kept
// (hnsw_fits_in_place): nothing is copied, the new positions are appended into the slack.
void hnsw_extend(const HnswGraph &base, const VectorSet &s, std::uint8_t *section, const HnswParams &p,
                 const std::function<bool()> &poll = std::function<bool()>(), bool in_place = false);

// Opens the graph of a snapshot with FLAG_HNSW. Throws std::runtime_error with the cause. verify
// reads every link (vload); without it only the header is checked (every query).
HnswGraph hnsw_open(const VectorSet &s, bool verify);

// Approximate k nearest neighbours of every query of s (s.metric, s.stride, s.queries, s.k,
// s.radius, s.threads as in flat_search). ef (raised to k) is the candidate list size of the
// search. Positions marked in skip (journal mask, tombstones; may be null) are traversed but never
// returned. extra (may be null) holds rows searched exactly beside the graph (the journal's live
// vectors); both result lists are merged. Result layout and order as flat_search: (key, id).
// With radius only the found neighbours within it are returned; when ef is below k (range search),
// the walk starts with ef and grows it fourfold, up to k, while more than a quarter of the candidates found
// are within the radius, so a large k costs only as much as the vectors within the radius. One thread per query; the result
// does not depend on the number of threads.
void hnsw_search(const FlatSearch &s, const VectorSet &set, const HnswGraph &g, std::uint32_t ef,
                 const std::uint64_t *skip, const RowBlock *extra, std::vector<Neighbor> &out,
                 std::vector<std::uint32_t> &count, const std::function<bool()> &poll = std::function<bool()>());

// The same walk on the sq8 codes (sq8.h) of set and of the queries (s.query_codes, s.query_sums):
// the s.k nearest allowed positions of every query by their sq8 keys, in flat_search's layout, with
// the position in place of the id (ties by position). No journal rows, and the radius only steers
// the range walk of hnsw_search (on the approximate keys); the result is not cut by it: the caller
// rescores the result and applies the radius (search.h).
void hnsw_search_codes(const FlatSearch &s, const VectorSet &set, const Sq8Codes &codes, const HnswGraph &g,
                       std::uint32_t ef, const std::uint64_t *skip, std::vector<Neighbor> &out,
                       std::vector<std::uint32_t> &count, const std::function<bool()> &poll = std::function<bool()>());

} // namespace vvector

#endif
