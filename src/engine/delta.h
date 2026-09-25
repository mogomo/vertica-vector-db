// vvector engine: incremental build. A new snapshot made from a base snapshot and the changes of
// the journal since the base was built (consolidated: one add or delete per id).
//
// The base's rows, ids, tombstones and graph are kept. A deleted or changed id's position becomes
// a tombstone; a new or changed vector is appended as a new position and, on an HNSW index,
// inserted into the graph with the insertion code of the full build (hnsw_extend). Appended ids
// are not in id order, so the new snapshot gets an id_index (FLAG_ID_INDEX) unless every appended
// id is larger than every id before. An id can then be at several positions: the highest is the
// only one that can be live (VectorSet::find).
//
// Two ways to make the new snapshot (milestone M7):
// - in place (finish_in_place): on a copy-on-write mapping of the base's file, when the base's
//   layout has room for the appended positions (FLAG_CAPACITY). Every section stays where it is,
//   only the changed bytes are touched, and snapshot_diff finds them: a refresh then sends those
//   bytes to the nodes, which assemble the new file from their copy of the base;
// - by copying (finish): into a new buffer, when the base has no room (or no FLAG_CAPACITY). The
//   base's layout is kept when it fits, so both ways give the same bytes.
//
// An add whose vector is bit for bit the live vector of its id changes nothing: the delta margin
// brings rows back that the base already holds. A delete of an id that is not live changes nothing.
#ifndef VVECTOR_ENGINE_DELTA_H
#define VVECTOR_ENGINE_DELTA_H

#include "hnsw.h"
#include "snapshot.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace vvector {

// What an incremental build did.
struct DeltaStats {
    std::uint64_t appended = 0;        // new positions: new ids and changed vectors
    std::uint64_t tombstoned = 0;      // positions that died: changed or deleted ids
    std::uint64_t unchanged = 0;       // adds equal to the live vector: nothing to do
    std::uint64_t absent = 0;          // deletes of ids that are not live: nothing to do
};

// A byte range of a snapshot file.
struct ByteRange {
    std::uint64_t offset = 0, bytes = 0;
};

// What finish_in_place did.
enum class InPlace { Built, Unchanged, NoRoom };

class IncrementalBuilder {
public:
    // base must stay readable until finish returns.
    explicit IncrementalBuilder(const VectorSet &base);

    // Starts the add (or change) of id and returns its row: row_stride floats, zero. Write dims
    // floats, then call end_add. dims must be the base's.
    float *begin_add(std::int64_t id, std::uint32_t dims);
    // Finishes the row: normalises it for a cosine index.
    void end_add();
    void remove(std::int64_t id);

    // Rows given so far (adds and deletes).
    std::uint64_t rows() const { return adds_.size() + removes_.size(); }
    // Room to grow of a snapshot laid out anew (SnapshotBuilder::set_growth); 0 = none.
    void set_growth(std::uint32_t percent);

    // Applies the changes and writes the new snapshot to out, with base_snapshot as its base. The
    // base's layout is kept when it has room (then the bytes equal those of finish_in_place), else
    // the snapshot is laid out anew with room to grow.
    // graph: the graph parameters for an HNSW base (p.m must be the base's m), null for a flat one.
    // Returns false, and leaves out alone, when the changes change nothing. Throws
    // std::runtime_error on an id given twice, on more than 4,294,967,295 positions, on a graph
    // parameter that does not fit the base, and Cancelled (parallel.h) when poll says so.
    bool finish(std::int64_t max_ver, std::int64_t base_snapshot, SnapshotBuffer &out, const HnswParams *graph,
                const std::function<bool()> &poll = std::function<bool()>());

    // Applies the changes in place. copy: a writable copy-on-write mapping (MAP_PRIVATE) of the
    // base's file, size bytes (MappedSnapshot::open with writable_copy); the base this builder was
    // made with must be a read-only mapping of the same file. Afterwards copy holds the new
    // snapshot in the base's layout, unsealed: the caller finds the changed bytes with snapshot_diff
    // over `candidates` (the ranges that may differ; everything else is untouched) and writes the
    // checksum with seal_in_place. NoRoom: the base has no FLAG_CAPACITY, or not enough room for
    // the appended positions: use finish. Unchanged: nothing to do, copy untouched. Errors as finish.
    InPlace finish_in_place(std::int64_t max_ver, std::int64_t base_snapshot, std::uint8_t *copy, std::uint64_t size,
                            const HnswParams *graph, std::vector<ByteRange> &candidates,
                            const std::function<bool()> &poll = std::function<bool()>());

    const DeltaStats &stats() const { return stats_; }

private:
    struct Plan;
    // Sorts and checks the changes and decides what happens to every id. False = nothing changes.
    bool prepare(Plan &plan, const HnswParams *graph, HnswGraph &bg);

    const VectorSet &base_;
    std::uint32_t growth_ = DEFAULT_GROWTH_PERCENT;
    bool open_row_ = false;
    std::vector<std::int64_t> adds_;      // id of staged row i
    SnapshotBuffer staged_;               // adds_.size() rows of row_stride floats
    std::vector<std::int64_t> removes_;
    DeltaStats stats_;
};

// The unit of the change scan: a level-0 link list of an HNSW graph is 132 bytes, a 128-dim row 512.
constexpr std::uint64_t DIFF_BLOCK = 512;

// The runs of DIFF_BLOCK-byte blocks within `candidates` where a and b (size bytes each) differ:
// adjacent differing blocks make one run, no run crosses a multiple of chunk or is longer than
// chunk. candidates may overlap and come in any order; they are clipped to size. Offsets are
// multiples of DIFF_BLOCK, lengths too except at the end of the file.
std::vector<ByteRange> snapshot_diff(const std::uint8_t *a, const std::uint8_t *b, std::uint64_t size,
                                     const std::vector<ByteRange> &candidates, std::uint64_t chunk);

// Writes the checksum of the snapshot in copy, which differs from base (the same size, a valid
// sealed snapshot) exactly in `runs` (snapshot_diff of the two): the base's checksum combined with
// the contributions of the old and new bytes of the runs (docs/format.md: the checksum is an XOR
// over words). The first run must start at 0 (the header changed). Costs the bytes of the runs, not
// the file.
void seal_in_place(const std::uint8_t *base, std::uint8_t *copy, std::uint64_t size, const std::vector<ByteRange> &runs);

} // namespace vvector

#endif
