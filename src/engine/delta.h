// vvector engine: incremental build. A new snapshot made from a base snapshot and the changes of
// the journal since the base was built (consolidated: one add or delete per id).
//
// The base's rows, ids, tombstones and graph are copied. A deleted or changed id's position becomes
// a tombstone; a new or changed vector is appended as a new position and, on an HNSW index,
// inserted into the graph with the insertion code of the full build (hnsw_extend). Appended ids
// are not in id order, so the new snapshot gets an id_index (FLAG_ID_INDEX) unless every appended
// id is larger than every id before. An id can then be at several positions: the highest is the
// only one that can be live (VectorSet::find).
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

    // Applies the changes and writes the new snapshot to out, with base_snapshot as its base.
    // graph: the graph parameters for an HNSW base (p.m must be the base's m), null for a flat one.
    // Returns false, and leaves out alone, when the changes change nothing. Throws
    // std::runtime_error on an id given twice, on more than 4,294,967,295 positions, on a graph
    // parameter that does not fit the base, and Cancelled (parallel.h) when poll says so.
    bool finish(std::int64_t max_ver, std::int64_t base_snapshot, SnapshotBuffer &out, const HnswParams *graph,
                const std::function<bool()> &poll = std::function<bool()>());

    const DeltaStats &stats() const { return stats_; }

private:
    const VectorSet &base_;
    bool open_row_ = false;
    std::vector<std::int64_t> adds_;      // id of staged row i
    SnapshotBuffer staged_;               // adds_.size() rows of row_stride floats
    std::vector<std::int64_t> removes_;
    DeltaStats stats_;
};

} // namespace vvector

#endif
