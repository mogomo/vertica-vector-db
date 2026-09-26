// vvector engine: per-node snapshot cache.
//   <cache_dir>/<index>/<snapshot_id>.vv   snapshot file, read with mmap
//   <cache_dir>/<index>/ACTIVE             text file with the active snapshot id
//   <cache_dir>/<index>/OPTIONS            text file with the index defaults (set_index_options);
//                                          it may name another cache directory of the index (the
//                                          index option cache_dir): a query then reads ACTIVE,
//                                          OPTIONS and the snapshot there (one hop, never a chain)
// POSIX only. No Vertica includes.
#ifndef VVECTOR_ENGINE_CACHE_H
#define VVECTOR_ENGINE_CACHE_H

#include "snapshot.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vvector {

constexpr const char *DEFAULT_CACHE_DIR = "/tmp/vvector";

// Snapshots travel through Vertica in chunks of this size (the last one is shorter).
constexpr std::uint64_t CHUNK_BYTES = 8u * 1024u * 1024u;

// How long a query trusts what it last read from ACTIVE and OPTIONS, in milliseconds. Within this
// time a warm query does no file system work at all. A newer snapshot is noticed at the latest
// after this time, and at once when the query needs it (see MappedSnapshot::open_active).
constexpr int ACTIVE_CHECK_MS = 200;

// Index names become directory names: letters, digits and underscore, 1 to 64 characters.
bool valid_index_name(const std::string &index);

std::string snapshot_path(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id);

// <cache_dir>/<index>, created (mode 0700) when it is missing.
std::string ensure_index_dir(const std::string &cache_dir, const std::string &index);

// Returns false when the index has no ACTIVE file in this cache.
bool read_active(const std::string &cache_dir, const std::string &index, std::int64_t &snapshot_id);

// Names of the indexes that have an ACTIVE or an OPTIONS file in this cache, in directories owned
// by this process's user.
std::vector<std::string> list_cached_indexes(const std::string &cache_dir);

// A cache directory given as an index option: an absolute path of letters, digits and / . _ -,
// without "." or ".." components, at most 1000 characters.
bool valid_cache_dir(const std::string &dir);

// Index defaults: name -> value. Names: precision, freshness, ef_search, threads, memory_mode,
// cache_dir (the directory that holds the index, when it is not this one).
using IndexOptions = std::map<std::string, std::string>;

// Parses "name=value" items separated by newlines or commas. Empty values are left out.
// Throws std::runtime_error on an unknown name or a malformed item.
IndexOptions parse_index_options(const std::string &text);

// Writes the OPTIONS file of an index (creating the directories), atomically.
void write_index_options(const std::string &cache_dir, const std::string &index, const IndexOptions &options);

// Reads the OPTIONS file. Missing file = no options.
IndexOptions read_index_options(const std::string &cache_dir, const std::string &index);

// Read-ahead of a query mapping. The kernel is asked (MADV_WILLNEED) to read the sections a search
// needs, in the order it needs them: ids, id index, tombstones, sq8 header, codes and sums, graph
// header, levels, level 0, upper index and upper levels, and the float rows last; every section is
// asked for whole or not at all, within PREWARM_BUDGET_BYTES per mapping. The advice is served before
// the first search of the mapping runs, so the budget bounds that wait to a few seconds of disk.
// Beyond the budget: a file that fits in the node's available memory is left to the kernel, whose
// read-around (read_ahead_kb, 4 MB per page fault on the test cluster) warms it at sequential speed
// as searches touch it (a cold 63 GB HNSW file: the first search 46 s, a cold batch of 1000 queries
// 98 to 129 s, then everything resident; the 100M proof, docs/design.md); a file that cannot fit
// gets MADV_RANDOM on those sections, so a search reads the pages it needs and nothing around them
// (a reload would thrash and evict every other index; with the mark a cold search reads some
// 20,000 pages at the disk's latency, 12 to 39 s there, and a cold batch takes far longer than the
// reload). What every search reads whole is always asked for: the rows of a flat index, the codes of
// a flat coded index. vload and load_all read the whole file. compact (memory_mode compact): the
// float rows get MADV_RANDOM whatever the budget (rescoring reads a few of them per query).
constexpr std::uint64_t PREWARM_BUDGET_BYTES = 8ull << 30;

struct PrewarmRange {
    std::uint64_t offset = 0;   // from the start of the file, page aligned by the caller
    std::uint64_t bytes = 0;
    bool willneed = true;       // false: MADV_RANDOM
};

// The ranges prewarm asks for, in order, for the sections of `set` mapped at `data` (a file of
// `size` bytes). random_beyond: sections beyond the budget are marked MADV_RANDOM (the file does not
// fit in memory); else they are left out (no advice). Pure, so tests can check the order and the
// budget without a file.
std::vector<PrewarmRange> prewarm_plan(const std::uint8_t *data, std::uint64_t size, const VectorSet &set,
                                       bool compact, std::uint64_t budget = PREWARM_BUDGET_BYTES,
                                       bool random_beyond = true);
// The memory a new file could occupy now: MemAvailable of /proc/meminfo (free plus reclaimable page
// cache); the largest value where it cannot be read.
std::uint64_t memory_available();

// A snapshot file mapped read-only. Throws std::runtime_error with the cause.
class MappedSnapshot {
public:
    MappedSnapshot() = default;
    ~MappedSnapshot();
    MappedSnapshot(const MappedSnapshot &) = delete;
    MappedSnapshot &operator=(const MappedSnapshot &) = delete;

    // verify: read and check the whole file (vload). Otherwise (a query mapping) the kernel is asked
    // to read the file ahead (prewarm_plan above); compact (memory_mode compact, an index with sq8
    // codes): everything but the float rows, which only rescoring reads, a few rows per query.
    // prewarm false: no read-ahead advice (vinfo, which must not load a file just to report how much
    // of it is in memory).
    // writable_copy: a private copy-on-write mapping (MAP_PRIVATE, read-write) for an incremental
    // build in place (delta.h): writes change the mapping, never the file, and cost memory for the
    // pages written only. No read-ahead, never kept.
    void open(const std::string &path, bool verify, bool compact = false, bool prewarm = true, bool writable_copy = false);
    // The bytes of a writable copy (open with writable_copy), else null.
    std::uint8_t *writable() { return writable_ ? static_cast<std::uint8_t *>(map_) : nullptr; }
    const std::uint8_t *data() const { return static_cast<const std::uint8_t *>(map_); }
    // Opens the active snapshot of an index. The mapping is kept by the process and shared by later
    // calls: a new mapping of a large file pays a page fault for every page a search touches, which
    // costs several times the search. What ACTIVE and OPTIONS say is trusted for ACTIVE_CHECK_MS;
    // after that, or when the kept snapshot is older than at_least, both are read again (at_least =
    // the largest int64: always read them again). A kept
    // mapping is dropped when its file is gone or replaced, or its index has a newer snapshot.
    void open_active(const std::string &cache_dir, const std::string &index, std::int64_t at_least = 0,
                     bool prewarm = true);

    const VectorSet &vectors() const { return set_; }
    std::int64_t snapshot_id() const { return snapshot_id_; }
    const std::string &path() const { return path_; }
    std::uint64_t size() const { return size_; }
    // Bytes of the mapped file that are in memory now (page cache), from mincore: what a query can
    // read without going to disk. Takes about 1 ms per 4 GB.
    std::uint64_t resident_bytes() const;
    // The index defaults read with the snapshot (open_active only).
    const IndexOptions &options() const { return options_; }

private:
    void *map_ = nullptr;
    std::uint64_t size_ = 0;
    bool writable_ = false;
    std::uint64_t dev_ = 0, ino_ = 0;     // the file that is mapped
    std::shared_ptr<void> kept_;          // open_active: the mapping is shared and outlives this object
    std::int64_t snapshot_id_ = 0;
    std::string path_;
    VectorSet set_;
    IndexOptions options_;
};

// Gives back the kept mappings (MappedSnapshot::open_active) of indexes that had no query for
// idle_ms milliseconds; a mapping still in use stays valid until its last user ends. open_active
// calls it with 10 minutes, so an unregistered index or an old cache directory does not keep its
// file mapped for the life of an unfenced process. Returns how many were given back.
std::size_t release_idle_mappings(std::int64_t idle_ms);

// Writes one snapshot file from pieces that may arrive in any order, then makes it the active one.
// A piece is "these bytes at this offset": a whole 8 MB chunk of a whole copy, or a run of changed
// bytes of a patch (milestone M7), which is written over a copy of the base snapshot. A failed or
// abandoned load leaves the cache as it was.
class CacheWriter {
public:
    CacheWriter() = default;
    ~CacheWriter();
    CacheWriter(const CacheWriter &) = delete;
    CacheWriter &operator=(const CacheWriter &) = delete;

    // base_snapshot > 0: the file starts as a copy of <index>/<base_snapshot>.vv in the same
    // directory, which must be a regular file of this user; the pieces are then written over it.
    // Throws when the base is missing. reflink: a reflink clone where the file system has it (xfs
    // reflink=1, btrfs), else a copy by read and write. A reflink is free, but every write into it
    // then unshares an extent (copy-on-write): 3 ms per scattered 512-byte write measured on xfs, 300
    // to 437 s for 100,000, against 0.7 s into a plain copy (sessions 18 and 19). So the caller asks
    // for it only when the patch has few runs (vload: PATCH_REFLINK_RUNS).
    void begin(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id, std::int64_t base_snapshot = 0,
               bool reflink = true);
    // A load in several passes (a large snapshot): every pass writes its pieces into the partial file
    // <snapshot>.vv.part.<part> (part: letters and digits naming the load); the first pass creates it
    // (from the base, if given), later passes (resume) continue it, keep() ends a pass without
    // verifying, commit() the last one.
    void begin_part(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id,
                    const std::string &part, bool resume, std::int64_t base_snapshot = 0, bool reflink = true);
    // Pieces may arrive in any order, 1 to CHUNK_BYTES bytes each. A piece written twice is fine
    // (the same bytes); a missing piece is found by the checksum, unless it was all zero (a whole
    // copy leaves such chunks out: the file is sized from its header).
    void write_at(std::int64_t byte_offset, const char *data, std::uint64_t len);
    // Ends a pass of a load in several passes: the partial file stays for the next pass.
    void keep();

    // Sizes the file from its header, verifies it (structure, checksum, id order, links), renames it
    // into place, flips ACTIVE (write temp, rename), syncs the directory, gives the previous
    // snapshot's pages back to the kernel, and removes snapshot files other than the new and the
    // previously active one. Only files in the index's directory that start with the vvector magic
    // are ever removed. Returns the snapshot size in bytes.
    std::uint64_t commit();

private:
    void discard();
    void start_from_base(std::int64_t base_snapshot, bool reflink);
    int fd_ = -1;
    std::string cache_dir_, index_, dir_, tmp_path_, final_path_;
    std::int64_t snapshot_id_ = 0;
    std::uint64_t bytes_written_ = 0, end_offset_ = 0;
    bool in_parts_ = false;
};

// Copies the content of one file into another. With reflink: a reflink clone where the file system
// supports it (xfs with reflink=1, btrfs), else copy_file_range, else read and write; without: read
// and write only, because copy_file_range on a reflink file system is a reflink too (session 19).
// Both must be open; to_fd is truncated first. Throws std::runtime_error with the cause. Returns how
// it was done: "reflink", "copy_file_range" or "copy".
const char *clone_file(int from_fd, int to_fd, const std::string &what, bool reflink = true);

} // namespace vvector

#endif
