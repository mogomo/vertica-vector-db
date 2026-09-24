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

// A snapshot file mapped read-only. Throws std::runtime_error with the cause.
class MappedSnapshot {
public:
    MappedSnapshot() = default;
    ~MappedSnapshot();
    MappedSnapshot(const MappedSnapshot &) = delete;
    MappedSnapshot &operator=(const MappedSnapshot &) = delete;

    // verify: read and check the whole file (vload). Otherwise (a query mapping) the kernel is asked
    // to read the file ahead; compact (memory_mode compact, an index with sq8 codes): everything but
    // the float rows, which only rescoring reads, a few rows per query. prewarm false: no read-ahead
    // advice (vinfo, which must not load a file just to report how much of it is in memory).
    void open(const std::string &path, bool verify, bool compact = false, bool prewarm = true);
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
    std::uint64_t dev_ = 0, ino_ = 0;     // the file that is mapped
    std::shared_ptr<void> kept_;          // open_active: the mapping is shared and outlives this object
    std::int64_t snapshot_id_ = 0;
    std::string path_;
    VectorSet set_;
    IndexOptions options_;
};

// Writes one snapshot file from pieces that may arrive in any order, then
// makes it the active one. A failed or abandoned load leaves the cache as it was.
class CacheWriter {
public:
    CacheWriter() = default;
    ~CacheWriter();
    CacheWriter(const CacheWriter &) = delete;
    CacheWriter &operator=(const CacheWriter &) = delete;

    void begin(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id);
    // Pieces may arrive in any order. Together they must cover the file exactly once.
    void write_at(std::int64_t byte_offset, const char *data, std::uint64_t len);

    // Verifies the file (size, structure, checksum, id order), renames it into place,
    // flips ACTIVE (write temp, rename), syncs the directory, and removes snapshot files other
    // than the new and the previously active one. Only files in the index's directory that start
    // with the vvector magic are ever removed. Returns the snapshot size in bytes.
    std::uint64_t commit();

private:
    void discard();
    int fd_ = -1;
    std::string cache_dir_, index_, dir_, tmp_path_, final_path_;
    std::int64_t snapshot_id_ = 0;
    std::uint64_t bytes_written_ = 0, end_offset_ = 0;
};

} // namespace vvector

#endif
