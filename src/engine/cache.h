// vvector engine: per-node snapshot cache.
//   <cache_dir>/<index>/<snapshot_id>.vv   snapshot file, read with mmap
//   <cache_dir>/<index>/ACTIVE             text file with the active snapshot id
// POSIX only. No Vertica includes.
#ifndef VVECTOR_ENGINE_CACHE_H
#define VVECTOR_ENGINE_CACHE_H

#include "snapshot.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vvector {

constexpr const char *DEFAULT_CACHE_DIR = "/tmp/vvector";

// Snapshots travel through Vertica in chunks of this size (the last one is shorter).
constexpr std::uint64_t CHUNK_BYTES = 8u * 1024u * 1024u;

// Index names become directory names: letters, digits and underscore, 1 to 64 characters.
bool valid_index_name(const std::string &index);

std::string snapshot_path(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id);

// Returns false when the index has no ACTIVE file in this cache.
bool read_active(const std::string &cache_dir, const std::string &index, std::int64_t &snapshot_id);

// Names of the indexes that have a directory in this cache.
std::vector<std::string> list_cached_indexes(const std::string &cache_dir);

// A snapshot file mapped read-only. Throws std::runtime_error with the cause.
class MappedSnapshot {
public:
    MappedSnapshot() = default;
    ~MappedSnapshot();
    MappedSnapshot(const MappedSnapshot &) = delete;
    MappedSnapshot &operator=(const MappedSnapshot &) = delete;

    void open(const std::string &path, bool verify_checksum);
    // Opens the active snapshot of an index. The mapping is kept by the process and shared by later
    // calls: a new mapping of a large file pays a page fault for every page a search touches, which
    // costs several times the search. A kept mapping is dropped at the next open_active of any
    // index when its file is gone or replaced, or when its index has a newer active snapshot.
    void open_active(const std::string &cache_dir, const std::string &index);

    const VectorSet &vectors() const { return set_; }
    std::int64_t snapshot_id() const { return snapshot_id_; }
    const std::string &path() const { return path_; }
    std::uint64_t size() const { return size_; }

private:
    void *map_ = nullptr;
    std::uint64_t size_ = 0;
    std::uint64_t dev_ = 0, ino_ = 0;     // the file that is mapped
    std::shared_ptr<void> kept_;          // open_active: the mapping is shared and outlives this object
    std::int64_t snapshot_id_ = 0;
    std::string path_;
    VectorSet set_;
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

    // Verifies the file (size, structure, checksum), renames it into place,
    // flips ACTIVE (write temp, rename) and removes snapshot files other than
    // the new and the previously active one. Only files in the index's
    // directory that start with the vvector magic are ever removed.
    // Returns the snapshot size in bytes.
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
