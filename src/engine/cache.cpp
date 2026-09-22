#include "cache.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vvector {

namespace {

[[noreturn]] void fail(const std::string &what, const std::string &path, bool with_errno = true)
{
    std::string msg = what + " " + path;
    if (with_errno) msg += std::string(": ") + std::strerror(errno);
    throw std::runtime_error(msg);
}

// Like mkdir -p. New directories get mode 0700.
void make_dir(const std::string &path)
{
    for (std::size_t at = 1; at <= path.size(); ++at) {
        if (at != path.size() && path[at] != '/') continue;
        const std::string part = path.substr(0, at);
        if (mkdir(part.c_str(), 0700) != 0 && errno != EEXIST) fail("cannot create directory", part);
    }
}

// True if the file starts with the vvector magic.
bool file_has_magic(const std::string &path)
{
    std::uint8_t head[sizeof(SNAPSHOT_MAGIC)];
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const ssize_t got = ::read(fd, head, sizeof(head));
    ::close(fd);
    return got == static_cast<ssize_t>(sizeof(head)) && snapshot_has_magic(head, sizeof(head));
}

// Writes a small text file atomically: temp file, then rename.
void write_atomically(const std::string &path, const std::string &content)
{
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) fail("cannot create", tmp);
    const bool ok = ::write(fd, content.data(), content.size()) == static_cast<ssize_t>(content.size()) &&
                    ::fsync(fd) == 0;
    ::close(fd);
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        fail("cannot write", path);
    }
}

} // namespace

bool valid_index_name(const std::string &index)
{
    if (index.empty() || index.size() > 64) return false;
    for (char c : index)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}

static std::string index_dir(const std::string &cache_dir, const std::string &index)
{
    if (!valid_index_name(index))
        throw std::runtime_error("index name '" + index + "' is not valid: use letters, digits and underscore");
    if (cache_dir.empty() || cache_dir[0] != '/')
        throw std::runtime_error("cache_dir '" + cache_dir + "' must be an absolute path");
    return cache_dir + "/" + index;
}

std::string snapshot_path(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id)
{
    return index_dir(cache_dir, index) + "/" + std::to_string(snapshot_id) + ".vv";
}

bool read_active(const std::string &cache_dir, const std::string &index, std::int64_t &snapshot_id)
{
    const std::string path = index_dir(cache_dir, index) + "/ACTIVE";
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    long long id = 0;
    const bool ok = std::fscanf(f, "%lld", &id) == 1;
    std::fclose(f);
    snapshot_id = id;
    return ok;
}

std::vector<std::string> list_cached_indexes(const std::string &cache_dir)
{
    std::vector<std::string> indexes;
    DIR *d = opendir(cache_dir.c_str());
    if (!d) return indexes;
    while (dirent *e = readdir(d))
        if (valid_index_name(e->d_name)) indexes.push_back(e->d_name);
    closedir(d);
    return indexes;
}

// ---- MappedSnapshot

namespace {

// The mappings kept by open_active, by file path. An entry is used only while the file at that
// path is still the file that was mapped.
struct KeptMapping {
    void *map = nullptr;
    std::uint64_t size = 0;
    std::uint64_t dev = 0, ino = 0;
    VectorSet set;
    ~KeptMapping() { if (map) munmap(map, size); }
    bool is_file(const std::string &path) const {
        struct stat st;
        return stat(path.c_str(), &st) == 0 && static_cast<std::uint64_t>(st.st_dev) == dev &&
               static_cast<std::uint64_t>(st.st_ino) == ino &&
               static_cast<std::uint64_t>(st.st_size) == size;
    }
};
std::mutex kept_lock;
std::map<std::string, std::shared_ptr<KeptMapping>> kept_mappings;

} // namespace

MappedSnapshot::~MappedSnapshot()
{
    if (map_ && !kept_) munmap(map_, size_);
}

void MappedSnapshot::open(const std::string &path, bool verify_checksum)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open", path);
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); fail("cannot stat", path); }
    if (st.st_size == 0) { ::close(fd); fail("empty snapshot file", path, false); }
    void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) fail("cannot mmap", path);
    if (map_ && !kept_) munmap(map_, size_);
    kept_.reset();
    map_ = m;
    size_ = st.st_size;
    dev_ = st.st_dev;
    ino_ = st.st_ino;
    path_ = path;
    try {
        set_ = snapshot_open(static_cast<const std::uint8_t *>(map_), size_, verify_checksum);
    } catch (const std::runtime_error &e) {
        throw std::runtime_error(std::string(e.what()) + " in " + path);
    }
}

void MappedSnapshot::open_active(const std::string &cache_dir, const std::string &index)
{
    std::int64_t id = 0;
    if (!read_active(cache_dir, index, id))
        throw std::runtime_error("no snapshot cache for index '" + index + "' in " + cache_dir + ": run vload");
    const std::string path = snapshot_path(cache_dir, index, id);
    const std::string dir = path.substr(0, path.rfind('/') + 1);
    std::lock_guard<std::mutex> hold(kept_lock);
    for (auto it = kept_mappings.begin(); it != kept_mappings.end(); ) {
        const bool older = it->first != path && it->first.compare(0, dir.size(), dir) == 0;
        if (older || !it->second->is_file(it->first)) it = kept_mappings.erase(it);      // unmapped when its last query ends
        else ++it;
    }
    auto it = kept_mappings.find(path);
    if (it == kept_mappings.end()) {
        try {
            open(path, false);
        } catch (const std::runtime_error &e) {
            throw std::runtime_error(std::string("snapshot cache of index '") + index + "' is missing or damaged (" +
                                     e.what() + "): run vload");
        }
        std::shared_ptr<KeptMapping> keep(new KeptMapping);
        keep->map = map_; keep->size = size_; keep->dev = dev_; keep->ino = ino_; keep->set = set_;
        kept_ = keep;
        kept_mappings[path] = keep;
    } else {
        if (map_ && !kept_) munmap(map_, size_);
        kept_ = it->second;
        map_ = it->second->map;
        size_ = it->second->size;
        set_ = it->second->set;
        path_ = path;
    }
    snapshot_id_ = id;
}

// ---- CacheWriter

CacheWriter::~CacheWriter() { discard(); }

void CacheWriter::discard()
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
        ::unlink(tmp_path_.c_str());
    }
}

void CacheWriter::begin(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id)
{
    dir_ = index_dir(cache_dir, index);
    cache_dir_ = cache_dir;
    index_ = index;
    make_dir(cache_dir);
    make_dir(dir_);
    snapshot_id_ = snapshot_id;
    final_path_ = snapshot_path(cache_dir, index, snapshot_id);
    tmp_path_ = final_path_ + ".tmp." + std::to_string(getpid());
    fd_ = ::open(tmp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) fail("cannot create", tmp_path_);
}

void CacheWriter::write_at(std::int64_t byte_offset, const char *data, std::uint64_t len)
{
    if (byte_offset < 0 || len == 0 || len > CHUNK_BYTES)
        throw std::runtime_error("bad piece at offset " + std::to_string(byte_offset) + " of " + std::to_string(len) + " bytes");
    std::uint64_t offset = static_cast<std::uint64_t>(byte_offset);
    std::uint64_t left = len;
    while (left > 0) {
        const ssize_t n = pwrite(fd_, data, left, static_cast<off_t>(offset));
        if (n <= 0) fail("cannot write", tmp_path_);
        data += n;
        offset += n;
        left -= n;
    }
    bytes_written_ += len;
    if (offset > end_offset_) end_offset_ = offset;
}

std::uint64_t CacheWriter::commit()
{
    // Every byte written exactly once: no piece missing, none twice. (The checksum below catches
    // the rest: a hole reads as zeros, and zero words do not match the expected checksum.)
    if (bytes_written_ != end_offset_)
        throw std::runtime_error("pieces are missing or duplicated: " + std::to_string(bytes_written_) +
                                 " bytes received for a file of " + std::to_string(end_offset_));
    if (::fsync(fd_) != 0) fail("cannot sync", tmp_path_);
    {
        MappedSnapshot check;
        check.open(tmp_path_, true);
    }
    std::int64_t previous = 0;
    const bool had_previous = read_active(cache_dir_, index_, previous);

    if (::rename(tmp_path_.c_str(), final_path_.c_str()) != 0) fail("cannot rename to", final_path_);
    ::close(fd_);
    fd_ = -1;
    write_atomically(dir_ + "/ACTIVE", std::to_string(snapshot_id_) + "\n");

    // Keep the new and the previous snapshot. Remove other vvector files only.
    const std::string keep_new = std::to_string(snapshot_id_) + ".vv";
    const std::string keep_old = had_previous ? std::to_string(previous) + ".vv" : keep_new;
    if (DIR *d = opendir(dir_.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string name = e->d_name;
            if (name == keep_new || name == keep_old || name == "ACTIVE" || name == "." || name == "..") continue;
            const std::string path = dir_ + "/" + name;
            struct stat st;
            if (lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && file_has_magic(path)) ::unlink(path.c_str());
        }
        closedir(d);
    }
    return end_offset_;
}

} // namespace vvector
