#include "cache.h"
#include "hnsw.h"
#include "sq8.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
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

// Makes the renames in a directory durable.
void sync_dir(const std::string &dir)
{
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open directory", dir);
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    if (!ok) fail("cannot sync directory", dir);
}

// The suffix of a temp file: unique per process and call (two sessions of one unfenced process can
// write the same file at the same time).
std::string temp_suffix()
{
    static std::atomic<unsigned> calls{0};
    return ".tmp." + std::to_string(getpid()) + "." + std::to_string(++calls);
}

// Writes a small text file atomically: temp file, then rename.
void write_atomically(const std::string &path, const std::string &content)
{
    const std::string tmp = path + temp_suffix();
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

// The whole content of a small regular file, or false. O_NONBLOCK: a FIFO in its place must not
// block the query.
bool read_small_file(const std::string &path, std::string &out)
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) return false;
    struct stat st;
    char buf[4096];
    const ssize_t n = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) ? ::read(fd, buf, sizeof(buf)) : -1;
    ::close(fd);
    if (n < 0) return false;
    out.assign(buf, static_cast<std::size_t>(n));
    return true;
}

bool is_file(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// True if path exists and belongs to the user of this process (the database's operating system
// user). vload makes every cache directory and file itself; a search maps a cache file without
// checking all of it, so it must never map a file that someone else could have written.
bool owned(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && st.st_uid == geteuid();
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

std::string ensure_index_dir(const std::string &cache_dir, const std::string &index)
{
    const std::string dir = index_dir(cache_dir, index);
    make_dir(cache_dir);
    make_dir(dir);
    return dir;
}

std::string snapshot_path(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id)
{
    return index_dir(cache_dir, index) + "/" + std::to_string(snapshot_id) + ".vv";
}

bool read_active(const std::string &cache_dir, const std::string &index, std::int64_t &snapshot_id)
{
    std::string text;
    if (!read_small_file(index_dir(cache_dir, index) + "/ACTIVE", text)) return false;
    long long id = 0;
    if (std::sscanf(text.c_str(), "%lld", &id) != 1) return false;
    snapshot_id = id;
    return true;
}

std::vector<std::string> list_cached_indexes(const std::string &cache_dir)
{
    std::vector<std::string> indexes;
    DIR *d = opendir(cache_dir.c_str());
    if (!d) return indexes;
    while (dirent *e = readdir(d)) {
        const std::string dir = cache_dir + "/" + e->d_name;
        if (valid_index_name(e->d_name) && owned(dir) && (is_file(dir + "/ACTIVE") || is_file(dir + "/OPTIONS")))
            indexes.push_back(e->d_name);
    }
    closedir(d);
    std::sort(indexes.begin(), indexes.end());
    return indexes;
}

bool valid_cache_dir(const std::string &dir)
{
    if (dir.size() < 2 || dir.size() > 1000 || dir[0] != '/') return false;
    std::size_t start = 1;
    for (std::size_t i = 1; i <= dir.size(); ++i) {
        if (i < dir.size() && dir[i] != '/') {
            const char c = dir[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
                return false;
            continue;
        }
        const std::string part = dir.substr(start, i - start);
        if (part.empty() ? i < dir.size() : part.find_first_not_of('.') == std::string::npos) return false;
        start = i + 1;
    }
    return true;
}

// ---- index options

static const char *const OPTION_NAMES[] = {"precision", "freshness", "ef_search", "threads", "memory_mode", "cache_dir"};

IndexOptions parse_index_options(const std::string &text)
{
    IndexOptions out;
    std::size_t at = 0;
    while (at <= text.size()) {
        std::size_t end = text.find_first_of(",\n", at);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(at, end - at);
        while (!item.empty() && (item.back() == ' ' || item.back() == '\r')) item.pop_back();
        while (!item.empty() && item.front() == ' ') item.erase(0, 1);
        at = end + 1;
        if (item.empty()) continue;
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) throw std::runtime_error("index option '" + item + "' is not name=value");
        const std::string name = item.substr(0, eq), value = item.substr(eq + 1);
        bool known = false;
        for (const char *n : OPTION_NAMES) known = known || name == n;
        if (!known) throw std::runtime_error("unknown index option '" + name + "'");
        if (name == "cache_dir") {
            if (!value.empty() && !valid_cache_dir(value))
                throw std::runtime_error("index option cache_dir: '" + value + "' is not an absolute path of letters, digits and / . _ -");
            if (!value.empty()) out[name] = value;
            continue;
        }
        for (char c : value)
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')))
                throw std::runtime_error("index option " + name + ": value '" + value + "' is not valid");
        if (!value.empty()) out[name] = value;
    }
    return out;
}

void write_index_options(const std::string &cache_dir, const std::string &index, const IndexOptions &options)
{
    const std::string dir = index_dir(cache_dir, index);
    make_dir(cache_dir);
    make_dir(dir);
    std::string text;
    for (const auto &o : options) text += o.first + "=" + o.second + "\n";
    write_atomically(dir + "/OPTIONS", text);
}

IndexOptions read_index_options(const std::string &cache_dir, const std::string &index)
{
    std::string text;
    if (!read_small_file(index_dir(cache_dir, index) + "/OPTIONS", text)) return IndexOptions();
    return parse_index_options(text);
}

// ---- MappedSnapshot

namespace {

using Clock = std::chrono::steady_clock;

// A mapping kept by open_active.
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

// What the process knows about one index of one cache directory.
struct IndexState {
    std::int64_t snapshot_id = 0;
    std::string path;
    std::shared_ptr<KeptMapping> mapping;
    IndexOptions options;
    Clock::time_point checked;
};

std::mutex state_lock;
std::map<std::string, IndexState> states;      // by <cache_dir>/<index>

} // namespace

// Read-ahead advice for a query mapping. compact: the float rows (the first section) are read at
// random by rescoring only; the sections after them (ids, codes, graph) are read ahead.
static void prewarm(void *m, std::uint64_t size, bool compact)
{
    SnapshotHeader h;
    if (!compact || size < sizeof(h)) { madvise(m, size, MADV_WILLNEED); return; }
    std::memcpy(&h, m, sizeof(h));
    const std::uint64_t page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    const std::uint64_t rest = h.off_ids / page * page;
    if (h.off_ids == 0 || h.off_ids >= size) { madvise(m, size, MADV_WILLNEED); return; }
    madvise(m, rest, MADV_RANDOM);
    madvise(static_cast<std::uint8_t *>(m) + rest, size - rest, MADV_WILLNEED);
}

std::uint64_t MappedSnapshot::resident_bytes() const
{
    if (!map_ || size_ == 0) return 0;
    const std::uint64_t page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    const std::uint64_t pages = (size_ + page - 1) / page;
#if defined(__APPLE__)
    std::vector<char> in(pages);
#else
    std::vector<unsigned char> in(pages);
#endif
    if (mincore(map_, size_, in.data()) != 0) return 0;
    std::uint64_t n = 0;
    for (auto v : in) n += v & 1;
    return std::min(n * page, size_);
}

MappedSnapshot::~MappedSnapshot()
{
    if (map_ && !kept_) munmap(map_, size_);
}

void MappedSnapshot::open(const std::string &path, bool verify, bool compact, bool prewarm_it)
{
    int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) fail("cannot open", path);
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); fail("cannot stat", path); }
    if (!S_ISREG(st.st_mode)) { ::close(fd); fail("not a regular file:", path, false); }
    if (st.st_uid != geteuid()) { ::close(fd); fail("not owned by the database's operating system user (vload writes every cache file):", path, false); }
    if (st.st_size == 0) { ::close(fd); fail("empty snapshot file", path, false); }
    void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) fail("cannot mmap", path);
    // A mapping for queries: ask the kernel to read the file ahead. It costs nothing measurable when
    // the file is in the page cache (vload reads all of it to verify it), and after a restart it
    // reads the file in large pieces instead of page by page as the search touches it. Pre-mapping
    // every page (MAP_POPULATE) was measured: it makes the first query of a session slower.
    if (!verify && prewarm_it) prewarm(m, st.st_size, compact);
    if (map_ && !kept_) munmap(map_, size_);
    kept_.reset();
    map_ = m;
    size_ = st.st_size;
    dev_ = st.st_dev;
    ino_ = st.st_ino;
    path_ = path;
    try {
        set_ = snapshot_open(static_cast<const std::uint8_t *>(map_), size_, verify);
        if (verify && set_.has_graph()) hnsw_open(set_, true);
        if (verify && (set_.flags & FLAG_SQ8)) sq8_open(set_, true);
    } catch (const std::runtime_error &e) {
        throw std::runtime_error(std::string(e.what()) + " in " + path);
    }
}

static std::size_t release_idle_locked(Clock::time_point now, std::int64_t idle_ms, const std::string &keep)
{
    std::size_t n = 0;
    for (auto it = states.begin(); it != states.end();) {
        if (it->first != keep && now - it->second.checked > std::chrono::milliseconds(idle_ms)) {
            it = states.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

std::size_t release_idle_mappings(std::int64_t idle_ms)
{
    std::lock_guard<std::mutex> hold(state_lock);
    return release_idle_locked(Clock::now(), idle_ms, std::string());
}

void MappedSnapshot::open_active(const std::string &cache_dir, const std::string &index, std::int64_t at_least,
                                 bool prewarm_it)
{
    const std::string key = index_dir(cache_dir, index);
    const Clock::time_point now = Clock::now();
    std::lock_guard<std::mutex> hold(state_lock);
    IndexState &st = states[key];
    const bool fresh = st.mapping && now - st.checked < std::chrono::milliseconds(ACTIVE_CHECK_MS) &&
                       st.snapshot_id >= at_least;
    if (!fresh) {
        release_idle_locked(now, 600000, key);
        std::int64_t id = 0;
        // The index directory, and the one its OPTIONS file names as the index's cache directory
        // (the index option cache_dir; one hop, never a chain).
        auto check_owner = [&](const std::string &dir) {
            struct stat dir_st;
            if (stat(index_dir(dir, index).c_str(), &dir_st) == 0 && dir_st.st_uid != geteuid()) {
                states.erase(key);
                throw std::runtime_error("no snapshot cache for index '" + index + "' in " + dir +
                                         ": the directory is not owned by the database's operating system user");
            }
        };
        auto options_of = [&](const std::string &dir) {
            try {
                return read_index_options(dir, index);
            } catch (const std::runtime_error &e) {
                throw std::runtime_error("index '" + index + "': OPTIONS file in the cache " + dir + ": " + e.what() + ": run vvector.load_all");
            }
        };
        std::string dir = cache_dir;
        check_owner(dir);
        IndexOptions options = options_of(dir);
        const auto home = options.find("cache_dir");
        if (home != options.end() && home->second != dir) {
            dir = home->second;
            check_owner(dir);
            options = options_of(dir);
        }
        options.erase("cache_dir");
        if (!read_active(dir, index, id)) {
            states.erase(key);
            throw std::runtime_error("no snapshot cache for index '" + index + "' in " + dir + ": run vload");
        }
        const std::string path = snapshot_path(dir, index, id);
        if (!st.mapping || st.path != path || !st.mapping->is_file(path)) {
            try {
                const auto mode = options.find("memory_mode");
                open(path, false, mode != options.end() && mode->second == "compact", prewarm_it);
            } catch (const std::runtime_error &e) {
                states.erase(key);
                throw std::runtime_error(std::string("snapshot cache of index '") + index + "' is missing or damaged (" +
                                         e.what() + "): run vload");
            }
            std::shared_ptr<KeptMapping> keep(new KeptMapping);
            keep->map = map_; keep->size = size_; keep->dev = dev_; keep->ino = ino_; keep->set = set_;
            kept_ = keep;
            st.mapping = keep;          // the mapping it replaces is unmapped when its last query ends
            st.path = path;
        }
        st.options = options;
        st.snapshot_id = id;
        st.checked = now;
    }
    if (kept_ != st.mapping) {
        if (map_ && !kept_) munmap(map_, size_);
        kept_ = st.mapping;
        map_ = st.mapping->map;
        size_ = st.mapping->size;
        dev_ = st.mapping->dev;
        ino_ = st.mapping->ino;
        set_ = st.mapping->set;
    }
    path_ = st.path;
    options_ = st.options;
    snapshot_id_ = st.snapshot_id;
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
    tmp_path_ = final_path_ + temp_suffix();
    fd_ = ::open(tmp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) fail("cannot create", tmp_path_);
}

void CacheWriter::begin_part(const std::string &cache_dir, const std::string &index, std::int64_t snapshot_id,
                             const std::string &part, bool resume)
{
    if (part.empty() || part.size() > 64 ||
        part.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") != std::string::npos)
        throw std::runtime_error("part must be 1 to 64 letters and digits");
    dir_ = index_dir(cache_dir, index);
    cache_dir_ = cache_dir;
    index_ = index;
    make_dir(cache_dir);
    make_dir(dir_);
    snapshot_id_ = snapshot_id;
    final_path_ = snapshot_path(cache_dir, index, snapshot_id);
    tmp_path_ = final_path_ + ".part." + part;
    in_parts_ = true;
    if (resume) {
        fd_ = ::open(tmp_path_.c_str(), O_RDWR | O_NOFOLLOW);
        if (fd_ < 0) fail("cannot continue the partial file (an earlier pass of this load failed?)", tmp_path_);
        struct stat st;
        if (fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid()) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("not a partial file of this user: " + tmp_path_);
        }
    } else {
        fd_ = ::open(tmp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
        if (fd_ < 0) fail("cannot create", tmp_path_);
    }
}

void CacheWriter::keep()
{
    if (fd_ < 0 || !in_parts_) throw std::runtime_error("keep() without a partial file");
    ::close(fd_);
    fd_ = -1;
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
    // the rest: a hole reads as zeros, and zero words do not match the expected checksum.) A load in
    // passes counts only its last pass here: the file size and the checksum check the whole file.
    if (in_parts_) {
        struct stat st;
        if (fstat(fd_, &st) != 0) fail("cannot stat", tmp_path_);
        end_offset_ = static_cast<std::uint64_t>(st.st_size);
    } else if (bytes_written_ != end_offset_)
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
    sync_dir(dir_);

    // Keep the new and the previous snapshot. Remove other vvector files only.
    const std::string keep_new = std::to_string(snapshot_id_) + ".vv";
    const std::string keep_old = had_previous ? std::to_string(previous) + ".vv" : keep_new;
    if (DIR *d = opendir(dir_.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string name = e->d_name;
            if (name == keep_new || name == keep_old || name == "ACTIVE" || name == "OPTIONS" || name == "." || name == "..")
                continue;
            const std::string path = dir_ + "/" + name;
            struct stat st;
            if (lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && file_has_magic(path)) ::unlink(path.c_str());
        }
        closedir(d);
    }
    return end_offset_;
}

} // namespace vvector
