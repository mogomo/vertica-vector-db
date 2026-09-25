// vvector engine: snapshot binary format, version 2. See docs/format.md.
// Little-endian only. Every section starts on a 64-byte boundary.
#ifndef VVECTOR_ENGINE_SNAPSHOT_H
#define VVECTOR_ENGINE_SNAPSHOT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "vvector snapshots are little-endian only"
#endif

namespace vvector {

constexpr char SNAPSHOT_MAGIC[8] = {'V', 'V', 'E', 'C', 'T', 'O', 'R', '1'};

constexpr std::uint32_t FLAG_HNSW = 1u;          // graph section present (milestone M2)
constexpr std::uint32_t FLAG_NORMALISED = 2u;    // vectors have unit length (cosine indexes)
constexpr std::uint32_t FLAG_SQ8 = 4u;           // int8 codes section present (sq8.h)
constexpr std::uint32_t FLAG_ID_INDEX = 8u;      // id_index section present; ids are then in any order
constexpr std::uint32_t FLAG_TOMBSTONES = 16u;   // tombstones bitset present
constexpr std::uint32_t KNOWN_FLAGS = 31u;

constexpr std::uint64_t HEADER_BYTES = 256;
constexpr std::uint64_t SECTION_ALIGN = 64;      // one cache line: every vector row starts on one
constexpr std::uint32_t LANES = 16;              // floats per SIMD block; row_stride is a multiple of it
constexpr std::uint32_t MAX_DIMS = 32768;
constexpr std::uint64_t MAX_COUNT = 0xFFFFFFFFull;   // positions are uint32

// Floats per stored row: dims rounded up to a multiple of 16. The padding is zero.
inline std::uint32_t row_stride_for(std::uint32_t dims) { return (dims + LANES - 1) / LANES * LANES; }

// The distance measure an index is built for. Same meaning as Vertica's built-in functions.
enum class Metric : std::uint32_t {
    L2 = 1,        // VECTOR_L2: Euclidean distance, smaller is closer
    Cosine = 2,    // COSINE_SIMILARITY: larger is closer
    Dot = 3,       // DOT_PRODUCT: larger is closer
    L1 = 4,        // Manhattan distance (no built-in): smaller is closer
};

// Parses 'l2', 'cosine', 'dot' or 'l1'. Throws std::runtime_error for anything else.
Metric parse_metric(const std::string &name);
const char *metric_name(Metric m);

// 256 bytes. Section offsets are from the start of the file; 0 = section absent.
struct SnapshotHeader {
    char magic[8];
    std::uint32_t format_version;
    std::uint32_t flags;
    std::uint64_t count;          // number of positions (vectors), tombstoned ones included
    std::uint32_t dims;           // elements per vector
    std::uint32_t row_stride;     // floats per stored row: row_stride_for(dims)
    std::uint32_t metric;         // Metric
    std::uint32_t reserved0;
    std::int64_t max_ver;         // journal watermark of the build
    std::int64_t base_snapshot;   // snapshot this one was built from, 0 = full build
    std::uint64_t tombstones;     // number of set bits in the tombstones section
    std::uint64_t checksum;       // over the whole file, with this field as 0
    std::uint64_t total_bytes;    // file size, a multiple of 64
    std::uint64_t off_vectors;    // float32[count x row_stride]
    std::uint64_t off_ids;        // int64[count]
    std::uint64_t off_id_index;   // uint32[count], FLAG_ID_INDEX
    std::uint64_t off_tombstones; // uint64[(count + 63) / 64], FLAG_TOMBSTONES
    std::uint64_t off_sq8;        // FLAG_SQ8
    std::uint64_t sq8_bytes;
    std::uint64_t off_graph;      // FLAG_HNSW
    std::uint64_t graph_bytes;
    std::uint64_t reserved[14];
};
static_assert(sizeof(SnapshotHeader) == HEADER_BYTES, "snapshot header must be 256 bytes");

// A snapshot opened for reading. Points into bytes owned by someone else.
struct VectorSet {
    std::uint64_t count = 0;
    std::uint32_t dims = 0;
    std::uint32_t row_stride = 0;
    Metric metric = Metric::L2;
    std::uint32_t flags = 0;
    std::int64_t max_ver = 0;
    std::int64_t base_snapshot = 0;
    std::uint64_t tombstones = 0;
    const float *vectors = nullptr;
    const std::int64_t *ids = nullptr;
    const std::uint32_t *id_index = nullptr;         // null unless FLAG_ID_INDEX
    const std::uint64_t *tombstone_bits = nullptr;   // null unless FLAG_TOMBSTONES
    const std::uint8_t *sq8 = nullptr;
    std::uint64_t sq8_bytes = 0;
    const std::uint8_t *graph = nullptr;
    std::uint64_t graph_bytes = 0;

    const float *vector(std::uint64_t i) const { return vectors + i * row_stride; }
    bool has_graph() const { return (flags & FLAG_HNSW) != 0; }
    bool normalised() const { return (flags & FLAG_NORMALISED) != 0; }
    bool dead(std::uint64_t i) const { return tombstone_bits && (tombstone_bits[i >> 6] >> (i & 63) & 1u); }
    // Position of id, or -1. Binary search over ids, or over id_index when present.
    std::int64_t find(std::int64_t id) const;
    // out[i] = find(want[i]) for n ids in ascending order, in one pass: each search starts where the
    // one before ended and gallops forward (for many ids far faster than n binary searches).
    void find_sorted(const std::int64_t *want, std::uint64_t n, std::int64_t *out) const;
};

// Owning, zero-filled snapshot bytes, aligned to 4096 bytes. On Linux the memory is mapped
// anonymously and grows with mremap: growing never copies, and pages that are never written
// cost no memory. Elsewhere (unit tests on other systems) growing copies.
class SnapshotBuffer {
public:
    SnapshotBuffer() = default;
    ~SnapshotBuffer();
    SnapshotBuffer(const SnapshotBuffer &) = delete;
    SnapshotBuffer &operator=(const SnapshotBuffer &) = delete;

    // A new zero-filled buffer of exactly this many bytes. Earlier content is dropped.
    void allocate(std::uint64_t bytes);
    // Makes room for at least `bytes` and keeps the content. Bytes never written read as 0.
    void reserve(std::uint64_t bytes);
    // Sets the size. It must not exceed what reserve made room for.
    void set_size(std::uint64_t bytes);
    // Exchanges the memory of two buffers. Nothing is copied.
    void swap(SnapshotBuffer &other) noexcept;
    // Drops the memory.
    void clear() { release(); }
    // Backs the buffer with a file in dir instead of anonymous memory (Linux; elsewhere a no-op):
    // its pages are then page cache that the kernel can write out and reclaim, so a buffer larger
    // than the free memory still works (slower). The file is unlinked at once: nothing is left
    // behind, also after a crash. Call it while the buffer is empty.
    void back_with_file(const std::string &dir);
    bool file_backed() const { return fd_ >= 0; }
    const std::string &file_dir() const { return dir_; }

    std::uint8_t *data() { return data_; }
    const std::uint8_t *data() const { return data_; }
    std::uint64_t size() const { return size_; }
    std::uint64_t capacity() const { return capacity_; }

private:
    void release();
    std::uint8_t *data_ = nullptr;
    std::uint64_t size_ = 0, capacity_ = 0;
    int fd_ = -1;                 // back_with_file: the unlinked file
    std::string dir_;             // and its directory
};

// Fills the section offsets and total_bytes of h from count, row_stride, flags, sq8_bytes and graph_bytes.
void snapshot_layout(SnapshotHeader &h);

// Checksum of a complete snapshot. The stored checksum field counts as 0.
std::uint64_t snapshot_checksum(const std::uint8_t *data, std::uint64_t size);

// The checksum is the XOR of one value per 8-byte word, mixed with the word's position in the file.
// So the checksum of a file is the XOR of the checksums of its parts, in any order: sections that
// are written by separate statements can be summed up afterwards. A zero word contributes nothing.
// first_word = file offset of data / 8. size is a multiple of 8.
std::uint64_t snapshot_checksum_part(const std::uint8_t *data, std::uint64_t size, std::uint64_t first_word);

// Validates the bytes and returns a view on them. data must be 64-byte aligned and must outlive
// the view. Throws std::runtime_error with the cause. verify reads the whole file (checksum, id
// order, tombstone count): use it in vload, not in every query.
VectorSet snapshot_open(const std::uint8_t *data, std::uint64_t size, bool verify);

// True if the bytes start with the vvector magic. Used before deleting cache files.
bool snapshot_has_magic(const std::uint8_t *data, std::uint64_t size);

// A section that SnapshotBuilder::finish builds in place, after the rows are sorted: the HNSW
// graph (hnsw.h). bytes() gives its size for the ids in position order; fill() writes it into the
// zero-filled section of the finished snapshot s, or, for a build in a file, into zero-filled
// memory that is copied there afterwards: fill writes only through its section argument. The
// checksum is computed after fill.
struct GraphSection {
    std::function<std::uint64_t(const std::int64_t *ids, std::uint64_t n)> bytes;
    std::function<void(const VectorSet &s, std::uint8_t *section)> fill;
};

// The same for the sq8 codes (sq8.h): bytes() gives the size for n rows of row_stride floats; fill()
// trains on the finished rows of s and writes the section. Filled before the graph.
struct CodeSection {
    std::function<std::uint64_t(std::uint64_t n, std::uint32_t row_stride)> bytes;
    std::function<void(const VectorSet &s, std::uint8_t *section)> fill;
};

// Builds a full snapshot. Rows are written straight into the final buffer as they arrive; ids
// may arrive in any order and are sorted at finish by moving the rows in place, so the builder
// never holds a second copy of the vectors in memory. A build in a file sorts into a second file
// instead (random writes to a file run at the disk's speed) and builds the graph in memory.
// Memory: 4 x row_stride + 12 bytes per vector (rows, ids, sort order), plus the unwritten part
// of the last growth step, which costs address space only on Linux.
class SnapshotBuilder {
public:
    explicit SnapshotBuilder(Metric metric) : metric_(metric) {}

    // Starts the row of id and returns it: row_stride floats, zero. Write dims floats, then call
    // end_row. dims must be the same for every row, 1 to MAX_DIMS.
    float *begin_row(std::int64_t id, std::uint32_t dims);
    // Finishes the row: normalises it for a cosine index.
    void end_row();
    // begin_row, copy, end_row.
    void add(std::int64_t id, const float *v, std::uint32_t dims);

    std::uint64_t count() const { return ids_.size(); }
    std::uint32_t dims() const { return dims_; }
    // Builds in a file in dir instead of anonymous memory (SnapshotBuffer::back_with_file). Call it
    // before the first row.
    void build_in_file(const std::string &dir) { buffer_.back_with_file(dir); }

    // Sorts by id, writes ids, header, the sq8 and graph sections if given, and the checksum into
    // the buffer and hands it over to out. Throws std::runtime_error on a repeated id or no vectors,
    // and whatever codes->fill or graph->fill throw. The builder is empty afterwards.
    void finish(std::int64_t max_ver, SnapshotBuffer &out, const GraphSection *graph = nullptr,
                const CodeSection *codes = nullptr);

private:
    Metric metric_;
    std::uint32_t dims_ = 0, stride_ = 0;
    bool ordered_ = true, open_row_ = false;
    std::vector<std::int64_t> ids_;
    SnapshotBuffer buffer_;       // header space, then the rows
};

} // namespace vvector

#endif
