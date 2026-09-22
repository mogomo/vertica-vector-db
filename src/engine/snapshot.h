// vvector engine: snapshot binary format. See docs/format.md.
// Little-endian only. Every section starts on an 8-byte boundary.
#ifndef VVECTOR_ENGINE_SNAPSHOT_H
#define VVECTOR_ENGINE_SNAPSHOT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "vvector snapshots are little-endian only"
#endif

namespace vvector {

constexpr char SNAPSHOT_MAGIC[8] = {'V', 'V', 'E', 'C', 'T', 'O', 'R', '1'};

constexpr std::uint32_t FLAG_HNSW = 1u;     // an HNSW graph section is stored (not written yet: see hnsw.h)

// The distance measure an index is built for. Same meaning as Vertica's built-in functions.
enum class Metric : std::uint32_t {
    L2 = 1,        // VECTOR_L2: smaller is closer
    Cosine = 2,    // COSINE_SIMILARITY: larger is closer
    Dot = 3,       // DOT_PRODUCT: larger is closer
};

// Parses 'l2', 'cosine' or 'dot'. Throws std::runtime_error for anything else.
Metric parse_metric(const std::string &name);
const char *metric_name(Metric m);

// 128 bytes. Section offsets are from the start of the file; 0 = section absent.
struct SnapshotHeader {
    char magic[8];
    std::uint32_t format_version;
    std::uint32_t flags;
    std::uint64_t count;          // number of vectors
    std::uint32_t dims;           // elements per vector
    std::uint32_t metric;         // Metric
    std::int64_t max_ver;         // journal watermark of the build
    std::uint64_t checksum;       // over the whole file, with this field as 0
    std::uint64_t total_bytes;    // multiple of 8
    std::uint64_t off_ids;        // int64[count], ascending, unique
    std::uint64_t off_vectors;    // float32[count * dims], row i belongs to ids[i]
    std::uint64_t off_graph;      // HNSW section, FLAG_HNSW only
    std::uint64_t graph_bytes;
    std::uint64_t reserved[5];
};
static_assert(sizeof(SnapshotHeader) == 128, "snapshot header must be 128 bytes");

// A snapshot opened for reading. Points into bytes owned by someone else.
struct VectorSet {
    std::uint64_t count = 0;
    std::uint32_t dims = 0;
    Metric metric = Metric::L2;
    std::int64_t max_ver = 0;
    bool has_graph = false;
    const std::int64_t *ids = nullptr;
    const float *vectors = nullptr;
    const float *vector(std::uint64_t i) const { return vectors + i * dims; }
};

// Owning, 8-byte aligned snapshot bytes.
class SnapshotBuffer {
public:
    void allocate(std::uint64_t bytes) { words_.assign(bytes / 8, 0); }
    std::uint8_t *data() { return reinterpret_cast<std::uint8_t *>(words_.data()); }
    const std::uint8_t *data() const { return reinterpret_cast<const std::uint8_t *>(words_.data()); }
    std::uint64_t size() const { return words_.size() * 8; }
private:
    std::vector<std::uint64_t> words_;
};

// Fills the section offsets and total_bytes of h from its counts and flags.
void snapshot_layout(SnapshotHeader &h);

// Checksum of a complete snapshot. The stored checksum field counts as 0.
std::uint64_t snapshot_checksum(const std::uint8_t *data, std::uint64_t size);

// The checksum is the XOR of one value per 8-byte word, mixed with the word's position in the file.
// So the checksum of a file is the XOR of the checksums of its parts, in any order: sections that
// are written by separate statements can be summed up afterwards. A zero word contributes nothing.
// first_word = file offset of data / 8. size is a multiple of 8.
std::uint64_t snapshot_checksum_part(const std::uint8_t *data, std::uint64_t size, std::uint64_t first_word);

// Validates the bytes and returns a view on them. data must be 8-byte aligned
// and must outlive the view. Throws std::runtime_error with the cause.
// verify_checksum reads the whole file: use it in vload, not in every query.
VectorSet snapshot_open(const std::uint8_t *data, std::uint64_t size, bool verify_checksum);

// True if the bytes start with the vvector magic. Used before deleting cache files.
bool snapshot_has_magic(const std::uint8_t *data, std::uint64_t size);

// Collects (id, vector) pairs and writes a snapshot. Vectors are stored as float32.
// Ids must be unique; they may arrive in any order (vbuild gets them ordered by id).
// Memory: 8 + 4 * dims bytes per vector while collecting, twice that in finish when
// the input was not ordered.
class SnapshotBuilder {
public:
    explicit SnapshotBuilder(Metric metric) : metric_(metric) {}
    // All vectors must have the same number of elements, at least 1.
    void add(std::int64_t id, const float *v, std::uint32_t dims);
    std::uint64_t count() const { return ids_.size(); }
    // Writes the snapshot into out. Throws std::runtime_error on a repeated id or no vectors.
    void finish(std::int64_t max_ver, SnapshotBuffer &out);
private:
    Metric metric_;
    std::uint32_t dims_ = 0;
    bool ordered_ = true;
    std::vector<std::int64_t> ids_;
    std::vector<float> values_;
};

} // namespace vvector

#endif
