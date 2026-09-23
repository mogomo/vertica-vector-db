#include "snapshot.h"
#include "kernels.h"
#include "version.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace vvector {

Metric parse_metric(const std::string &name)
{
    if (name == "l2") return Metric::L2;
    if (name == "cosine") return Metric::Cosine;
    if (name == "dot") return Metric::Dot;
    if (name == "l1") return Metric::L1;
    throw std::runtime_error("metric must be l2, cosine, dot or l1, not '" + name + "'");
}

const char *metric_name(Metric m)
{
    switch (m) {
    case Metric::L2: return "l2";
    case Metric::Cosine: return "cosine";
    case Metric::Dot: return "dot";
    case Metric::L1: return "l1";
    }
    return "unknown";
}

std::int64_t VectorSet::find(std::int64_t id) const
{
    std::uint64_t lo = 0, hi = count;
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        const std::int64_t at = ids[id_index ? id_index[mid] : mid];
        if (at < id) lo = mid + 1;
        else hi = mid;
    }
    if (lo == count) return -1;
    // With an id_index the first entry of an id is its highest position: the only one that can be live.
    const std::uint64_t pos = id_index ? id_index[lo] : lo;
    return ids[pos] == id && !dead(pos) ? static_cast<std::int64_t>(pos) : -1;
}

// ---- SnapshotBuffer

namespace {
constexpr std::uint64_t PAGE = 4096;
std::uint64_t round_page(std::uint64_t n) { return (n + PAGE - 1) / PAGE * PAGE; }
}

SnapshotBuffer::~SnapshotBuffer() { release(); }

void SnapshotBuffer::release()
{
    if (!data_) return;
#if defined(__linux__)
    munmap(data_, capacity_);
#else
    std::free(data_);
#endif
    data_ = nullptr;
    size_ = capacity_ = 0;
}

void SnapshotBuffer::allocate(std::uint64_t bytes)
{
    release();
    reserve(bytes);
    size_ = bytes;
}

void SnapshotBuffer::reserve(std::uint64_t bytes)
{
    if (bytes <= capacity_) return;
    const std::uint64_t want = round_page(bytes);
#if defined(__linux__)
    void *p = data_ ? mremap(data_, capacity_, want, MREMAP_MAYMOVE)
                    : mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        throw std::runtime_error("out of memory: cannot map " + std::to_string(want >> 20) + " MB for the snapshot");
#else
    void *p = std::aligned_alloc(PAGE, want);
    if (!p) throw std::runtime_error("out of memory: cannot allocate " + std::to_string(want >> 20) + " MB for the snapshot");
    std::memset(p, 0, want);
    if (data_) std::memcpy(p, data_, size_);
    std::free(data_);
#endif
    data_ = static_cast<std::uint8_t *>(p);
    capacity_ = want;
}

void SnapshotBuffer::swap(SnapshotBuffer &other) noexcept
{
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
}

void SnapshotBuffer::set_size(std::uint64_t bytes)
{
    if (bytes > capacity_) throw std::logic_error("SnapshotBuffer::set_size beyond the reserved capacity");
    size_ = bytes;
}

// ---- layout, checksum, open

static std::uint64_t align_section(std::uint64_t v) { return (v + SECTION_ALIGN - 1) / SECTION_ALIGN * SECTION_ALIGN; }

void snapshot_layout(SnapshotHeader &h)
{
    std::uint64_t at = HEADER_BYTES;
    auto place = [&at](std::uint64_t bytes) {
        const std::uint64_t off = at;
        at = align_section(at + bytes);
        return off;
    };
    h.off_vectors = place(h.count * h.row_stride * 4);
    h.off_ids = place(h.count * 8);
    h.off_id_index = (h.flags & FLAG_ID_INDEX) ? place(h.count * 4) : 0;
    h.off_tombstones = (h.flags & FLAG_TOMBSTONES) ? place((h.count + 63) / 64 * 8) : 0;
    h.off_sq8 = (h.flags & FLAG_SQ8) ? place(h.sq8_bytes) : 0;
    h.off_graph = (h.flags & FLAG_HNSW) ? place(h.graph_bytes) : 0;
    h.total_bytes = at;
}

std::uint64_t snapshot_checksum_part(const std::uint8_t *data, std::uint64_t size, std::uint64_t first_word)
{
    std::uint64_t sum = 0;
    for (std::uint64_t i = 0; i < size / 8; ++i) {
        std::uint64_t w;
        std::memcpy(&w, data + i * 8, 8);
        if (w == 0) continue;
        // splitmix64 finalizer over the word and its position
        std::uint64_t z = w + (first_word + i + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        sum ^= z ^ (z >> 31);
    }
    return sum;
}

std::uint64_t snapshot_checksum(const std::uint8_t *data, std::uint64_t size)
{
    // Everything except the checksum field of the header.
    const std::uint64_t at = offsetof(SnapshotHeader, checksum);
    return snapshot_checksum_part(data, at, 0) ^
           snapshot_checksum_part(data + at + 8, size - at - 8, at / 8 + 1);
}

bool snapshot_has_magic(const std::uint8_t *data, std::uint64_t size)
{
    return size >= sizeof(SNAPSHOT_MAGIC) && std::memcmp(data, SNAPSHOT_MAGIC, sizeof(SNAPSHOT_MAGIC)) == 0;
}

static void fail(const std::string &why) { throw std::runtime_error("bad snapshot: " + why); }

VectorSet snapshot_open(const std::uint8_t *data, std::uint64_t size, bool verify)
{
    if (size < 16) fail("shorter than the header");
    if (reinterpret_cast<std::uintptr_t>(data) % SECTION_ALIGN != 0) fail("buffer is not 64-byte aligned");
    if (!snapshot_has_magic(data, size)) fail("wrong magic");
    std::uint32_t version;
    std::memcpy(&version, data + offsetof(SnapshotHeader, format_version), 4);
    if (version != static_cast<std::uint32_t>(FORMAT_VERSION))
        fail("format version " + std::to_string(version) + ", this library reads version " +
             std::to_string(FORMAT_VERSION) + ": refresh the index to rebuild its snapshot");
    if (size < HEADER_BYTES) fail("shorter than the header");

    SnapshotHeader h;
    std::memcpy(&h, data, sizeof(h));
    if (h.flags & ~KNOWN_FLAGS) fail("unknown flags " + std::to_string(h.flags));
    if (h.dims == 0 || h.dims > MAX_DIMS) fail("dims " + std::to_string(h.dims) + " out of range");
    if (h.row_stride != row_stride_for(h.dims)) fail("row_stride does not match dims");
    if (h.metric < 1 || h.metric > 4) fail("unknown metric " + std::to_string(h.metric));
    if (h.count > MAX_COUNT) fail("more than 4,294,967,295 vectors");
    if (((h.flags & FLAG_NORMALISED) != 0) != (h.metric == static_cast<std::uint32_t>(Metric::Cosine)))
        fail("cosine indexes and only they must be normalised");
    if (!(h.flags & FLAG_SQ8) && h.sq8_bytes) fail("sq8_bytes without the sq8 section");
    if (!(h.flags & FLAG_HNSW) && h.graph_bytes) fail("graph_bytes without the graph section");
    if (!(h.flags & FLAG_TOMBSTONES) && h.tombstones) fail("tombstones without the tombstones section");
    if (h.tombstones > h.count) fail("more tombstones than vectors");

    // The offsets must be exactly what the layout rule gives for these counts.
    SnapshotHeader expect = h;
    snapshot_layout(expect);
    if (std::memcmp(&expect, &h, sizeof(h)) != 0) fail("section offsets do not match the counts");
    if (h.total_bytes != size)
        fail("size is " + std::to_string(size) + " bytes, header says " + std::to_string(h.total_bytes));

    VectorSet s;
    s.count = h.count;
    s.dims = h.dims;
    s.row_stride = h.row_stride;
    s.metric = static_cast<Metric>(h.metric);
    s.flags = h.flags;
    s.max_ver = h.max_ver;
    s.base_snapshot = h.base_snapshot;
    s.tombstones = h.tombstones;
    s.vectors = reinterpret_cast<const float *>(data + h.off_vectors);
    s.ids = reinterpret_cast<const std::int64_t *>(data + h.off_ids);
    if (h.off_id_index) s.id_index = reinterpret_cast<const std::uint32_t *>(data + h.off_id_index);
    if (h.off_tombstones) s.tombstone_bits = reinterpret_cast<const std::uint64_t *>(data + h.off_tombstones);
    if (h.off_sq8) { s.sq8 = data + h.off_sq8; s.sq8_bytes = h.sq8_bytes; }
    if (h.off_graph) { s.graph = data + h.off_graph; s.graph_bytes = h.graph_bytes; }

    if (verify) {
        if (snapshot_checksum(data, size) != h.checksum) fail("checksum mismatch");
        if (s.tombstone_bits) {
            std::uint64_t dead = 0;
            for (std::uint64_t w = 0; w < (s.count + 63) / 64; ++w) dead += __builtin_popcountll(s.tombstone_bits[w]);
            if (dead != s.tombstones) fail("tombstone count does not match the bitset");
            if (s.count % 64 && s.tombstone_bits[s.count / 64] >> (s.count % 64)) fail("tombstone bits beyond the vectors");
        }
        if (!s.id_index) {
            for (std::uint64_t i = 1; i < s.count; ++i)
                if (s.ids[i - 1] >= s.ids[i]) fail("ids are not unique and ascending");
        } else {
            // Every position once; ids ascending; an id at several positions (an incremental build
            // appends a changed vector and tombstones the old position): highest position first,
            // and only that one may be live.
            std::vector<std::uint64_t> listed((s.count + 63) / 64, 0);
            for (std::uint64_t i = 0; i < s.count; ++i) {
                const std::uint32_t p = s.id_index[i];
                if (p >= s.count) fail("id_index points outside the vectors");
                if (listed[p >> 6] >> (p & 63) & 1u) fail("id_index lists a position twice");
                listed[p >> 6] |= 1ull << (p & 63);
                if (i == 0) continue;
                const std::uint32_t q = s.id_index[i - 1];
                if (s.ids[q] > s.ids[p]) fail("id_index is not sorted by id");
                if (s.ids[q] == s.ids[p] && (q < p || !s.dead(p))) fail("an id is live at two positions");
            }
        }
    }
    return s;
}

// ---- SnapshotBuilder

float *SnapshotBuilder::begin_row(std::int64_t id, std::uint32_t dims)
{
    if (open_row_) throw std::logic_error("SnapshotBuilder::begin_row without end_row");
    if (dims == 0) throw std::runtime_error("vector of id " + std::to_string(id) + " has no elements");
    if (dims_ == 0) {
        if (dims > MAX_DIMS)
            throw std::runtime_error("vectors have " + std::to_string(dims) + " elements, at most " +
                                     std::to_string(MAX_DIMS) + " are supported");
        dims_ = dims;
        stride_ = row_stride_for(dims);
    } else if (dims != dims_) {
        throw std::runtime_error("vector of id " + std::to_string(id) + " has " + std::to_string(dims) +
                                 " elements, the ones before have " + std::to_string(dims_));
    }
    const std::uint64_t n = ids_.size();
    if (n == MAX_COUNT) throw std::runtime_error("more than 4,294,967,295 vectors: that is the limit of one index");
    if (!ids_.empty() && id <= ids_.back()) ordered_ = false;
    ids_.push_back(id);

    const std::uint64_t row_bytes = std::uint64_t(stride_) * 4;
    const std::uint64_t end = HEADER_BYTES + (n + 1) * row_bytes;
    if (end > buffer_.capacity()) buffer_.reserve(std::max<std::uint64_t>(end, buffer_.capacity() / 2 * 3 + (1u << 20)));
    buffer_.set_size(end);
    open_row_ = true;
    // Fresh buffer memory is zero, so the padding after dims is already 0.
    return reinterpret_cast<float *>(buffer_.data() + HEADER_BYTES + n * row_bytes);
}

void SnapshotBuilder::end_row()
{
    if (!open_row_) throw std::logic_error("SnapshotBuilder::end_row without begin_row");
    open_row_ = false;
    if (metric_ == Metric::Cosine) {
        float *row = reinterpret_cast<float *>(buffer_.data() + HEADER_BYTES) + (ids_.size() - 1) * stride_;
        normalize(row, dims_);
    }
}

void SnapshotBuilder::add(std::int64_t id, const float *v, std::uint32_t dims)
{
    float *row = begin_row(id, dims);
    std::memcpy(row, v, std::size_t(dims) * sizeof(float));
    end_row();
}

void SnapshotBuilder::finish(std::int64_t max_ver, SnapshotBuffer &out, const GraphSection *graph)
{
    if (open_row_) throw std::logic_error("SnapshotBuilder::finish with an open row");
    if (ids_.empty()) throw std::runtime_error("no vectors");
    const std::uint64_t n = ids_.size();
    float *rows = reinterpret_cast<float *>(buffer_.data() + HEADER_BYTES);

    if (!ordered_) {
        // order[i] = the arrival position of the row that belongs at position i.
        std::vector<std::uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [this](std::uint32_t a, std::uint32_t b) { return ids_[a] < ids_[b]; });
        for (std::uint64_t i = 1; i < n; ++i)
            if (ids_[order[i]] == ids_[order[i - 1]])
                throw std::runtime_error("id " + std::to_string(ids_[order[i]]) + " appears twice");
        // Move the rows in place, one cycle of the permutation at a time, with one spare row.
        // A position is done when order[i] == i; finished positions are marked that way.
        std::vector<float> spare(stride_);
        const std::size_t row_bytes = std::size_t(stride_) * 4;
        for (std::uint64_t start = 0; start < n; ++start) {
            if (order[start] == start) continue;
            std::memcpy(spare.data(), rows + start * stride_, row_bytes);
            std::uint64_t at = start;
            for (;;) {
                const std::uint64_t from = order[at];
                order[at] = static_cast<std::uint32_t>(at);
                if (from == start) { std::memcpy(rows + at * stride_, spare.data(), row_bytes); break; }
                std::memcpy(rows + at * stride_, rows + from * stride_, row_bytes);
                at = from;
            }
        }
        std::sort(ids_.begin(), ids_.end());
    }

    SnapshotHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, SNAPSHOT_MAGIC, sizeof(h.magic));
    h.format_version = FORMAT_VERSION;
    h.flags = metric_ == Metric::Cosine ? FLAG_NORMALISED : 0;
    if (graph) {
        h.flags |= FLAG_HNSW;
        h.graph_bytes = graph->bytes(ids_.data(), n);
    }
    h.count = n;
    h.dims = dims_;
    h.row_stride = stride_;
    h.metric = static_cast<std::uint32_t>(metric_);
    h.max_ver = max_ver;
    snapshot_layout(h);

    buffer_.reserve(h.total_bytes);
    buffer_.set_size(h.total_bytes);
    std::uint8_t *base = buffer_.data();
    std::memcpy(base + h.off_ids, ids_.data(), n * 8);
    std::memcpy(base, &h, sizeof(h));
    if (graph) graph->fill(snapshot_open(base, h.total_bytes, false), base + h.off_graph);
    h.checksum = snapshot_checksum(base, h.total_bytes);
    std::memcpy(base + offsetof(SnapshotHeader, checksum), &h.checksum, sizeof(h.checksum));

    out.swap(buffer_);
    buffer_.clear();
    std::vector<std::int64_t>().swap(ids_);
    dims_ = stride_ = 0;
    ordered_ = true;
}

} // namespace vvector
