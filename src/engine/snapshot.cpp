#include "snapshot.h"
#include "version.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace vvector {

Metric parse_metric(const std::string &name)
{
    if (name == "l2") return Metric::L2;
    if (name == "cosine") return Metric::Cosine;
    if (name == "dot") return Metric::Dot;
    throw std::runtime_error("metric must be l2, cosine or dot, not '" + name + "'");
}

const char *metric_name(Metric m)
{
    switch (m) {
    case Metric::L2: return "l2";
    case Metric::Cosine: return "cosine";
    case Metric::Dot: return "dot";
    }
    return "unknown";
}

static std::uint64_t align8(std::uint64_t v) { return (v + 7) & ~std::uint64_t(7); }

void snapshot_layout(SnapshotHeader &h)
{
    std::uint64_t at = sizeof(SnapshotHeader);
    auto place = [&at](std::uint64_t bytes) {
        std::uint64_t off = at;
        at = align8(at + bytes);
        return off;
    };
    h.off_ids = place(h.count * 8);
    h.off_vectors = place(h.count * h.dims * 4);
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

VectorSet snapshot_open(const std::uint8_t *data, std::uint64_t size, bool verify_checksum)
{
    if (size < sizeof(SnapshotHeader)) fail("shorter than the header");
    if (reinterpret_cast<std::uintptr_t>(data) % 8 != 0) fail("buffer is not 8-byte aligned");
    if (!snapshot_has_magic(data, size)) fail("wrong magic");

    SnapshotHeader h;
    std::memcpy(&h, data, sizeof(h));
    if (h.format_version != static_cast<std::uint32_t>(FORMAT_VERSION))
        fail("format version " + std::to_string(h.format_version) + ", this library reads " +
             std::to_string(FORMAT_VERSION));
    if (h.dims == 0) fail("dims is 0");
    if (h.metric < 1 || h.metric > 3) fail("unknown metric " + std::to_string(h.metric));

    // The offsets must be exactly what the layout rule gives for these counts.
    SnapshotHeader expect = h;
    snapshot_layout(expect);
    if (std::memcmp(&expect, &h, sizeof(h)) != 0) fail("section offsets do not match the counts");
    if (h.total_bytes != size) fail("size is " + std::to_string(size) + " bytes, header says " +
                                    std::to_string(h.total_bytes));
    if (verify_checksum && snapshot_checksum(data, size) != h.checksum) fail("checksum mismatch");

    VectorSet s;
    s.count = h.count;
    s.dims = h.dims;
    s.metric = static_cast<Metric>(h.metric);
    s.max_ver = h.max_ver;
    s.has_graph = (h.flags & FLAG_HNSW) != 0;
    s.ids = reinterpret_cast<const std::int64_t *>(data + h.off_ids);
    s.vectors = reinterpret_cast<const float *>(data + h.off_vectors);
    return s;
}

void SnapshotBuilder::add(std::int64_t id, const float *v, std::uint32_t dims)
{
    if (dims == 0) throw std::runtime_error("vector of id " + std::to_string(id) + " has no elements");
    if (dims_ == 0) dims_ = dims;
    else if (dims != dims_)
        throw std::runtime_error("vector of id " + std::to_string(id) + " has " + std::to_string(dims) +
                                 " elements, the ones before have " + std::to_string(dims_));
    if (!ids_.empty() && id <= ids_.back()) ordered_ = false;
    ids_.push_back(id);
    values_.insert(values_.end(), v, v + dims);
}

void SnapshotBuilder::finish(std::int64_t max_ver, SnapshotBuffer &out)
{
    if (ids_.empty()) throw std::runtime_error("no vectors");
    const std::uint64_t n = ids_.size();

    // Positions in id order. Input ordered by id (the usual case) needs no sort.
    std::vector<std::uint64_t> order;
    if (!ordered_) {
        order.resize(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [this](std::uint64_t a, std::uint64_t b) { return ids_[a] < ids_[b]; });
    }
    auto at = [&](std::uint64_t i) { return ordered_ ? i : order[i]; };
    for (std::uint64_t i = 1; i < n; ++i)
        if (ids_[at(i)] == ids_[at(i - 1)])
            throw std::runtime_error("id " + std::to_string(ids_[at(i)]) + " appears twice");

    SnapshotHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, SNAPSHOT_MAGIC, sizeof(h.magic));
    h.format_version = FORMAT_VERSION;
    h.count = n;
    h.dims = dims_;
    h.metric = static_cast<std::uint32_t>(metric_);
    h.max_ver = max_ver;
    snapshot_layout(h);

    out.allocate(h.total_bytes);
    std::uint8_t *base = out.data();
    std::int64_t *ids = reinterpret_cast<std::int64_t *>(base + h.off_ids);
    float *vectors = reinterpret_cast<float *>(base + h.off_vectors);
    for (std::uint64_t i = 0; i < n; ++i) {
        ids[i] = ids_[at(i)];
        std::memcpy(vectors + i * dims_, values_.data() + at(i) * dims_, dims_ * sizeof(float));
    }
    std::memcpy(base, &h, sizeof(h));
    h.checksum = snapshot_checksum(base, h.total_bytes);
    std::memcpy(base + offsetof(SnapshotHeader, checksum), &h.checksum, sizeof(h.checksum));
}

} // namespace vvector
