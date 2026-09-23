#include "delta.h"
#include "kernels.h"
#include "version.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace vvector {

IncrementalBuilder::IncrementalBuilder(const VectorSet &base) : base_(base)
{
    if (base.count == 0) throw std::runtime_error("the base snapshot has no vectors");
}

float *IncrementalBuilder::begin_add(std::int64_t id, std::uint32_t dims)
{
    if (open_row_) throw std::logic_error("IncrementalBuilder::begin_add without end_add");
    if (dims != base_.dims)
        throw std::runtime_error("vector of id " + std::to_string(id) + " has " + std::to_string(dims) +
                                 " elements, the index has " + std::to_string(base_.dims));
    const std::uint64_t n = adds_.size();
    const std::uint64_t row_bytes = std::uint64_t(base_.row_stride) * 4;
    const std::uint64_t end = (n + 1) * row_bytes;
    if (end > staged_.capacity()) staged_.reserve(std::max<std::uint64_t>(end, staged_.capacity() / 2 * 3 + (1u << 20)));
    staged_.set_size(end);
    adds_.push_back(id);
    open_row_ = true;
    // Fresh buffer memory is zero, so the padding after dims is already 0.
    return reinterpret_cast<float *>(staged_.data() + n * row_bytes);
}

void IncrementalBuilder::end_add()
{
    if (!open_row_) throw std::logic_error("IncrementalBuilder::end_add without begin_add");
    open_row_ = false;
    if (base_.metric == Metric::Cosine)
        normalize(reinterpret_cast<float *>(staged_.data()) + (adds_.size() - 1) * base_.row_stride, base_.dims);
}

void IncrementalBuilder::remove(std::int64_t id)
{
    if (open_row_) throw std::logic_error("IncrementalBuilder::remove with an open row");
    removes_.push_back(id);
}

bool IncrementalBuilder::finish(std::int64_t max_ver, std::int64_t base_snapshot, SnapshotBuffer &out,
                                const HnswParams *graph, const std::function<bool()> &poll)
{
    if (open_row_) throw std::logic_error("IncrementalBuilder::finish with an open row");
    const VectorSet &b = base_;
    const std::uint64_t n0 = b.count;
    const std::uint32_t stride = b.row_stride;
    const float *staged = reinterpret_cast<const float *>(staged_.data());
    stats_ = DeltaStats();

    if (b.flags & FLAG_SQ8) throw std::runtime_error("the base snapshot has int8 codes: not supported yet (milestone M4)");
    HnswGraph bg;
    if (b.has_graph()) {
        if (!graph) throw std::runtime_error("the base snapshot is an HNSW index, the build is flat: a full build is needed");
        bg = hnsw_open(b, false);
        if (graph->m != bg.m)
            throw std::runtime_error("m is " + std::to_string(graph->m) + ", the base snapshot was built with m " +
                                     std::to_string(bg.m) + ": a full build is needed");
    } else if (graph) {
        throw std::runtime_error("the base snapshot is a flat index, the build is hnsw: a full build is needed");
    }

    // Changes in id order. An id given twice is an error, as in a full build.
    std::vector<std::uint32_t> order(adds_.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [this](std::uint32_t x, std::uint32_t y) { return adds_[x] < adds_[y]; });
    std::sort(removes_.begin(), removes_.end());
    auto twice = [](std::int64_t id) { return std::runtime_error("id " + std::to_string(id) + " appears twice"); };
    for (std::size_t i = 1; i < order.size(); ++i)
        if (adds_[order[i]] == adds_[order[i - 1]]) throw twice(adds_[order[i]]);
    for (std::size_t i = 1; i < removes_.size(); ++i)
        if (removes_[i] == removes_[i - 1]) throw twice(removes_[i]);
    for (std::size_t i = 0, j = 0; i < order.size() && j < removes_.size();) {
        const std::int64_t a = adds_[order[i]], r = removes_[j];
        if (a == r) throw twice(a);
        if (a < r) ++i; else ++j;
    }

    // Tombstones: the base's, and the positions of deleted and changed ids.
    std::vector<std::uint64_t> dead((n0 + 63) / 64, 0);
    if (b.tombstone_bits) std::memcpy(dead.data(), b.tombstone_bits, dead.size() * 8);
    auto kill = [&dead](std::int64_t pos) { dead[pos >> 6] |= 1ull << (pos & 63); };
    for (const std::int64_t id : removes_) {
        const std::int64_t pos = b.find(id);
        if (pos < 0) { ++stats_.absent; continue; }
        kill(pos);
        ++stats_.tombstoned;
    }
    std::vector<std::uint32_t> keep;          // staged rows to append, in id order
    std::vector<std::int64_t> new_ids;
    for (const std::uint32_t i : order) {
        const std::int64_t pos = b.find(adds_[i]);
        if (pos >= 0) {
            if (std::memcmp(b.vector(pos), staged + std::uint64_t(i) * stride, std::size_t(stride) * 4) == 0) {
                ++stats_.unchanged;
                continue;
            }
            kill(pos);
            ++stats_.tombstoned;
        }
        keep.push_back(i);
        new_ids.push_back(adds_[i]);
    }
    stats_.appended = keep.size();
    if (stats_.tombstoned == 0 && keep.empty()) return false;
    const std::uint64_t n = n0 + keep.size();
    if (n > MAX_COUNT) throw std::runtime_error("more than 4,294,967,295 vectors: that is the limit of one index");

    // An id_index is needed once an appended id is not larger than every id before it.
    const std::int64_t base_max_id = b.ids[b.id_index ? b.id_index[n0 - 1] : n0 - 1];
    const bool with_id_index = b.id_index || (!new_ids.empty() && new_ids.front() <= base_max_id);
    const std::uint64_t tombstones = b.tombstones + stats_.tombstoned;

    SnapshotHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, SNAPSHOT_MAGIC, sizeof(h.magic));
    h.format_version = FORMAT_VERSION;
    h.flags = (b.flags & (FLAG_NORMALISED | FLAG_HNSW)) | (with_id_index ? FLAG_ID_INDEX : 0) |
              (tombstones ? FLAG_TOMBSTONES : 0);
    h.count = n;
    h.dims = b.dims;
    h.row_stride = stride;
    h.metric = static_cast<std::uint32_t>(b.metric);
    h.max_ver = max_ver;
    h.base_snapshot = base_snapshot;
    h.tombstones = tombstones;
    if (graph) h.graph_bytes = hnsw_extended_bytes(bg, new_ids.data(), new_ids.size());
    snapshot_layout(h);

    SnapshotBuffer buf;
    buf.allocate(h.total_bytes);
    std::uint8_t *base = buf.data();
    const std::size_t row_bytes = std::size_t(stride) * 4;
    std::memcpy(base + h.off_vectors, b.vectors, n0 * row_bytes);
    for (std::size_t j = 0; j < keep.size(); ++j)
        std::memcpy(base + h.off_vectors + (n0 + j) * row_bytes, staged + std::uint64_t(keep[j]) * stride, row_bytes);
    std::memcpy(base + h.off_ids, b.ids, n0 * 8);
    if (!new_ids.empty()) std::memcpy(base + h.off_ids + n0 * 8, new_ids.data(), new_ids.size() * 8);
    if (with_id_index) {
        // Positions by id; an id at several positions: highest first (VectorSet::find). Appended
        // positions are higher than every base position, so on an equal id they go first.
        std::uint32_t *ix = reinterpret_cast<std::uint32_t *>(base + h.off_id_index);
        std::uint64_t i = 0, j = 0, k = 0;
        auto base_pos = [&b](std::uint64_t i) { return b.id_index ? b.id_index[i] : static_cast<std::uint32_t>(i); };
        while (i < n0 || j < new_ids.size()) {
            if (j < new_ids.size() && (i == n0 || new_ids[j] <= b.ids[base_pos(i)])) ix[k++] = static_cast<std::uint32_t>(n0 + j++);
            else ix[k++] = base_pos(i++);
        }
    }
    if (tombstones) std::memcpy(base + h.off_tombstones, dead.data(), dead.size() * 8);
    std::memcpy(base, &h, sizeof(h));
    if (graph) hnsw_extend(bg, snapshot_open(base, h.total_bytes, false), base + h.off_graph, *graph, poll);
    h.checksum = snapshot_checksum(base, h.total_bytes);
    std::memcpy(base + offsetof(SnapshotHeader, checksum), &h.checksum, sizeof(h.checksum));
    out.swap(buf);
    return true;
}

} // namespace vvector
