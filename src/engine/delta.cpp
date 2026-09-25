#include "delta.h"
#include "kernels.h"
#include "sq8.h"
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

void IncrementalBuilder::set_growth(std::uint32_t percent)
{
    if (percent > MAX_GROWTH_PERCENT) throw std::runtime_error("growth must be 0 to " + std::to_string(MAX_GROWTH_PERCENT) + " percent");
    growth_ = percent;
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

// What the changes do to the base: decided once, applied by finish or finish_in_place.
struct IncrementalBuilder::Plan {
    std::vector<std::uint32_t> keep;          // staged rows to append, in id order
    std::vector<std::int64_t> new_ids;        // their ids
    std::vector<std::uint64_t> dead;          // tombstone bits of the base's positions, after the changes
    std::uint64_t n = 0;                      // positions of the new snapshot
    std::uint64_t tombstones = 0;
    bool with_id_index = false;
    bool codes = false;
    Sq8Codes bc;
};

bool IncrementalBuilder::prepare(Plan &pl, const HnswParams *graph, HnswGraph &bg)
{
    if (open_row_) throw std::logic_error("IncrementalBuilder::finish with an open row");
    const VectorSet &b = base_;
    const std::uint64_t n0 = b.count;
    const std::uint32_t stride = b.row_stride;
    const float *staged = reinterpret_cast<const float *>(staged_.data());
    stats_ = DeltaStats();

    pl.codes = (b.flags & FLAG_SQ8) != 0;
    if (pl.codes) pl.bc = sq8_open(b, false);
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
    pl.dead.assign((n0 + 63) / 64, 0);
    if (b.tombstone_bits) std::memcpy(pl.dead.data(), b.tombstone_bits, pl.dead.size() * 8);
    auto kill = [&pl](std::int64_t pos) { pl.dead[pos >> 6] |= 1ull << (pos & 63); };
    for (const std::int64_t id : removes_) {
        const std::int64_t pos = b.find(id);
        if (pos < 0) { ++stats_.absent; continue; }
        kill(pos);
        ++stats_.tombstoned;
    }
    pl.keep.clear();
    pl.new_ids.clear();
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
        pl.keep.push_back(i);
        pl.new_ids.push_back(adds_[i]);
    }
    stats_.appended = pl.keep.size();
    if (stats_.tombstoned == 0 && pl.keep.empty()) return false;
    pl.n = n0 + pl.keep.size();
    if (pl.n > MAX_COUNT) throw std::runtime_error("more than 4,294,967,295 vectors: that is the limit of one index");

    // An id_index is needed once an appended id is not larger than every id before it.
    const std::int64_t base_max_id = b.ids[b.id_index ? b.id_index[n0 - 1] : n0 - 1];
    pl.with_id_index = b.id_index || (!pl.new_ids.empty() && pl.new_ids.front() <= base_max_id);
    pl.tombstones = b.tombstones + stats_.tombstoned;
    return true;
}

namespace {

// Positions by id for the new snapshot; an id at several positions: highest first (VectorSet::find).
// Appended positions are higher than every base position, so on an equal id they go first.
void merge_id_index(const VectorSet &b, const std::vector<std::int64_t> &new_ids, std::uint32_t *ix)
{
    const std::uint64_t n0 = b.count;
    std::uint64_t i = 0, j = 0, k = 0;
    auto base_pos = [&b](std::uint64_t i) { return b.id_index ? b.id_index[i] : static_cast<std::uint32_t>(i); };
    while (i < n0 || j < new_ids.size()) {
        if (j < new_ids.size() && (i == n0 || new_ids[j] <= b.ids[base_pos(i)])) ix[k++] = static_cast<std::uint32_t>(n0 + j++);
        else ix[k++] = base_pos(i++);
    }
}

// True when the base's layout has room for the new snapshot: then it is kept.
bool layout_fits(const VectorSet &b, const HnswGraph &bg, const HnswParams *graph, const std::vector<std::int64_t> &new_ids,
                 std::uint64_t n)
{
    if (!b.has_capacity() || n > b.capacity) return false;
    return !graph || hnsw_fits_in_place(bg, new_ids.data(), new_ids.size(), b.capacity);
}

} // namespace

bool IncrementalBuilder::finish(std::int64_t max_ver, std::int64_t base_snapshot, SnapshotBuffer &out,
                                const HnswParams *graph, const std::function<bool()> &poll)
{
    Plan pl;
    HnswGraph bg;
    if (!prepare(pl, graph, bg)) return false;
    const VectorSet &b = base_;
    const std::uint64_t n0 = b.count, n = pl.n;
    const std::uint32_t stride = b.row_stride;
    const float *staged = reinterpret_cast<const float *>(staged_.data());

    SnapshotHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, SNAPSHOT_MAGIC, sizeof(h.magic));
    h.format_version = FORMAT_VERSION;
    h.flags = (b.flags & (FLAG_NORMALISED | FLAG_HNSW | FLAG_SQ8)) | (pl.with_id_index ? FLAG_ID_INDEX : 0) |
              (pl.tombstones ? FLAG_TOMBSTONES : 0);
    // The base's layout when it has room (the same bytes as a build in place), else a new one with
    // room to grow.
    std::uint64_t cap;
    if (layout_fits(b, bg, graph, pl.new_ids, n)) {
        cap = b.capacity;
        h.flags |= FLAG_CAPACITY;
        h.capacity = cap;
    } else {
        const std::uint64_t slack = growth_rows(n, growth_);
        cap = std::min<std::uint64_t>(n + slack, MAX_COUNT);
        if (slack) {
            h.flags |= FLAG_CAPACITY;
            h.capacity = cap;
        }
    }
    h.count = n;
    h.dims = b.dims;
    h.row_stride = stride;
    h.metric = static_cast<std::uint32_t>(b.metric);
    h.max_ver = max_ver;
    h.base_snapshot = base_snapshot;
    h.tombstones = pl.tombstones;
    if (graph) h.graph_bytes = hnsw_extended_bytes(bg, pl.new_ids.data(), pl.new_ids.size(), cap);
    if (pl.codes) h.sq8_bytes = sq8_section_bytes(cap, stride);
    snapshot_layout(h);

    SnapshotBuffer buf;
    buf.swap(out);                  // takes over out's file backing, if it has one (build_in='file')
    buf.allocate(h.total_bytes);
    std::uint8_t *base = buf.data();
    const std::size_t row_bytes = std::size_t(stride) * 4;
    std::memcpy(base + h.off_vectors, b.vectors, n0 * row_bytes);
    for (std::size_t j = 0; j < pl.keep.size(); ++j)
        std::memcpy(base + h.off_vectors + (n0 + j) * row_bytes, staged + std::uint64_t(pl.keep[j]) * stride, row_bytes);
    std::memcpy(base + h.off_ids, b.ids, n0 * 8);
    if (!pl.new_ids.empty()) std::memcpy(base + h.off_ids + n0 * 8, pl.new_ids.data(), pl.new_ids.size() * 8);
    if (pl.with_id_index) merge_id_index(b, pl.new_ids, reinterpret_cast<std::uint32_t *>(base + h.off_id_index));
    if (pl.tombstones) std::memcpy(base + h.off_tombstones, pl.dead.data(), pl.dead.size() * 8);
    std::memcpy(base, &h, sizeof(h));
    if (pl.codes) sq8_extend(pl.bc, snapshot_open(base, h.total_bytes, false), base + h.off_sq8);
    if (graph) hnsw_extend(bg, snapshot_open(base, h.total_bytes, false), base + h.off_graph, *graph, poll);
    h.checksum = snapshot_checksum(base, h.total_bytes);
    std::memcpy(base + offsetof(SnapshotHeader, checksum), &h.checksum, sizeof(h.checksum));
    out.swap(buf);
    return true;
}

InPlace IncrementalBuilder::finish_in_place(std::int64_t max_ver, std::int64_t base_snapshot, std::uint8_t *copy,
                                            std::uint64_t size, const HnswParams *graph, std::vector<ByteRange> &candidates,
                                            const std::function<bool()> &poll)
{
    Plan pl;
    HnswGraph bg;
    candidates.clear();
    if (!prepare(pl, graph, bg)) return InPlace::Unchanged;
    const VectorSet &b = base_;
    if (!layout_fits(b, bg, graph, pl.new_ids, pl.n)) return InPlace::NoRoom;
    const std::uint64_t n0 = b.count, n = pl.n;
    const std::uint32_t stride = b.row_stride;
    const float *staged = reinterpret_cast<const float *>(staged_.data());

    // The base's header with the new counts; every offset stays (checked).
    SnapshotHeader h;
    std::memcpy(&h, copy, sizeof(h));
    if (h.total_bytes != size) throw std::logic_error("finish_in_place: the copy is not the base's file");
    h.count = n;
    h.flags |= (pl.with_id_index ? FLAG_ID_INDEX : 0) | (pl.tombstones ? FLAG_TOMBSTONES : 0);
    h.max_ver = max_ver;
    h.base_snapshot = base_snapshot;
    h.tombstones = pl.tombstones;
    {
        SnapshotHeader expect = h;
        snapshot_layout(expect);
        if (std::memcmp(&expect, &h, sizeof(h)) != 0) throw std::logic_error("finish_in_place: the layout moved");
    }
    auto touched = [&candidates](std::uint64_t offset, std::uint64_t bytes) {
        if (bytes) candidates.push_back(ByteRange{offset, bytes});
    };
    const std::size_t row_bytes = std::size_t(stride) * 4;
    touched(0, HEADER_BYTES);
    for (std::size_t j = 0; j < pl.keep.size(); ++j)
        std::memcpy(copy + h.off_vectors + (n0 + j) * row_bytes, staged + std::uint64_t(pl.keep[j]) * stride, row_bytes);
    touched(h.off_vectors + n0 * row_bytes, pl.keep.size() * row_bytes);
    if (!pl.new_ids.empty()) std::memcpy(copy + h.off_ids + n0 * 8, pl.new_ids.data(), pl.new_ids.size() * 8);
    touched(h.off_ids + n0 * 8, pl.new_ids.size() * 8);
    if (pl.with_id_index) {
        // Rebuilt whole (positions shift from the first appended id on); the diff sends what changed.
        std::vector<std::uint32_t> ix(n);
        merge_id_index(b, pl.new_ids, ix.data());
        const std::uint32_t *old = b.id_index;
        std::uint32_t *at = reinterpret_cast<std::uint32_t *>(copy + h.off_id_index);
        // Write only the entries that differ: an untouched page of the copy costs no memory.
        for (std::uint64_t i = 0; i < n; ++i)
            if (!old || i >= n0 || old[i] != ix[i]) at[i] = ix[i];
        touched(h.off_id_index, n * 4);
    }
    if (pl.tombstones) {
        const std::uint64_t *old = b.tombstone_bits;
        std::uint64_t *at = reinterpret_cast<std::uint64_t *>(copy + h.off_tombstones);
        for (std::uint64_t w = 0; w < pl.dead.size(); ++w)
            if (!old || old[w] != pl.dead[w]) at[w] = pl.dead[w];
        touched(h.off_tombstones, (n + 63) / 64 * 8);
    }
    std::memcpy(copy, &h, sizeof(h));
    const VectorSet s = snapshot_open(copy, size, false);
    if (pl.codes) {
        sq8_extend(pl.bc, s, copy + h.off_sq8, true);
        const Sq8Codes c = sq8_open(s, false);
        touched(h.off_sq8, SQ8_HEADER_BYTES);
        touched(static_cast<std::uint64_t>(c.codes - copy) + n0 * c.stride, (n - n0) * c.stride);
        touched(static_cast<std::uint64_t>(reinterpret_cast<const std::uint8_t *>(c.sums) - copy) + n0 * 4, (n - n0) * 4);
    }
    if (graph) {
        hnsw_extend(bg, s, copy + h.off_graph, *graph, poll, true);
        const HnswGraph g = hnsw_open(s, false);
        touched(h.off_graph, HNSW_HEADER_BYTES);
        touched(static_cast<std::uint64_t>(g.levels - copy) + n0, n - n0);
        // A new node links into lists anywhere on level 0 and in the upper part: scanned whole.
        touched(static_cast<std::uint64_t>(reinterpret_cast<const std::uint8_t *>(g.level0) - copy), n * (std::uint64_t(g.m0) + 1) * 4);
        touched(static_cast<std::uint64_t>(reinterpret_cast<const std::uint8_t *>(g.upper_index) - copy) + n0 * 4, (n - n0) * 4);
        touched(static_cast<std::uint64_t>(reinterpret_cast<const std::uint8_t *>(g.upper) - copy), g.upper_blocks * (std::uint64_t(g.m) + 1) * 4);
    }
    return InPlace::Built;
}

std::vector<ByteRange> snapshot_diff(const std::uint8_t *a, const std::uint8_t *b, std::uint64_t size,
                                     const std::vector<ByteRange> &candidates, std::uint64_t chunk)
{
    if (chunk == 0 || chunk % DIFF_BLOCK) throw std::logic_error("snapshot_diff: chunk must be a multiple of the block");
    // Candidate ranges as sorted, merged block ranges.
    std::vector<ByteRange> ranges;
    for (const ByteRange &r : candidates) {
        if (r.bytes == 0 || r.offset >= size) continue;
        const std::uint64_t from = r.offset / DIFF_BLOCK, to = (std::min(size, r.offset + r.bytes) + DIFF_BLOCK - 1) / DIFF_BLOCK;
        ranges.push_back(ByteRange{from, to - from});
    }
    std::sort(ranges.begin(), ranges.end(), [](const ByteRange &x, const ByteRange &y) { return x.offset < y.offset; });
    std::vector<ByteRange> runs;
    std::uint64_t scanned_to = 0;           // blocks before it are done
    for (const ByteRange &r : ranges) {
        const std::uint64_t to = r.offset + r.bytes;
        for (std::uint64_t blk = std::max(r.offset, scanned_to); blk < to; ++blk) {
            const std::uint64_t off = blk * DIFF_BLOCK, len = std::min(DIFF_BLOCK, size - off);
            if (std::memcmp(a + off, b + off, len) == 0) continue;
            // Join with the run before when adjacent and the chunk allows it.
            if (!runs.empty() && runs.back().offset + runs.back().bytes == off &&
                runs.back().offset / chunk == off / chunk && runs.back().bytes + len <= chunk)
                runs.back().bytes += len;
            else
                runs.push_back(ByteRange{off, len});
        }
        scanned_to = std::max(scanned_to, to);
    }
    return runs;
}

void seal_in_place(const std::uint8_t *base, std::uint8_t *copy, std::uint64_t size, const std::vector<ByteRange> &runs)
{
    if (runs.empty() || runs.front().offset != 0) throw std::logic_error("seal_in_place: the first run must hold the header");
    const std::uint64_t field = offsetof(SnapshotHeader, checksum);
    std::uint64_t sum;
    std::memcpy(&sum, base + field, 8);
    // The checksum field counts as 0 on both sides: the words of a run that holds it are summed
    // around it.
    auto part = [&](const std::uint8_t *data, std::uint64_t off, std::uint64_t len) {
        if (off <= field && field < off + len) {
            return snapshot_checksum_part(data + off, field - off, off / 8) ^
                   snapshot_checksum_part(data + field + 8, off + len - field - 8, field / 8 + 1);
        }
        return snapshot_checksum_part(data + off, len, off / 8);
    };
    for (const ByteRange &r : runs) {
        if (r.offset % 8 || r.offset + r.bytes > size) throw std::logic_error("seal_in_place: a run is not word aligned");
        const std::uint64_t len = r.bytes / 8 * 8;      // the file size is a multiple of 64: runs end on words
        sum ^= part(base, r.offset, len) ^ part(copy, r.offset, len);
    }
    std::memcpy(copy + field, &sum, 8);
}

void pack_runs(const std::uint8_t *data, const std::vector<ByteRange> &runs, std::uint64_t capacity,
               const std::function<void(std::uint64_t, const std::uint8_t *, std::uint64_t)> &emit)
{
    if (capacity <= 2 * RUN_RECORD_HEADER + 8) throw std::logic_error("pack_runs: capacity too small for a record");
    const std::uint64_t total = runs.size();
    std::vector<std::uint8_t> row;
    row.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(capacity, 1u << 20)));
    std::uint64_t first = 0;
    auto flush = [&] {
        if (row.empty()) return;
        emit(first, row.data(), row.size());
        row.clear();
    };
    auto record = [&](std::uint64_t off, const std::uint8_t *bytes, std::uint32_t n) {
        const std::size_t at = row.size();
        row.resize(at + RUN_RECORD_HEADER + n);
        std::memcpy(row.data() + at, &off, 8);
        std::memcpy(row.data() + at + 8, &n, 4);
        std::memcpy(row.data() + at + RUN_RECORD_HEADER, bytes, n);
    };
    for (const ByteRange &r : runs) {
        std::uint64_t off = r.offset, left = r.bytes;
        while (left > 0) {
            if (row.size() + RUN_RECORD_HEADER >= capacity) flush();
            if (row.empty()) {
                // Every row starts with the summary record: the run total of the patch (vload
                // decides on it before it clones the base).
                first = off;
                record(RUN_SUMMARY_OFFSET, reinterpret_cast<const std::uint8_t *>(&total), 8);
            }
            const std::uint64_t room = capacity - row.size() - RUN_RECORD_HEADER;
            const std::uint64_t n = std::min<std::uint64_t>(left, std::min<std::uint64_t>(room, 0xFFFFFFFFu));
            record(off, data + off, static_cast<std::uint32_t>(n));
            off += n;
            left -= n;
        }
    }
    flush();
}

std::uint64_t patch_runs_of(const std::uint8_t *row, std::uint64_t bytes)
{
    if (bytes < RUN_RECORD_HEADER + 8) throw std::runtime_error("a patch row is shorter than its summary record");
    std::uint64_t off, total;
    std::uint32_t n;
    std::memcpy(&off, row, 8);
    std::memcpy(&n, row + 8, 4);
    if (off != RUN_SUMMARY_OFFSET || n != 8) throw std::runtime_error("a patch row does not start with its summary record");
    std::memcpy(&total, row + RUN_RECORD_HEADER, 8);
    return total;
}

void unpack_runs(const std::uint8_t *row, std::uint64_t bytes,
                 const std::function<void(std::uint64_t, const std::uint8_t *, std::uint64_t)> &apply)
{
    patch_runs_of(row, bytes);                          // the summary record must be there
    std::uint64_t at = RUN_RECORD_HEADER + 8;
    while (at < bytes) {
        if (bytes - at < RUN_RECORD_HEADER) throw std::runtime_error("a patch row ends inside a record header");
        std::uint64_t off;
        std::uint32_t n;
        std::memcpy(&off, row + at, 8);
        std::memcpy(&n, row + at + 8, 4);
        at += RUN_RECORD_HEADER;
        if (n == 0 || off == RUN_SUMMARY_OFFSET || bytes - at < n) throw std::runtime_error("a patch row ends inside a record");
        apply(off, row + at, n);
        at += n;
    }
}

} // namespace vvector
