#include "flat.h"
#include "kernels.h"
#include "parallel.h"

#include <algorithm>

namespace vvector {

bool within_radius(const FlatSearch &s, float key)
{
    const double score = key_to_score(s.metric, key);
    return s.metric == Metric::L2 || s.metric == Metric::L1 ? score <= s.radius : score >= s.radius;
}

namespace {

// A bounded max-heap under `closer`: d[0] is the farthest of the kept neighbours.
struct Heap {
    Neighbor *d = nullptr;
    std::uint32_t size = 0, cap = 0;
};

inline void offer(Heap &h, const FlatSearch &s, float key, std::int64_t id)
{
    if (key != key) return;                                   // NaN: never a result
    if (s.has_radius && !within_radius(s, key)) return;
    if (h.size < h.cap) {
        h.d[h.size++] = Neighbor{key, id};
        std::push_heap(h.d, h.d + h.size, closer);
    } else {
        std::pop_heap(h.d, h.d + h.size, closer);
        h.d[h.size - 1] = Neighbor{key, id};
        std::push_heap(h.d, h.d + h.size, closer);
    }
}

// Offers n keys of rows first .. first + n - 1 of block b to one heap.
inline void offer_all(Heap &h, const FlatSearch &s, const RowBlock &b, std::uint64_t first, const float *keys,
                      std::uint64_t n)
{
    for (std::uint64_t i = 0; i < n; ++i) {
        const float key = keys[i];
        const std::uint64_t row = first + i;
        const std::int64_t id = b.ids ? b.ids[row] : static_cast<std::int64_t>(row);
        if (h.size == h.cap && !(key < h.d[0].key || (key == h.d[0].key && id < h.d[0].id))) continue;
        if (b.skip && (b.skip[row >> 6] >> (row & 63) & 1u)) continue;
        offer(h, s, key, id);
    }
}

// Rows per tile: about 256 KB, so a tile stays in the L2 cache while every group of 4 queries
// is scored against it. At least 16 rows.
inline std::uint64_t tile_rows(std::uint32_t stride)
{
    return std::max<std::uint64_t>(16, (256u * 1024u) / (std::uint64_t(stride) * 4));
}

// A block with sq8 codes: integer sums of a tile of code rows per query, turned into sq8 keys.
void scan_codes(const FlatSearch &s, const RowBlock &b, std::uint64_t r0, std::uint64_t r1, std::uint64_t q0,
                std::uint64_t q1, Heap *heaps, std::vector<float> &keys)
{
    const Sq8Codes &c = *b.codes;
    const std::uint64_t tile = std::max<std::uint64_t>(16, (256u * 1024u) / c.stride);
    keys.resize(tile);
    std::vector<std::uint32_t> sums(4 * tile);
    auto offer_codes = [&](std::uint64_t q, std::uint64_t t0, std::uint64_t n, const std::uint32_t *qs) {
        for (std::uint64_t i = 0; i < n; ++i) keys[i] = sq8_key(s.metric, c.range, qs[i], c.sums[t0 + i], s.query_sums[q], c.dims);
        offer_all(heaps[q - q0], s, b, t0, keys.data(), n);
    };
    for (std::uint64_t t0 = r0; t0 < r1; t0 += tile) {
        const std::uint64_t n = std::min(tile, r1 - t0);
        std::uint64_t q = q0;
        for (; q + 4 <= q1; q += 4) {        // 4 queries per pass over the tile, as the float scan
            const std::uint8_t *qs[4] = {s.query_codes + q * c.stride, s.query_codes + (q + 1) * c.stride,
                                         s.query_codes + (q + 2) * c.stride, s.query_codes + (q + 3) * c.stride};
            sq8_sums_4q(s.metric, c.row(t0), n, c.stride, qs, sums.data());
            for (int j = 0; j < 4; ++j) offer_codes(q + j, t0, n, sums.data() + j * n);
        }
        for (; q < q1; ++q) {
            sq8_sums_1q(s.metric, c.row(t0), n, c.stride, s.query_codes + q * c.stride, sums.data());
            offer_codes(q, t0, n, sums.data());
        }
    }
}

// Scores rows r0 .. r1 - 1 of block b against queries q0 .. q1 - 1 and offers them to heaps[q - q0].
void scan(const FlatSearch &s, const RowBlock &b, std::uint64_t r0, std::uint64_t r1, std::uint64_t q0,
          std::uint64_t q1, Heap *heaps, std::vector<float> &keys)
{
    if (b.codes) { scan_codes(s, b, r0, r1, q0, q1, heaps, keys); return; }
    const std::uint64_t tile = tile_rows(s.stride);
    keys.resize(4 * tile);
    for (std::uint64_t t0 = r0; t0 < r1; t0 += tile) {
        const std::uint64_t n = std::min(tile, r1 - t0);
        const float *rows = b.rows + t0 * s.stride;
        std::uint64_t q = q0;
        for (; q + 4 <= q1; q += 4) {
            const float *qs[4] = {s.queries + q * s.stride, s.queries + (q + 1) * s.stride,
                                  s.queries + (q + 2) * s.stride, s.queries + (q + 3) * s.stride};
            keys_4q(s.metric, rows, n, s.stride, qs, keys.data());
            for (int j = 0; j < 4; ++j) offer_all(heaps[q + j - q0], s, b, t0, keys.data() + j * n, n);
        }
        for (; q < q1; ++q) {
            keys_1q(s.metric, rows, n, s.stride, s.queries + q * s.stride, keys.data());
            offer_all(heaps[q - q0], s, b, t0, keys.data(), n);
        }
    }
}

// Per-thread heaps of the row split: at most this many bytes, else the queries are split.
constexpr std::uint64_t ROW_SPLIT_HEAP_BYTES = 64u << 20;
// Below this many multiply-adds the search runs on the calling thread only.
constexpr std::uint64_t SERIAL_WORK = 1u << 21;

} // namespace

void flat_search(const FlatSearch &s, const RowBlock *blocks, std::size_t n_blocks, std::vector<Neighbor> &out,
                 std::vector<std::uint32_t> &count, const std::function<bool()> &poll)
{
    const std::uint64_t nq = s.n_queries, k = s.k;
    out.assign(nq * k, Neighbor{0, 0});
    count.assign(nq, 0);
    if (nq == 0 || k == 0) return;

    std::uint64_t total_rows = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) total_rows += blocks[b].n;
    int threads = std::max(1, s.threads);
    const std::uint64_t work = total_rows * nq * s.stride;
    if (work < SERIAL_WORK) threads = 1;
    else threads = static_cast<int>(std::min<std::uint64_t>(threads, work / (SERIAL_WORK / 2)));

    const bool row_split = threads > 1 && std::uint64_t(threads) * nq * k * sizeof(Neighbor) <= ROW_SPLIT_HEAP_BYTES;
    std::vector<std::vector<float>> keys(threads);

    if (!row_split) {
        // Split the queries. Each query's heap lives in its slice of out and sees every row.
        std::vector<Heap> heaps(nq);
        for (std::uint64_t q = 0; q < nq; ++q) heaps[q] = Heap{out.data() + q * k, 0, static_cast<std::uint32_t>(k)};
        std::uint64_t per_unit = nq;
        if (threads > 1) per_unit = std::min<std::uint64_t>(64, std::max<std::uint64_t>(4, (nq + 2 * threads - 1) / (2 * threads)));
        per_unit = (per_unit + 3) / 4 * 4;
        parallel_ranges(nq, per_unit, threads, [&](int t, std::uint64_t, std::uint64_t q0, std::uint64_t q1) {
            for (std::size_t b = 0; b < n_blocks; ++b)
                scan(s, blocks[b], 0, blocks[b].n, q0, q1, heaps.data() + q0, keys[t]);
        }, poll);
        for (std::uint64_t q = 0; q < nq; ++q) {
            std::sort_heap(heaps[q].d, heaps[q].d + heaps[q].size, closer);
            count[q] = heaps[q].size;
        }
        return;
    }

    // Split the rows. Every thread keeps heaps for all queries; they are merged afterwards.
    struct Unit { std::size_t block; std::uint64_t r0, r1; };
    const std::uint64_t tile = tile_rows(s.stride);
    std::uint64_t per_unit = (total_rows + 8 * std::uint64_t(threads) - 1) / (8 * std::uint64_t(threads));
    per_unit = std::max(tile, (per_unit + tile - 1) / tile * tile);
    std::vector<Unit> units;
    for (std::size_t b = 0; b < n_blocks; ++b)
        for (std::uint64_t r = 0; r < blocks[b].n; r += per_unit)
            units.push_back(Unit{b, r, std::min(blocks[b].n, r + per_unit)});

    std::vector<Neighbor> store(std::uint64_t(threads) * nq * k);
    std::vector<Heap> heaps(std::uint64_t(threads) * nq);
    for (std::uint64_t h = 0; h < heaps.size(); ++h) heaps[h] = Heap{store.data() + h * k, 0, static_cast<std::uint32_t>(k)};
    parallel_ranges(units.size(), 1, threads, [&](int t, std::uint64_t u, std::uint64_t, std::uint64_t) {
        const Unit &x = units[u];
        scan(s, blocks[x.block], x.r0, x.r1, 0, nq, heaps.data() + std::uint64_t(t) * nq, keys[t]);
    }, poll);

    std::vector<Neighbor> all;
    for (std::uint64_t q = 0; q < nq; ++q) {
        all.clear();
        for (int t = 0; t < threads; ++t) {
            const Heap &h = heaps[std::uint64_t(t) * nq + q];
            all.insert(all.end(), h.d, h.d + h.size);
        }
        const std::uint64_t keep = std::min<std::uint64_t>(k, all.size());
        std::partial_sort(all.begin(), all.begin() + keep, all.end(), closer);
        std::copy(all.begin(), all.begin() + keep, out.begin() + q * k);
        count[q] = static_cast<std::uint32_t>(keep);
    }
}

void merge_block(const FlatSearch &s, const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                 const std::function<bool()> &poll)
{
    const std::uint64_t nq = s.n_queries, k = s.k;
    std::vector<Neighbor> jout, merged;
    std::vector<std::uint32_t> jcount;
    flat_search(s, extra, 1, jout, jcount, poll);
    for (std::uint64_t q = 0; q < nq; ++q) {
        merged.clear();
        std::merge(out.begin() + q * k, out.begin() + q * k + count[q], jout.begin() + q * k,
                   jout.begin() + q * k + jcount[q], std::back_inserter(merged), closer);
        const std::uint64_t keep = std::min<std::uint64_t>(k, merged.size());
        std::copy(merged.begin(), merged.begin() + keep, out.begin() + q * k);
        count[q] = static_cast<std::uint32_t>(keep);
    }
}

} // namespace vvector
