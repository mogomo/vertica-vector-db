#include "search.h"
#include "hnsw.h"
#include "kernels.h"
#include "parallel.h"
#include "sq8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace vvector {

std::uint32_t code_candidates(std::uint32_t k, const SearchPlan &p)
{
    if (!p.rescore) return k;
    const double c = std::ceil(double(k) * std::max(1.0, p.oversampling));
    return static_cast<std::uint32_t>(std::min<double>(c, 1u << 30));
}

std::uint64_t filter_exact_limit(std::uint32_t ef, std::uint64_t count)
{
    return std::max<std::uint64_t>(10000, static_cast<std::uint64_t>(std::sqrt(64.0 * double(ef) * double(count))));
}

namespace {

// The live allowed positions: sorted, once each, none in skip.
std::vector<std::uint32_t> live_allowed(const VectorSet &set, const SearchPlan &p, const std::uint64_t *skip)
{
    std::vector<std::uint32_t> live(p.allow, p.allow + p.n_allow);
    for (const std::uint32_t pos : live)
        if (pos >= set.count) throw std::runtime_error("filtered search: position " + std::to_string(pos) + " is not in the snapshot");
    if (!std::is_sorted(live.begin(), live.end())) std::sort(live.begin(), live.end());
    live.erase(std::unique(live.begin(), live.end()), live.end());
    if (skip) live.erase(std::remove_if(live.begin(), live.end(), [skip](std::uint32_t pos) {
        return (skip[pos >> 6] >> (pos & 63) & 1u) != 0;
    }), live.end());
    return live;
}

} // namespace

void index_search(const FlatSearch &s, const VectorSet &set, const SearchPlan &p, const std::uint64_t *skip,
                  const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                  const std::function<bool()> &poll)
{
    std::vector<std::uint64_t> mask;
    if (p.filtered) {
        const std::vector<std::uint32_t> live = live_allowed(set, p, skip);
        const std::uint64_t limit = p.exact_below >= 0 ? static_cast<std::uint64_t>(p.exact_below) : filter_exact_limit(p.ef, set.count);
        if (live.size() < limit) {
            // Few allowed rows: copied together (in parallel, into memory that is not cleared first) and
            // searched exactly with the tiled kernels, with the journal's rows beside them.
            const std::uint64_t stride = set.row_stride, n = live.size();
            std::unique_ptr<float[]> rows(new float[std::max<std::uint64_t>(n, 1) * stride]);
            std::unique_ptr<std::int64_t[]> ids(new std::int64_t[std::max<std::uint64_t>(n, 1)]);
            parallel_ranges(n, 4096, std::max(1, std::min<int>(s.threads, static_cast<int>(n / 4096) + 1)),
                            [&](int, std::uint64_t, std::uint64_t i0, std::uint64_t i1) {
                for (std::uint64_t i = i0; i < i1; ++i) {
                    std::memcpy(rows.get() + i * stride, set.vector(live[i]), stride * sizeof(float));
                    ids[i] = set.ids[live[i]];
                }
            }, poll);
            RowBlock blocks[2];
            blocks[0] = RowBlock{rows.get(), ids.get(), n, nullptr};
            std::size_t n_blocks = 1;
            if (extra && extra->n) blocks[n_blocks++] = *extra;
            flat_search(s, blocks, n_blocks, out, count, poll);
            return;
        }
        // Many: every position outside the list is skipped.
        mask.assign((set.count + 63) / 64, ~0ull);
        for (const std::uint32_t pos : live) mask[pos >> 6] &= ~(1ull << (pos & 63));
        skip = mask.data();
    }

    if (!p.codes) {
        if (p.graph) {
            const HnswGraph g = hnsw_open(set, false);
            hnsw_search(s, set, g, p.ef, skip, extra, out, count, poll);
            return;
        }
        RowBlock blocks[2];
        blocks[0] = RowBlock{set.vectors, set.ids, set.count, skip};
        std::size_t n = 1;
        if (extra && extra->n) blocks[n++] = *extra;
        flat_search(s, blocks, n, out, count, poll);
        return;
    }

    const std::uint64_t nq = s.n_queries, k = s.k;
    const Sq8Codes c = sq8_open(set, false);
    std::vector<std::uint8_t> qcodes(nq * c.stride, 0);
    std::vector<std::uint32_t> qsums(nq);
    for (std::uint64_t q = 0; q < nq; ++q) qsums[q] = sq8_encode(c.range, s.queries + q * s.stride, set.dims, &qcodes[q * c.stride]);

    // Candidates by their codes. The radius waits for the scores that are returned.
    FlatSearch cs = s;
    cs.k = std::min<std::uint32_t>(code_candidates(s.k, p), static_cast<std::uint32_t>(std::min<std::uint64_t>(std::max<std::uint64_t>(set.count, 1), 1u << 30)));   // never more than the positions
    cs.has_radius = s.has_radius && p.graph;      // the graph's range walk only (hnsw.h); keys not cut
    cs.query_codes = qcodes.data();
    cs.query_sums = qsums.data();
    std::vector<Neighbor> cand;
    std::vector<std::uint32_t> ccount;
    if (p.graph) {
        const HnswGraph g = hnsw_open(set, false);
        hnsw_search_codes(cs, set, c, g, p.ef, skip, cand, ccount, poll);
    } else {
        const RowBlock b{set.vectors, nullptr, set.count, skip, &c};      // ids null: candidates carry positions
        flat_search(cs, &b, 1, cand, ccount, poll);
    }

    // The candidates carry their positions. Rescoring: the exact key of every candidate from its
    // float row; then the ids, and the k best by (key, id). In parallel over the queries.
    out.assign(nq * k, Neighbor{0, 0});
    count.assign(nq, 0);
    const int threads = static_cast<int>(std::min<std::uint64_t>(std::max(1, s.threads), (nq + 15) / 16));
    std::vector<std::vector<Neighbor>> work(threads);
    parallel_ranges(nq, 16, threads, [&](int t, std::uint64_t, std::uint64_t q0, std::uint64_t q1) {
        std::vector<Neighbor> &r = work[t];
        for (std::uint64_t q = q0; q < q1; ++q) {
            r.assign(cand.begin() + q * cs.k, cand.begin() + q * cs.k + ccount[q]);
            const float *qv = s.queries + q * s.stride;
            for (Neighbor &nb : r) {
                const std::uint64_t pos = static_cast<std::uint64_t>(nb.id);
                if (p.rescore) nb.key = distance_key(s.metric, qv, set.vector(pos), s.stride);
                nb.id = set.ids[pos];
            }
            std::sort(r.begin(), r.end(), closer);
            std::uint32_t n = 0;
            for (const Neighbor &nb : r) {
                if (n == k) break;
                if (nb.key != nb.key) continue;
                if (s.has_radius && !within_radius(s, nb.key)) continue;
                out[q * k + n++] = nb;
            }
            count[q] = n;
        }
    }, poll);
    if (extra && extra->n) merge_block(s, extra, out, count, poll);
}

} // namespace vvector
