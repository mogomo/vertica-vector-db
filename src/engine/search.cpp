#include "search.h"
#include "hnsw.h"
#include "kernels.h"
#include "parallel.h"
#include "sq8.h"

#include <algorithm>
#include <cmath>

namespace vvector {

std::uint32_t code_candidates(std::uint32_t k, const SearchPlan &p)
{
    if (!p.rescore) return k;
    const double c = std::ceil(double(k) * std::max(1.0, p.oversampling));
    return static_cast<std::uint32_t>(std::min<double>(c, 1u << 30));
}

void index_search(const FlatSearch &s, const VectorSet &set, const SearchPlan &p, const std::uint64_t *skip,
                  const RowBlock *extra, std::vector<Neighbor> &out, std::vector<std::uint32_t> &count,
                  const std::function<bool()> &poll)
{
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
    cs.k = code_candidates(s.k, p);
    cs.has_radius = false;
    cs.query_codes = qcodes.data();
    cs.query_sums = qsums.data();
    std::vector<Neighbor> cand;
    std::vector<std::uint32_t> ccount;
    if (p.graph) {
        const HnswGraph g = hnsw_open(set, false);
        hnsw_search_codes(cs, set, c, g, std::max(p.ef, cs.k), skip, cand, ccount, poll);
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
