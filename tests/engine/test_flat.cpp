// Flat search against a naive implementation: single and batch queries, k from 1 to more than
// the rows, masks, journal adds, deletes and re-adds, radius, ties, every thread count and both
// ways of splitting the work. The results must be exactly equal, not only close.
#include "check.h"

#include "../../src/engine/flat.h"
#include "../../src/engine/kernels.h"
#include "../../src/engine/parallel.h"

#include <algorithm>
#include <cmath>

using namespace vvector;

struct Data {
    std::uint32_t dims, stride;
    std::vector<float> rows;
    std::vector<std::int64_t> ids;
    std::vector<std::uint64_t> skip;
    Data(std::uint64_t n, std::uint32_t dims, std::uint64_t seed, Metric m, std::int64_t id_base)
        : dims(dims), stride(row_stride_for(dims)), rows(n * stride, 0.0f), ids(n), skip((n + 63) / 64, 0)
    {
        for (std::uint64_t i = 0; i < n; ++i) {
            test_vector(i, dims, seed, true, rows.data() + i * stride);
            if (m == Metric::Cosine) normalize(rows.data() + i * stride, dims);
            ids[i] = id_base + static_cast<std::int64_t>(i) * 3;
        }
    }
    std::uint64_t n() const { return ids.size(); }
    RowBlock block(bool with_skip) const { return RowBlock{rows.data(), ids.data(), n(), with_skip ? skip.data() : nullptr}; }
};

static void naive(const FlatSearch &s, const RowBlock *blocks, std::size_t nb, std::vector<Neighbor> &out,
                  std::vector<std::uint32_t> &count)
{
    out.assign(s.n_queries * s.k, Neighbor{0, 0});
    count.assign(s.n_queries, 0);
    for (std::uint64_t q = 0; q < s.n_queries; ++q) {
        std::vector<Neighbor> all;
        for (std::size_t b = 0; b < nb; ++b)
            for (std::uint64_t i = 0; i < blocks[b].n; ++i) {
                if (blocks[b].skip && (blocks[b].skip[i / 64] >> (i % 64) & 1)) continue;
                const float key = distance_key(s.metric, blocks[b].rows + i * s.stride, s.queries + q * s.stride, s.stride);
                if (s.has_radius && !within_radius(s, key)) continue;
                all.push_back(Neighbor{key, blocks[b].ids[i]});
            }
        std::sort(all.begin(), all.end(), closer);
        const std::uint64_t keep = std::min<std::uint64_t>(s.k, all.size());
        std::copy(all.begin(), all.begin() + keep, out.begin() + q * s.k);
        count[q] = static_cast<std::uint32_t>(keep);
    }
}

static bool equal(const FlatSearch &s, const std::vector<Neighbor> &a, const std::vector<std::uint32_t> &ca,
                  const std::vector<Neighbor> &b, const std::vector<std::uint32_t> &cb)
{
    if (ca != cb) return false;
    for (std::uint64_t q = 0; q < s.n_queries; ++q)
        for (std::uint32_t i = 0; i < ca[q]; ++i) {
            const Neighbor &x = a[q * s.k + i], &y = b[q * s.k + i];
            if (x.id != y.id || std::memcmp(&x.key, &y.key, 4) != 0) return false;
        }
    return true;
}

// Runs the search with every thread count and compares with the naive result.
static void check_all(const char *what, FlatSearch s, const RowBlock *blocks, std::size_t nb)
{
    std::vector<Neighbor> ref, got;
    std::vector<std::uint32_t> rc, gc;
    naive(s, blocks, nb, ref, rc);
    for (int t : {1, 2, 7, 64}) {
        s.threads = t;
        flat_search(s, blocks, nb, got, gc);
        const bool ok = equal(s, got, gc, ref, rc);
        CHECK(ok);
        if (!ok) std::printf("  %s: %s, %llu queries, k %u, %d threads\n", what, metric_name(s.metric),
                             (unsigned long long)s.n_queries, s.k, t);
    }
}

int main()
{
    const Metric metrics[4] = {Metric::L2, Metric::Cosine, Metric::Dot, Metric::L1};
    for (Metric m : metrics) {
        // Big enough for the parallel paths: 20,000 rows of 40 dims (stride 48).
        Data snap(20000, 40, 5, m, 1000);
        Data queries(70, 40, 6, m, 0);
        FlatSearch s;
        s.metric = m;
        s.stride = snap.stride;
        s.queries = queries.rows.data();

        for (std::uint64_t nq : {1ull, 3ull, 4ull, 5ull, 17ull, 70ull})
            for (std::uint32_t k : {1u, 10u, 100u}) {
                s.n_queries = nq;
                s.k = k;
                const RowBlock b = snap.block(false);
                check_all("plain", s, &b, 1);
            }

        // k larger than the number of rows: every row, in order.
        {
            Data small(50, 40, 7, m, 5);
            s.n_queries = 5;
            s.k = 64;
            const RowBlock b = small.block(false);
            check_all("k > rows", s, &b, 1);
            std::vector<Neighbor> got;
            std::vector<std::uint32_t> gc;
            flat_search(s, &b, 1, got, gc);
            CHECK(gc[0] == 50 && gc[4] == 50);
        }

        // The journal: 300 deleted ids (masked, no new vector), 200 replaced ids (masked, new
        // vector with the same id: a re-add), 500 new ids.
        {
            Data journal(700, 40, 8, m, 0);
            for (std::uint64_t j = 0; j < 500; ++j) {
                const std::uint64_t pos = (j * 7919 + 11) % 20000;         // 500 different positions
                snap.skip[pos / 64] |= 1ull << (pos % 64);
                if (j < 200) journal.ids[j] = snap.ids[pos];                  // re-add of an existing id
            }
            for (std::uint64_t j = 200; j < 700; ++j) journal.ids[j] = 900000000 + static_cast<std::int64_t>(j);
            const RowBlock blocks[2] = {snap.block(true), journal.block(false)};
            s.n_queries = 17;
            s.k = 25;
            check_all("journal", s, blocks, 2);
            // No deleted id comes back.
            std::vector<Neighbor> got;
            std::vector<std::uint32_t> gc;
            s.threads = 4;
            s.k = 20000;
            s.n_queries = 1;
            flat_search(s, blocks, 2, got, gc);
            bool none_masked = true;
            for (std::uint32_t i = 0; i < gc[0]; ++i) {
                const std::int64_t pos = (got[i].id - 1000) / 3;
                const bool from_snap = got[i].id < 900000000 && (got[i].id - 1000) % 3 == 0;
                const bool replaced = std::find(journal.ids.begin(), journal.ids.begin() + 200, got[i].id) != journal.ids.begin() + 200;
                if (from_snap && !replaced && (snap.skip[pos / 64] >> (pos % 64) & 1)) none_masked = false;
            }
            CHECK(none_masked);
            std::fill(snap.skip.begin(), snap.skip.end(), 0);
        }

        // Radius: only candidates within it, at most k.
        {
            s.n_queries = 9;
            s.k = 50;
            s.has_radius = true;
            std::vector<Neighbor> all;
            std::vector<std::uint32_t> c;
            FlatSearch probe = s;
            probe.has_radius = false;
            probe.k = 200;
            const RowBlock b = snap.block(false);
            flat_search(probe, &b, 1, all, c);
            s.radius = key_to_score(m, all[100].key);          // about 100 rows within it for query 0
            check_all("radius", s, &b, 1);
            std::vector<Neighbor> got;
            flat_search(s, &b, 1, got, c);
            bool inside = true;
            for (std::uint64_t q = 0; q < s.n_queries; ++q)
                for (std::uint32_t i = 0; i < c[q]; ++i) inside = inside && within_radius(s, got[q * s.k + i].key);
            CHECK(inside && c[0] == 50);
            s.radius = m == Metric::L2 || m == Metric::L1 ? -1.0 : 1e9;   // nothing within
            flat_search(s, &b, 1, got, c);
            CHECK(c[0] == 0 && c[8] == 0);
            s.has_radius = false;
        }

        // Blocks merged one after another (vscan reads a table in blocks of rows and merges each
        // into the running result with merge_block) give exactly the one-block search, whatever
        // the block size, with and without a radius.
        for (bool radius : {false, true}) {
            s.n_queries = 17;
            s.k = 25;
            s.threads = 1;
            s.has_radius = radius;
            const RowBlock whole = snap.block(false);
            if (radius) {
                std::vector<Neighbor> all;
                std::vector<std::uint32_t> c;
                FlatSearch probe = s;
                probe.has_radius = false;
                probe.k = 100;
                flat_search(probe, &whole, 1, all, c);
                s.radius = key_to_score(m, all[60].key);
            }
            std::vector<Neighbor> ref, got;
            std::vector<std::uint32_t> rc, gc;
            flat_search(s, &whole, 1, ref, rc);
            for (std::uint64_t block_rows : {1ull, 7ull, 1000ull, 4096ull, 30000ull}) {
                got.assign(s.n_queries * s.k, Neighbor{0, 0});
                gc.assign(s.n_queries, 0);
                for (std::uint64_t at = 0; at < snap.n(); at += block_rows) {
                    RowBlock part;
                    part.rows = snap.rows.data() + at * snap.stride;
                    part.ids = snap.ids.data() + at;
                    part.n = std::min(block_rows, snap.n() - at);
                    merge_block(s, &part, got, gc);
                }
                const bool ok = equal(s, got, gc, ref, rc);
                CHECK(ok);
                if (!ok) std::printf("  merged blocks of %llu rows differ (%s, radius %d)\n", (unsigned long long)block_rows, metric_name(m), radius);
            }
            s.has_radius = false;
        }
    }

    // Ties: equal vectors under different ids come back ordered by id, whatever the threads.
    {
        const std::uint32_t dims = 16;
        std::vector<float> rows(4000 * dims), q(dims, 0.5f);
        std::vector<std::int64_t> ids(4000);
        for (std::uint64_t i = 0; i < 4000; ++i) {
            for (std::uint32_t d = 0; d < dims; ++d) rows[i * dims + d] = static_cast<float>(i % 7);
            ids[i] = static_cast<std::int64_t>((i * 7919) % 4000) - 2000;
        }
        FlatSearch s;
        s.metric = Metric::L2;
        s.stride = dims;
        s.queries = q.data();
        s.n_queries = 1;
        s.k = 600;
        const RowBlock b{rows.data(), ids.data(), 4000, nullptr};
        check_all("ties", s, &b, 1);
        std::vector<Neighbor> got;
        std::vector<std::uint32_t> c;
        s.threads = 8;
        flat_search(s, &b, 1, got, c);
        bool by_id = true;
        for (std::uint32_t i = 1; i < c[0]; ++i)
            by_id = by_id && (got[i - 1].key < got[i].key || (got[i - 1].key == got[i].key && got[i - 1].id < got[i].id));
        CHECK(by_id && got[0].key == got[570].key);      // values 0 and 1: 1143 rows share the smallest distance
    }

    // Cancellation stops the search.
    {
        Data snap(20000, 40, 5, Metric::L2, 0);
        FlatSearch s;
        s.stride = snap.stride;
        s.queries = snap.rows.data();
        s.n_queries = 64;
        s.threads = 4;
        const RowBlock b = snap.block(false);
        std::vector<Neighbor> got;
        std::vector<std::uint32_t> c;
        CHECK(throws([&] { flat_search(s, &b, 1, got, c, [] { return true; }); }, "cancelled"));
    }
    return finish("test_flat");
}
