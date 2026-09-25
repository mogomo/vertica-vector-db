#include "hnsw.h"
#include "kernels.h"
#include "parallel.h"
#include "reach.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vvector {

namespace {

// A candidate: its key (kernels.h: smaller is closer) and its position.
struct Cand {
    float key;
    std::uint32_t pos;
};

// The order of candidates: smaller key first, ties by smaller position. A total order, so every
// heap below behaves the same on every platform.
inline bool nearer(const Cand &a, const Cand &b) { return a.key < b.key || (a.key == b.key && a.pos < b.pos); }
// Heap comparators: with FartherTop the heap front is the farthest, with NearerTop the nearest.
struct FartherTop { bool operator()(const Cand &a, const Cand &b) const { return nearer(a, b); } };
struct NearerTop { bool operator()(const Cand &a, const Cand &b) const { return nearer(b, a); } };

std::uint64_t align64(std::uint64_t v) { return (v + 63) / 64 * 64; }

// Offsets of the parts of a graph section, from the section start.
struct GraphLayout {
    std::uint64_t levels, level0, upper_index, upper, bytes;
};

// The parts are sized by what the layout has room for (capacity positions, upper_capacity blocks;
// without FLAG_CAPACITY count and upper_blocks), so appended positions move nothing.
GraphLayout graph_layout(std::uint64_t capacity, std::uint32_t m, std::uint64_t upper_capacity)
{
    GraphLayout l;
    l.levels = HNSW_HEADER_BYTES;
    l.level0 = align64(l.levels + capacity);
    l.upper_index = align64(l.level0 + capacity * (2 * std::uint64_t(m) + 1) * 4);
    l.upper = align64(l.upper_index + capacity * 4);
    l.bytes = align64(l.upper + upper_capacity * (std::uint64_t(m) + 1) * 4);
    return l;
}

// Levels of n new ids, and the blocks they add.
std::uint64_t levels_of(const std::int64_t *ids, std::uint64_t n, std::uint32_t m, std::uint8_t *levels, std::uint32_t *first,
                        std::uint64_t blocks)
{
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint32_t l = hnsw_level(ids[i], m);
        if (levels) levels[i] = static_cast<std::uint8_t>(l);
        if (first) first[i] = l == 0 ? HNSW_NO_UPPER : static_cast<std::uint32_t>(blocks);
        blocks += l;
    }
    return blocks;
}

void check_m(std::uint32_t m)
{
    if (m < HNSW_MIN_M || m > HNSW_MAX_M) throw std::runtime_error("m must be 2 to 256, not " + std::to_string(m));
}

// Stamps of the positions visited by one search. A new search takes the next stamp instead of
// clearing the array; the array is cleared only when the 16-bit stamp wraps (as hnswlib).
class Visited {
public:
    void reserve(std::uint64_t n)
    {
        if (marks_.size() >= n) return;
        marks_.assign(n, 0);
        stamp_ = 0;
    }
    void next()
    {
        if (++stamp_ == 0) {
            std::fill(marks_.begin(), marks_.end(), 0);
            stamp_ = 1;
        }
    }
    // True when pos was visited before in this search; marks it.
    bool test_set(std::uint32_t pos)
    {
        if (marks_[pos] == stamp_) return true;
        marks_[pos] = stamp_;
        return false;
    }
    void prefetch(std::uint32_t pos) const { __builtin_prefetch(marks_.data() + pos); }
    std::uint64_t bytes() const { return marks_.size() * sizeof(std::uint16_t); }

private:
    std::vector<std::uint16_t> marks_;
    std::uint16_t stamp_ = 0;
};

// Visited arrays kept by the process between searches (2 bytes per vector each): a new array of a
// large index would pay a page fault per 4 KB on every call. At most 64 arrays and 512 MB are kept
// (a 100M index has 200 MB arrays: a build on 64 threads must not pin 12.8 GB for the life of an
// unfenced process); beyond that an array is freed when its search ends.
class VisitedPool {
public:
    std::unique_ptr<Visited> get(std::uint64_t n)
    {
        std::unique_ptr<Visited> v;
        {
            std::lock_guard<std::mutex> hold(lock_);
            if (!free_.empty()) { v = std::move(free_.back()); free_.pop_back(); kept_ -= v->bytes(); }
        }
        if (!v) v.reset(new Visited());
        v->reserve(n);
        return v;
    }
    void put(std::unique_ptr<Visited> v)
    {
        std::lock_guard<std::mutex> hold(lock_);
        if (free_.size() < 64 && kept_ + v->bytes() <= (512ull << 20)) {
            kept_ += v->bytes();
            free_.push_back(std::move(v));
        }
    }

private:
    std::mutex lock_;
    std::vector<std::unique_ptr<Visited>> free_;
    std::uint64_t kept_ = 0;        // bytes of the arrays in free_
};

VisitedPool &visited_pool()
{
    static VisitedPool pool;
    return pool;
}

// Per-thread working memory of a build or a search.
struct Worker {
    std::unique_ptr<Visited> visited;
    std::vector<Cand> top, cand, shrink;
    std::vector<std::vector<Cand>> sel_by_level;     // a build's selected neighbours of each level
    std::vector<std::uint32_t> links, fresh, sums;
    std::vector<float> keys;
    explicit Worker(std::uint64_t count) : visited(visited_pool().get(count)) {}
    ~Worker() { if (visited) visited_pool().put(std::move(visited)); }
    Worker(Worker &&) = default;
};

// Keys of the positions in w.fresh against q, into w.keys.
inline void score_fresh(const VectorSet &s, const float *q, Worker &w)
{
    w.keys.resize(w.fresh.size());
    keys_gather(s.metric, s.vectors, s.row_stride, w.fresh.data(), static_cast<std::uint32_t>(w.fresh.size()), q,
                w.keys.data());
}

// ---- build

class Builder {
public:
    Builder(const VectorSet &s, std::uint8_t *section, const GraphLayout &l, const HnswParams &p)
        : s_(s), m_(p.m), m0_(2 * p.m), efc_(std::max<std::uint32_t>(p.ef_construction, p.m)), stripes_(stripes_for(s.count))
    {
        levels_ = section + l.levels;
        level0_ = reinterpret_cast<std::uint32_t *>(section + l.level0);
        upper_index_ = reinterpret_cast<std::uint32_t *>(section + l.upper_index);
        upper_ = reinterpret_cast<std::uint32_t *>(section + l.upper);
    }

    // Continues the graph of a base snapshot (incremental build): its links are in the section already.
    void start_from(std::uint32_t entry, std::uint32_t max_level)
    {
        entry_ = entry;
        max_level_ = static_cast<int>(max_level);
    }

    // Inserts position p (Algorithm 1 of the paper, as hnswlib's addPoint). Tombstoned positions
    // (incremental builds) are passed through but never chosen as neighbours, as in hnswlib.
    // Two phases: first the searches of every level, top down, then the links, bottom up. While p
    // searches, no node links to p, so no other thread can reach it; once p is linked on a level,
    // its lists on the levels below are complete. (hnswlib links each level right after its search:
    // a thread that reaches p on a level before p's lower levels are linked then only links to p and
    // other such late nodes there.)
    void insert(std::uint32_t p, Worker &w)
    {
        const std::uint32_t level = levels_[p];
        std::unique_lock<std::mutex> hold(global_);
        const int top_level = max_level_;
        const std::uint32_t start = entry_;
        if (top_level < 0) {            // the first node
            entry_ = p;
            max_level_ = static_cast<int>(level);
            return;
        }
        // Keep the global lock only when p becomes the new entry point.
        if (static_cast<int>(level) <= top_level) hold.unlock();

        const float *q = s_.vector(p);
        Cand cur{distance_key(s_.metric, q, s_.vector(start), s_.row_stride), start};
        for (int l = top_level; l > static_cast<int>(level); --l) cur = greedy(q, cur, static_cast<std::uint32_t>(l), w);
        const int low_top = std::min(static_cast<int>(level), top_level);
        if (w.sel_by_level.size() < static_cast<std::size_t>(low_top) + 1) w.sel_by_level.resize(low_top + 1);
        for (int l = low_top; l >= 0; --l) {
            std::vector<Cand> &sel = w.sel_by_level[l];
            search_layer(q, cur, static_cast<std::uint32_t>(l), w, true, p);
            std::sort(w.top.begin(), w.top.end(), nearer);
            select(w.top, m_, sel);
            if (!sel.empty()) cur = sel.front();         // empty only when every node found is tombstoned
        }
        for (int l = 0; l <= low_top; ++l) connect(p, static_cast<std::uint32_t>(l), w.sel_by_level[l], w);
        if (static_cast<int>(level) > top_level) {
            entry_ = p;
            max_level_ = static_cast<int>(level);
        }
    }

    // Makes every position reachable on level 0 from the entry point. The neighbour heuristic can
    // prune the last link to a node (hnswlib has the same effect, mostly with small m and with many
    // equal vectors); such a node can never be found. A breadth-first walk marks what is
    // reachable; every node it misses is linked from a reachable node: the nearest one with room
    // in its list among the results of a search for it; else the node linked just before, when it
    // is as near as those results (runs of equal vectors); else the nearest result gives up its
    // farthest link. Tombstoned nodes are not repaired, but serve as links. Runs on one thread
    // after the inserts. Returns the number of links made.
    std::uint64_t repair(Worker &w)
    {
        const std::uint64_t n = s_.count;
        std::vector<std::uint64_t> seen((n + 63) / 64, 0);
        auto is_seen = [&](std::uint32_t p) { return (seen[p >> 6] >> (p & 63) & 1u) != 0; };
        std::vector<std::uint32_t> todo;
        auto walk = [&](std::uint32_t from) {
            if (is_seen(from)) return;
            seen[from >> 6] |= 1ull << (from & 63);
            todo.assign(1, from);
            while (!todo.empty()) {
                const std::uint32_t *l = links(todo.back(), 0);
                todo.pop_back();
                for (std::uint32_t i = 1; i <= l[0]; ++i)
                    if (!is_seen(l[i])) { seen[l[i] >> 6] |= 1ull << (l[i] & 63); todo.push_back(l[i]); }
            }
        };
        auto append = [&](std::uint32_t to, std::uint32_t p) {
            std::uint32_t *l = links(to, 0);
            l[1 + l[0]] = p;
            ++l[0];
        };
        std::uint64_t linked = 0;
        for (int pass = 0; pass < 16; ++pass) {
            std::fill(seen.begin(), seen.end(), 0);
            walk(entry_);
            std::uint64_t made = 0;
            std::uint32_t last = HNSW_NO_UPPER;
            for (std::uint32_t p = 0; p < n; ++p) {
                if (is_seen(p) || s_.dead(p)) continue;
                // A search from the entry point only visits reachable nodes.
                const float *q = s_.vector(p);
                search_layer(q, Cand{distance_key(s_.metric, q, s_.vector(entry_), s_.row_stride), entry_}, 0, w, false);
                std::sort(w.top.begin(), w.top.end(), nearer);
                const Cand *room = nullptr, *nearest = nullptr;
                for (const Cand &c : w.top) {
                    if (c.pos == p || !is_seen(c.pos)) continue;
                    if (!nearest) nearest = &c;
                    if (links(c.pos, 0)[0] < m0_) { room = &c; break; }
                }
                if (!nearest) continue;
                if (room) {
                    append(room->pos, p);
                } else if (last != HNSW_NO_UPPER && is_seen(last) && links(last, 0)[0] < m0_ &&
                           !(w.top.back().key < key(last, p))) {
                    append(last, p);
                } else {
                    std::uint32_t *l = links(nearest->pos, 0);
                    std::uint32_t far = 1;
                    float far_key = key(nearest->pos, l[1]);
                    for (std::uint32_t i = 2; i <= l[0]; ++i) {
                        const float k = key(nearest->pos, l[i]);
                        if (far_key < k) { far_key = k; far = i; }
                    }
                    l[far] = p;
                }
                ++made;
                last = p;
                walk(p);
            }
            linked += made;
            if (made == 0) break;
        }
        return linked;
    }

    std::uint32_t entry() const { return entry_; }
    std::uint32_t max_level() const { return static_cast<std::uint32_t>(std::max(max_level_, 0)); }

private:
    static std::uint32_t stripes_for(std::uint64_t count)
    {
        std::uint32_t n = 1;
        while (n < 65536 && n < count) n <<= 1;
        return n;
    }

    std::uint32_t *links(std::uint32_t pos, std::uint32_t level)
    {
        return level == 0 ? level0_ + std::uint64_t(pos) * (m0_ + 1)
                          : upper_ + (std::uint64_t(upper_index_[pos]) + level - 1) * (m_ + 1);
    }

    std::mutex &lock_of(std::uint32_t pos) { return stripes_[pos & (stripes_.size() - 1)]; }

    // Copies the links of pos on a level into out, under the lock of pos.
    void copy_links(std::uint32_t pos, std::uint32_t level, std::vector<std::uint32_t> &out)
    {
        std::lock_guard<std::mutex> hold(lock_of(pos));
        const std::uint32_t *l = links(pos, level);
        out.assign(l + 1, l + 1 + l[0]);
    }

    float key(std::uint32_t a, std::uint32_t b) const
    {
        return distance_key(s_.metric, s_.vector(a), s_.vector(b), s_.row_stride);
    }

    // Moves to the nearest neighbour on a level until no neighbour is nearer (ef = 1).
    Cand greedy(const float *q, Cand cur, std::uint32_t level, Worker &w)
    {
        for (;;) {
            copy_links(cur.pos, level, w.fresh);
            score_fresh(s_, q, w);
            Cand best = cur;
            for (std::size_t i = 0; i < w.fresh.size(); ++i) {
                const Cand x{w.keys[i], w.fresh[i]};
                if (nearer(x, best)) best = x;
            }
            if (best.pos == cur.pos) return cur;
            cur = best;
        }
    }

    // Beam search of one level from ep with ef_construction: w.top holds the nearest found; with
    // live_only, tombstoned nodes are passed through but not kept in w.top. self (the position being
    // inserted) is never kept: a defence, the two-phase insert makes it unreachable anyway.
    void search_layer(const float *q, Cand ep, std::uint32_t level, Worker &w, bool live_only,
                      std::uint32_t self = HNSW_NO_UPPER)
    {
        Visited &seen = *w.visited;
        seen.next();
        if (self != HNSW_NO_UPPER) seen.test_set(self);
        seen.test_set(ep.pos);
        w.top.clear();
        if (!(live_only && s_.dead(ep.pos)) && ep.pos != self) w.top.push_back(ep);
        w.cand.assign(1, ep);
        while (!w.cand.empty()) {
            const Cand c = w.cand.front();
            if (w.top.size() >= efc_ && nearer(w.top.front(), c)) break;
            std::pop_heap(w.cand.begin(), w.cand.end(), NearerTop());
            w.cand.pop_back();
            copy_links(c.pos, level, w.links);
            w.fresh.clear();
            for (std::uint32_t x : w.links)
                if (!seen.test_set(x)) w.fresh.push_back(x);
            score_fresh(s_, q, w);
            for (std::size_t i = 0; i < w.fresh.size(); ++i) {
                const Cand x{w.keys[i], w.fresh[i]};
                if (x.key != x.key) continue;
                if (w.top.size() < efc_ || nearer(x, w.top.front())) {
                    w.cand.push_back(x);
                    std::push_heap(w.cand.begin(), w.cand.end(), NearerTop());
                    if (live_only && s_.dead(x.pos)) continue;
                    w.top.push_back(x);
                    std::push_heap(w.top.begin(), w.top.end(), FartherTop());
                    if (w.top.size() > efc_) {
                        std::pop_heap(w.top.begin(), w.top.end(), FartherTop());
                        w.top.pop_back();
                    }
                }
            }
        }
    }

    // The neighbour selection heuristic (Algorithm 4 of the paper, as hnswlib: no extension of the
    // candidates, pruned candidates are not kept): walking the candidates nearest first, keep one
    // when it is nearer to the base than to every candidate kept before it; at most max_n.
    // c is sorted nearest first and its keys are distances to the base.
    void select(const std::vector<Cand> &c, std::uint32_t max_n, std::vector<Cand> &out) const
    {
        out.clear();
        if (c.size() < max_n) { out = c; return; }
        for (const Cand &x : c) {
            if (out.size() >= max_n) break;
            bool keep = true;
            for (const Cand &r : out)
                if (key(x.pos, r.pos) < x.key) { keep = false; break; }
            if (keep) out.push_back(x);
        }
    }

    // Links p to the selected neighbours sel on a level, and each of them back to p. A full list
    // is shrunk with the heuristic. sel never holds p, and no list holds a position twice.
    void connect(std::uint32_t p, std::uint32_t level, const std::vector<Cand> &sel, Worker &w)
    {
        const std::uint32_t cap = level == 0 ? m0_ : m_;
        {
            // With the two-phase insert no other thread can link to p before this, so p's list is
            // empty here. Links found anyway are kept, not overwritten (a defence).
            std::lock_guard<std::mutex> hold(lock_of(p));
            std::uint32_t *l = links(p, level);
            const std::uint32_t had = l[0];
            if (had == 0) {
                std::uint32_t n = 0;
                for (const Cand &c : sel)
                    if (c.pos != p) l[1 + n++] = c.pos;
                l[0] = n;
            } else {
                w.shrink.clear();
                for (const Cand &c : sel)
                    if (c.pos != p) w.shrink.push_back(c);
                for (std::uint32_t i = 0; i < had; ++i) {
                    const std::uint32_t x = l[1 + i];
                    bool dup = x == p;
                    for (const Cand &c : sel) dup |= c.pos == x;
                    if (!dup) w.shrink.push_back(Cand{key(p, x), x});
                }
                std::sort(w.shrink.begin(), w.shrink.end(), nearer);
                if (w.shrink.size() > cap) select(w.shrink, cap, w.cand);
                else w.cand.assign(w.shrink.begin(), w.shrink.end());
                l[0] = static_cast<std::uint32_t>(w.cand.size());
                for (std::size_t i = 0; i < w.cand.size(); ++i) l[1 + i] = w.cand[i].pos;
            }
        }
        for (const Cand &nb : sel) {
            if (nb.pos == p) continue;
            std::lock_guard<std::mutex> hold(lock_of(nb.pos));
            std::uint32_t *l = links(nb.pos, level);
            const std::uint32_t n = l[0];
            bool linked = false;
            for (std::uint32_t i = 1; i <= n; ++i) linked |= l[i] == p;
            if (linked) continue;
            if (n < cap) {
                l[1 + n] = p;
                l[0] = n + 1;
                continue;
            }
            // Keys are symmetric bit for bit (kernels.h), so key(nb, p) = nb.key.
            w.shrink.clear();
            w.shrink.push_back(Cand{nb.key, p});
            for (std::uint32_t i = 0; i < n; ++i) w.shrink.push_back(Cand{key(nb.pos, l[1 + i]), l[1 + i]});
            std::sort(w.shrink.begin(), w.shrink.end(), nearer);
            select(w.shrink, cap, w.cand);           // w.cand is free here
            l[0] = static_cast<std::uint32_t>(w.cand.size());
            for (std::size_t i = 0; i < w.cand.size(); ++i) l[1 + i] = w.cand[i].pos;
        }
    }

    const VectorSet &s_;
    const std::uint32_t m_, m0_, efc_;
    std::vector<std::mutex> stripes_;
    std::mutex global_;
    std::uint32_t entry_ = 0;
    int max_level_ = -1;
    std::uint8_t *levels_ = nullptr;
    std::uint32_t *level0_ = nullptr, *upper_index_ = nullptr, *upper_ = nullptr;
};

void graph_fail(const std::string &why) { throw std::runtime_error("bad snapshot graph: " + why); }

// ---- search

// How a search scores positions against one query: the float rows, or the sq8 codes (sq8.h).
// one(pos) is the key of a position; fresh(w) puts the keys of the positions in w.fresh into w.keys.
struct FloatScorer {
    const VectorSet &set;
    const float *q;
    float one(std::uint32_t pos) const { return distance_key(set.metric, q, set.vector(pos), set.row_stride); }
    void fresh(Worker &w) const { score_fresh(set, q, w); }
};

struct CodeScorer {
    Metric metric;
    const Sq8Codes &c;
    const std::uint8_t *q;
    std::uint32_t q_sum;
    float one(std::uint32_t pos) const
    {
        return sq8_key(metric, c.range, sq8_sum(metric, c.row(pos), q, c.stride), c.sums[pos], q_sum, c.dims);
    }
    void fresh(Worker &w) const
    {
        const std::uint32_t n = static_cast<std::uint32_t>(w.fresh.size());
        w.sums.resize(n);
        w.keys.resize(n);
        sq8_sums_gather(metric, c.codes, c.stride, w.fresh.data(), n, q, w.sums.data());
        for (std::uint32_t i = 0; i < n; ++i) w.keys[i] = sq8_key(metric, c.range, w.sums[i], c.sums[w.fresh[i]], q_sum, c.dims);
    }
};

// Searches one query: the up to ef nearest allowed positions end up in w.top (a heap).
template <class Scorer>
void search_one(const HnswGraph &g, const Scorer &sc, std::uint32_t ef, const std::uint64_t *skip, Worker &w)
{
    w.top.clear();
    if (g.count == 0) return;
    Cand cur{sc.one(g.entry_point), g.entry_point};
    for (std::uint32_t level = g.max_level; level >= 1; --level) {
        for (;;) {
            const std::uint32_t *l = g.links(cur.pos, level);
            w.fresh.assign(l + 1, l + 1 + l[0]);
            sc.fresh(w);
            Cand best = cur;
            for (std::size_t i = 0; i < w.fresh.size(); ++i) {
                const Cand x{w.keys[i], w.fresh[i]};
                if (nearer(x, best)) best = x;
            }
            if (best.pos == cur.pos) break;
            cur = best;
        }
    }

    auto allowed = [skip](std::uint32_t pos) { return !skip || !(skip[pos >> 6] >> (pos & 63) & 1u); };
    Visited &seen = *w.visited;
    seen.next();
    seen.test_set(cur.pos);
    w.cand.assign(1, cur);
    if (allowed(cur.pos) && cur.key == cur.key) w.top.push_back(cur);
    while (!w.cand.empty()) {
        const Cand c = w.cand.front();
        if (w.top.size() >= ef && nearer(w.top.front(), c)) break;
        std::pop_heap(w.cand.begin(), w.cand.end(), NearerTop());
        w.cand.pop_back();
        const std::uint32_t *l = g.links(c.pos, 0);
        const std::uint32_t n = l[0];
        for (std::uint32_t i = 1; i <= n; ++i) seen.prefetch(l[i]);
        w.fresh.clear();
        for (std::uint32_t i = 1; i <= n; ++i)
            if (!seen.test_set(l[i])) w.fresh.push_back(l[i]);
        sc.fresh(w);
        for (std::size_t i = 0; i < w.fresh.size(); ++i) {
            const Cand x{w.keys[i], w.fresh[i]};
            if (x.key != x.key) continue;
            if (w.top.size() < ef || nearer(x, w.top.front())) {
                w.cand.push_back(x);
                std::push_heap(w.cand.begin(), w.cand.end(), NearerTop());
                if (allowed(x.pos)) {
                    w.top.push_back(x);
                    std::push_heap(w.top.begin(), w.top.end(), FartherTop());
                    if (w.top.size() > ef) {
                        std::pop_heap(w.top.begin(), w.top.end(), FartherTop());
                        w.top.pop_back();
                    }
                }
            }
        }
    }
}

// One query, the up to max(ef, k) nearest allowed positions into w.top. With a radius (range
// search) and ef below k, the walk starts with ef and grows it four times, up to k, while more than a
// quarter of the candidate list is within the radius: the list stays at least four times the answer,
// and the work follows the number of vectors within the radius, not k. Every step walks again from
// the start (a walk cannot be resumed with a larger list: it drops what did not fit), so the steps
// before the last cost at most a third more. The walk stops growing when the list is not full,
// because then every reachable position was seen.
template <class Scorer>
void walk(const FlatSearch &s, const HnswGraph &g, const Scorer &sc, std::uint32_t ef, std::uint32_t k,
          const std::uint64_t *skip, Worker &w)
{
    if (!s.has_radius || ef >= k) { search_one(g, sc, std::max(ef, k), skip, w); return; }
    for (std::uint32_t e = std::max<std::uint32_t>(ef, 1);; e = static_cast<std::uint32_t>(std::min<std::uint64_t>(4ull * e, k))) {
        search_one(g, sc, e, skip, w);
        if (e >= k || w.top.size() < e) return;
        std::uint64_t within = 0;
        for (const Cand &c : w.top) within += within_radius(s, c.key);
        if (4 * within <= e) return;
    }
}

} // namespace

std::uint32_t hnsw_level(std::int64_t id, std::uint32_t m, std::uint64_t seed)
{
    // splitmix64 of the id and the seed
    std::uint64_t z = static_cast<std::uint64_t>(id) + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const double u = static_cast<double>((z >> 11) + 1) * 0x1.0p-53;     // (0, 1]
    const double level = -std::log(u) / std::log(static_cast<double>(m));
    return level >= HNSW_MAX_LEVEL ? HNSW_MAX_LEVEL : static_cast<std::uint32_t>(level);
}

std::uint64_t hnsw_upper_capacity(std::uint64_t blocks, std::uint64_t slack, std::uint32_t m)
{
    if (slack == 0) return blocks;
    // The expected level sum of slack positions is slack / (m - 1): twice that, and at least 64.
    const std::uint64_t room = std::max<std::uint64_t>(64, 2 * slack / (std::uint64_t(m) - 1));
    return std::min<std::uint64_t>(blocks + room, HNSW_NO_UPPER - 1);
}

std::uint64_t hnsw_section_bytes(const std::int64_t *ids, std::uint64_t n, std::uint32_t m, std::uint64_t capacity)
{
    check_m(m);
    if (capacity < n) throw std::logic_error("hnsw_section_bytes: capacity below the count");
    const std::uint64_t blocks = levels_of(ids, n, m, nullptr, nullptr, 0);
    if (blocks >= HNSW_NO_UPPER) throw std::runtime_error("too many vectors for an HNSW graph with m = " + std::to_string(m));
    return graph_layout(capacity, m, hnsw_upper_capacity(blocks, capacity - n, m)).bytes;
}

// The header field unreachable1 of the graph just built into section (its layout; h is the header
// without the field): the live positions no search can reach (reach.h) plus 1, or 0 when this build
// does not count them: reachability off, or auto on an incremental build of HNSW_REACH_AUTO_LIMIT
// positions or more.
static std::uint64_t unreachable_field(const VectorSet &s, const std::uint8_t *section, const GraphLayout &layout,
                                       const HnswHeader &h, const HnswParams &p, bool incremental,
                                       const std::function<bool()> &poll)
{
    if (p.reachability == Reachability::Off) return 0;
    if (p.reachability == Reachability::Auto && incremental && h.count >= HNSW_REACH_AUTO_LIMIT) return 0;
    HnswGraph g;
    g.m = h.m;
    g.m0 = h.m0;
    g.max_level = h.max_level;
    g.entry_point = h.entry_point;
    g.count = h.count;
    g.upper_blocks = h.upper_blocks;
    g.levels = section + layout.levels;
    g.level0 = reinterpret_cast<const std::uint32_t *>(section + layout.level0);
    g.upper_index = reinterpret_cast<const std::uint32_t *>(section + layout.upper_index);
    g.upper = reinterpret_cast<const std::uint32_t *>(section + layout.upper);
    return hnsw_unreachable(s, g, p.threads, poll) + 1;
}

void hnsw_build(const VectorSet &s, std::uint8_t *section, const HnswParams &p, const std::function<bool()> &poll)
{
    check_m(p.m);
    if (p.ef_construction < 1) throw std::runtime_error("ef_construction must be at least 1");
    if (s.count == 0) throw std::runtime_error("no vectors");
    const std::uint64_t n = s.count;

    // Levels and the upper part's index first: they fix the layout before the first insert.
    std::vector<std::uint32_t> first(n);
    std::vector<std::uint8_t> levels(n);
    const std::uint64_t blocks = levels_of(s.ids, n, p.m, levels.data(), first.data(), 0);
    const std::uint64_t upper_cap = hnsw_upper_capacity(blocks, s.capacity - n, p.m);
    const GraphLayout layout = graph_layout(s.capacity, p.m, upper_cap);
    std::memcpy(section + layout.levels, levels.data(), n);
    std::memcpy(section + layout.upper_index, first.data(), n * 4);
    std::vector<std::uint8_t>().swap(levels);
    std::vector<std::uint32_t>().swap(first);

    Builder b(s, section, layout, p);
    const int threads = std::max(1, std::min<int>(p.threads, static_cast<int>(std::min<std::uint64_t>(n, 64))));
    std::vector<Worker> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) workers.emplace_back(n);
    b.insert(0, workers[0]);
    if (n > 1)
        parallel_ranges(n - 1, 64, threads, [&](int t, std::uint64_t, std::uint64_t r0, std::uint64_t r1) {
            for (std::uint64_t i = r0; i < r1; ++i) b.insert(static_cast<std::uint32_t>(i + 1), workers[t]);
        }, poll);
    if (poll && poll()) throw Cancelled();
    b.repair(workers[0]);

    HnswHeader h;
    std::memset(&h, 0, sizeof(h));
    h.m = p.m;
    h.m0 = 2 * p.m;
    h.ef_construction = p.ef_construction;
    h.max_level = b.max_level();
    h.entry_point = b.entry();
    h.count = n;
    h.level_seed = HNSW_LEVEL_SEED;
    h.upper_blocks = blocks;
    h.upper_capacity = s.has_capacity() ? upper_cap : 0;
    h.unreachable1 = unreachable_field(s, section, layout, h, p, false, poll);
    std::memcpy(section, &h, sizeof(h));
}

// The upper part of an extended graph: the base's room when the base's layout is kept (it fits),
// else room for the slack of the new layout. Used by hnsw_extended_bytes and hnsw_extend alike.
static std::uint64_t extended_upper_capacity(const HnswGraph &base, std::uint64_t blocks, std::uint64_t n_new, std::uint64_t capacity)
{
    const std::uint64_t n = base.count + n_new;
    if (capacity < n) throw std::logic_error("hnsw_extend: capacity below the count");
    if (blocks >= HNSW_NO_UPPER) throw std::runtime_error("too many vectors for an HNSW graph with m = " + std::to_string(base.m));
    if (capacity == base.capacity && blocks <= base.upper_capacity) return base.upper_capacity;
    return hnsw_upper_capacity(blocks, capacity - n, base.m);
}

std::uint64_t hnsw_extended_bytes(const HnswGraph &base, const std::int64_t *new_ids, std::uint64_t n_new, std::uint64_t capacity)
{
    const std::uint64_t blocks = levels_of(new_ids, n_new, base.m, nullptr, nullptr, base.upper_blocks);
    return graph_layout(capacity, base.m, extended_upper_capacity(base, blocks, n_new, capacity)).bytes;
}

bool hnsw_fits_in_place(const HnswGraph &base, const std::int64_t *new_ids, std::uint64_t n_new, std::uint64_t capacity)
{
    if (capacity != base.capacity || base.count + n_new > capacity) return false;
    const std::uint64_t blocks = levels_of(new_ids, n_new, base.m, nullptr, nullptr, base.upper_blocks);
    return blocks <= base.upper_capacity;
}

void hnsw_extend(const HnswGraph &base, const VectorSet &s, std::uint8_t *section, const HnswParams &p,
                 const std::function<bool()> &poll, bool in_place)
{
    if (p.m != base.m) throw std::runtime_error("m is " + std::to_string(p.m) + ", the base graph has " + std::to_string(base.m));
    if (s.count < base.count) throw std::logic_error("hnsw_extend: fewer positions than the base graph");
    const std::uint64_t n0 = base.count, n = s.count;

    // The base parts at the same place in the new layout, then the levels of the new positions.
    std::vector<std::uint32_t> first(n - n0);
    std::vector<std::uint8_t> levels(n - n0);
    const std::uint64_t blocks = levels_of(s.ids + n0, n - n0, p.m, levels.data(), first.data(), base.upper_blocks);
    const std::uint64_t upper_cap = extended_upper_capacity(base, blocks, n - n0, s.capacity);
    if (in_place && (s.capacity != base.capacity || upper_cap != base.upper_capacity))
        throw std::logic_error("hnsw_extend: the base layout has no room");
    const GraphLayout layout = graph_layout(s.capacity, p.m, upper_cap);
    if (!in_place) {
        std::memcpy(section + layout.levels, base.levels, n0);
        std::memcpy(section + layout.level0, base.level0, n0 * (2 * std::uint64_t(p.m) + 1) * 4);
        std::memcpy(section + layout.upper_index, base.upper_index, n0 * 4);
        std::memcpy(section + layout.upper, base.upper, base.upper_blocks * (std::uint64_t(p.m) + 1) * 4);
    }
    std::memcpy(section + layout.levels + n0, levels.data(), n - n0);
    std::memcpy(section + layout.upper_index + n0 * 4, first.data(), (n - n0) * 4);

    Builder b(s, section, layout, p);
    // A tombstoned entry point costs every search a hop through a dead node: a live node of the same
    // top level takes its place when there is one (the top level itself cannot change).
    std::uint32_t entry = base.entry_point;
    if (s.dead(entry))
        for (std::uint64_t i = 0; i < n0; ++i)
            if (base.levels[i] == base.max_level && !s.dead(i)) { entry = static_cast<std::uint32_t>(i); break; }
    b.start_from(entry, base.max_level);
    const int threads = std::max(1, std::min<int>(p.threads, static_cast<int>(std::min<std::uint64_t>(std::max<std::uint64_t>(n - n0, 1), 64))));
    std::vector<Worker> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) workers.emplace_back(n);
    if (n > n0)
        parallel_ranges(n - n0, 64, threads, [&](int t, std::uint64_t, std::uint64_t r0, std::uint64_t r1) {
            for (std::uint64_t i = r0; i < r1; ++i) b.insert(static_cast<std::uint32_t>(n0 + i), workers[t]);
        }, poll);
    if (poll && poll()) throw Cancelled();
    // Shrunk link lists can leave a node unreachable, as in a full build.
    b.repair(workers[0]);

    HnswHeader h;
    std::memset(&h, 0, sizeof(h));
    h.m = p.m;
    h.m0 = 2 * p.m;
    h.ef_construction = p.ef_construction;
    h.max_level = b.max_level();
    h.entry_point = b.entry();
    h.count = n;
    h.level_seed = HNSW_LEVEL_SEED;
    h.upper_blocks = blocks;
    h.upper_capacity = s.has_capacity() ? upper_cap : 0;
    h.unreachable1 = unreachable_field(s, section, layout, h, p, true, poll);
    std::memcpy(section, &h, sizeof(h));
}

GraphSection hnsw_graph_section(const HnswParams &p, const std::function<bool()> &poll)
{
    GraphSection g;
    const std::uint32_t m = p.m;
    g.bytes = [m](const std::int64_t *ids, std::uint64_t n, std::uint64_t capacity) { return hnsw_section_bytes(ids, n, m, capacity); };
    g.fill = [p, poll](const VectorSet &s, std::uint8_t *section) { hnsw_build(s, section, p, poll); };
    return g;
}

HnswGraph hnsw_open(const VectorSet &s, bool verify)
{
    if (!s.has_graph() || !s.graph) graph_fail("the snapshot has no graph section");
    if (s.graph_bytes < HNSW_HEADER_BYTES) graph_fail("shorter than its header");
    HnswHeader h;
    std::memcpy(&h, s.graph, sizeof(h));
    if (h.m < HNSW_MIN_M || h.m > HNSW_MAX_M || h.m0 != 2 * h.m) graph_fail("m out of range");
    if (h.count != s.count) graph_fail("count differs from the snapshot");
    if (h.count == 0 || h.entry_point >= h.count) graph_fail("entry point outside the vectors");
    if (h.max_level > HNSW_MAX_LEVEL) graph_fail("max_level out of range");
    if (h.upper_blocks >= HNSW_NO_UPPER) graph_fail("too many upper blocks");
    if (h.reserved0) graph_fail("a reserved header field is not 0");
    if (h.unreachable1 > h.count) graph_fail("the unreachable count is above the count");
    if (s.has_capacity() ? h.upper_capacity < h.upper_blocks || h.upper_capacity >= HNSW_NO_UPPER : h.upper_capacity != 0)
        graph_fail("upper_capacity does not fit the snapshot");
    const std::uint64_t upper_cap = s.has_capacity() ? h.upper_capacity : h.upper_blocks;
    const GraphLayout l = graph_layout(s.capacity, h.m, upper_cap);
    if (l.bytes != s.graph_bytes) graph_fail("section size does not match its header");

    HnswGraph g;
    g.m = h.m;
    g.m0 = h.m0;
    g.ef_construction = h.ef_construction;
    g.max_level = h.max_level;
    g.entry_point = h.entry_point;
    g.count = h.count;
    g.upper_blocks = h.upper_blocks;
    g.capacity = s.capacity;
    g.upper_capacity = upper_cap;
    g.counted = h.unreachable1 != 0;
    g.unreachable = g.counted ? h.unreachable1 - 1 : 0;
    g.levels = s.graph + l.levels;
    g.level0 = reinterpret_cast<const std::uint32_t *>(s.graph + l.level0);
    g.upper_index = reinterpret_cast<const std::uint32_t *>(s.graph + l.upper_index);
    g.upper = reinterpret_cast<const std::uint32_t *>(s.graph + l.upper);
    if (g.levels[g.entry_point] != g.max_level) graph_fail("the entry point is not on the top level");
    if (!verify) return g;

    // The levels and the upper index in one pass (a running sum), the links of every position on
    // several threads (a 100M graph has 13 GB of links).
    std::uint64_t blocks = 0;
    for (std::uint64_t i = 0; i < g.count; ++i) {
        const std::uint32_t level = g.levels[i];
        if (level > g.max_level) graph_fail("a level above max_level");
        if (level == 0 ? g.upper_index[i] != HNSW_NO_UPPER : g.upper_index[i] != blocks) graph_fail("upper_index is not consistent with the levels");
        blocks += level;
    }
    if (blocks != g.upper_blocks) graph_fail("upper_blocks does not match the levels");
    const int threads = g.count < 1000000 ? 1 : std::min(8, resolve_threads(0));
    parallel_ranges(g.count, 65536, threads, [&](int, std::uint64_t, std::uint64_t p0, std::uint64_t p1) {
        for (std::uint64_t i = p0; i < p1; ++i) {
            const std::uint32_t level = g.levels[i];
            for (std::uint32_t lv = 0; lv <= level; ++lv) {
                const std::uint32_t *x = g.links(static_cast<std::uint32_t>(i), lv);
                if (x[0] > (lv == 0 ? g.m0 : g.m)) graph_fail("a link list is too long");
                for (std::uint32_t j = 1; j <= x[0]; ++j)
                    if (x[j] >= g.count || x[j] == i || g.levels[x[j]] < lv) graph_fail("a link points to a wrong position");
                // Lists are short (at most 2m) and in cache: a pairwise check is cheaper than a lookup table.
                std::uint32_t dup = 0;
                for (std::uint32_t j = 2; j <= x[0]; ++j)
                    for (std::uint32_t k = 1; k < j; ++k) dup |= x[k] == x[j];
                if (dup) graph_fail("a link list holds a position twice");
            }
        }
    });
    return g;
}

void hnsw_search(const FlatSearch &s, const VectorSet &set, const HnswGraph &g, std::uint32_t ef,
                 const std::uint64_t *skip, const RowBlock *extra, std::vector<Neighbor> &out,
                 std::vector<std::uint32_t> &count, const std::function<bool()> &poll)
{
    const std::uint64_t nq = s.n_queries, k = s.k;
    out.assign(nq * k, Neighbor{0, 0});
    count.assign(nq, 0);
    if (nq == 0 || k == 0) return;

    const int threads = static_cast<int>(std::min<std::uint64_t>(std::max(1, s.threads), nq));
    std::vector<Worker> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) workers.emplace_back(g.count);
    const std::uint64_t per_unit = std::max<std::uint64_t>(1, std::min<std::uint64_t>(16, nq / (4 * std::uint64_t(threads))));
    std::vector<std::vector<Neighbor>> sorted(threads);
    parallel_ranges(nq, per_unit, threads, [&](int t, std::uint64_t, std::uint64_t q0, std::uint64_t q1) {
        Worker &w = workers[t];
        std::vector<Neighbor> &found = sorted[t];
        for (std::uint64_t q = q0; q < q1; ++q) {
            walk(s, g, FloatScorer{set, s.queries + q * s.stride}, ef, s.k, skip, w);
            found.clear();
            for (const Cand &c : w.top) found.push_back(Neighbor{c.key, set.ids[c.pos]});
            std::sort(found.begin(), found.end(), closer);
            std::uint32_t n = 0;
            for (const Neighbor &nb : found) {
                if (n == k) break;
                if (s.has_radius && !within_radius(s, nb.key)) continue;
                out[q * k + n++] = nb;
            }
            count[q] = n;
        }
    }, poll);

    if (!extra || extra->n == 0) return;
    merge_block(s, extra, out, count, poll);
}

void hnsw_search_codes(const FlatSearch &s, const VectorSet &set, const Sq8Codes &codes, const HnswGraph &g,
                       std::uint32_t ef, const std::uint64_t *skip, std::vector<Neighbor> &out,
                       std::vector<std::uint32_t> &count, const std::function<bool()> &poll)
{
    const std::uint64_t nq = s.n_queries, k = s.k;
    out.assign(nq * k, Neighbor{0, 0});
    count.assign(nq, 0);
    if (nq == 0 || k == 0) return;

    const int threads = static_cast<int>(std::min<std::uint64_t>(std::max(1, s.threads), nq));
    std::vector<Worker> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) workers.emplace_back(g.count);
    const std::uint64_t per_unit = std::max<std::uint64_t>(1, std::min<std::uint64_t>(16, nq / (4 * std::uint64_t(threads))));
    std::vector<std::vector<Neighbor>> sorted(threads);
    parallel_ranges(nq, per_unit, threads, [&](int t, std::uint64_t, std::uint64_t q0, std::uint64_t q1) {
        Worker &w = workers[t];
        std::vector<Neighbor> &found = sorted[t];
        for (std::uint64_t q = q0; q < q1; ++q) {
            walk(s, g, CodeScorer{set.metric, codes, s.query_codes + q * codes.stride, s.query_sums[q]}, ef, s.k, skip, w);
            found.clear();
            for (const Cand &c : w.top) found.push_back(Neighbor{c.key, c.pos});
            std::sort(found.begin(), found.end(), closer);
            const std::uint64_t n = std::min<std::uint64_t>(k, found.size());
            std::copy(found.begin(), found.begin() + n, out.begin() + q * k);
            count[q] = static_cast<std::uint32_t>(n);
        }
    }, poll);
}

} // namespace vvector
