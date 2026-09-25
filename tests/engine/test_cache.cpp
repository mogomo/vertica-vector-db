// Snapshot cache: chunked write in any order, ACTIVE flip, cleanup rules, damage detection.
#include "check.h"

#include "../../src/engine/cache.h"
#include "../../src/engine/delta.h"
#include "../../src/engine/hnsw.h"
#include "../../src/engine/sq8.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace vvector;

static bool exists(const std::string &path) { return access(path.c_str(), F_OK) == 0; }

// Loads the buffer as snapshot `id`, chunks in reverse order. drop_chunk: that chunk is left out
// (a missing piece); skip_zero: all-zero chunks are left out (what vbuild does with the room to grow).
static void load(const std::string &dir, const std::string &name, std::int64_t id, const SnapshotBuffer &b,
                 std::int64_t drop_chunk = -1, bool skip_zero = false)
{
    CacheWriter w;
    w.begin(dir, name, id);
    const std::int64_t chunks = static_cast<std::int64_t>((b.size() + CHUNK_BYTES - 1) / CHUNK_BYTES);
    for (std::int64_t c = chunks - 1; c >= 0; --c) {
        if (c == drop_chunk) continue;
        const std::uint64_t off = c * CHUNK_BYTES;
        const std::uint64_t len = std::min<std::uint64_t>(CHUNK_BYTES, b.size() - off);
        if (skip_zero && c > 0 && all_zero(b.data() + off, len)) continue;
        w.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(b.data()) + off, len);
    }
    w.commit();
}

// The read-ahead plan of a query mapping: sections in the order a search needs them, the float rows
// last, whole sections only within the budget, MADV_RANDOM beyond it (willneed false); what a search
// reads whole is always asked for; a compact index's rows are MADV_RANDOM whatever the budget.
static void test_prewarm_plan()
{
    auto off = [](const std::uint8_t *base, const void *at) {
        return static_cast<std::uint64_t>(static_cast<const std::uint8_t *>(at) - base);
    };
    TestSet flat;
    build(flat, 1000, 8);
    const std::uint8_t *fd = flat.buffer.data();
    const std::uint64_t fsize = flat.buffer.size();
    std::vector<PrewarmRange> plan = prewarm_plan(fd, fsize, flat.set, false);
    CHECK(plan.size() == 2);
    CHECK(plan[0].offset == off(fd, flat.set.ids) && plan[0].bytes == 1000 * 8 && plan[0].willneed);
    CHECK(plan[1].offset == 0 && plan[1].bytes == HEADER_BYTES + 1000ull * flat.set.row_stride * 4 && plan[1].willneed);
    CHECK(plan[1].bytes <= off(fd, flat.set.ids));
    // A flat index without codes reads every row at each search: its rows are asked for whatever the
    // budget; the ids follow the budget (MADV_RANDOM beyond it).
    plan = prewarm_plan(fd, fsize, flat.set, false, 1000 * 8 + 100);
    CHECK(plan.size() == 2 && plan[0].offset == off(fd, flat.set.ids) && plan[0].willneed && plan[1].offset == 0 && plan[1].willneed);
    plan = prewarm_plan(fd, fsize, flat.set, false, 0);
    CHECK(plan.size() == 2 && plan[0].offset == off(fd, flat.set.ids) && !plan[0].willneed && plan[1].offset == 0 && plan[1].willneed);

    // HNSW with sq8 codes: ids, codes, graph, then the rows.
    TestSet coded;
    {
        SnapshotBuilder b(Metric::L2);
        std::vector<float> v(8);
        for (std::uint64_t i = 0; i < 1000; ++i) {
            test_vector(i, 8, 5, true, v.data());
            b.add(static_cast<std::int64_t>(3 + 2 * i), v.data(), 8);
        }
        HnswParams p;
        p.m = 12;
        p.ef_construction = 40;
        p.threads = 1;
        const GraphSection g = hnsw_graph_section(p);
        const CodeSection c = sq8_code_section();
        b.finish(0, coded.buffer, &g, &c);
        coded.set = snapshot_open(coded.buffer.data(), coded.buffer.size(), true);
    }
    const std::uint8_t *cd = coded.buffer.data();
    const std::uint64_t csize = coded.buffer.size();
    const HnswGraph g = hnsw_open(coded.set, false);
    plan = prewarm_plan(cd, csize, coded.set, false);
    CHECK(plan.size() == 10);
    CHECK(plan[0].offset == off(cd, coded.set.ids));
    CHECK(plan[1].offset == off(cd, coded.set.sq8) && plan[1].bytes == SQ8_HEADER_BYTES);
    CHECK(plan[4].offset == off(cd, coded.set.graph) && plan[4].bytes == HNSW_HEADER_BYTES);
    CHECK(plan[6].offset == off(cd, g.level0) && plan[6].bytes == 1000ull * (g.m0 + 1) * 4);
    CHECK(plan[9].offset == 0 && plan[9].willneed);
    std::uint64_t total = 0;
    for (std::size_t i = 0; i + 1 < plan.size(); ++i) {
        CHECK(plan[i].willneed && plan[i].offset < plan[i + 1].offset + (i + 2 == plan.size() ? csize : 0));
        total += plan[i].bytes;
    }
    total += plan[9].bytes;
    CHECK(total <= csize);
    // A budget of everything but level 0 (the largest section): level 0 becomes MADV_RANDOM, the
    // smaller sections after it (upper index, upper levels, the rows) are still asked for.
    CHECK(plan[6].bytes > plan[7].bytes + plan[8].bytes + plan[9].bytes);
    std::vector<PrewarmRange> cut = prewarm_plan(cd, csize, coded.set, false, total - plan[6].bytes);
    CHECK(cut.size() == 10);
    for (std::size_t i = 0; i < cut.size(); ++i) CHECK(cut[i].offset == plan[i].offset && cut[i].willneed == (i != 6));
    // Budget 0 on a graph index: every section MADV_RANDOM.
    cut = prewarm_plan(cd, csize, coded.set, false, 0);
    CHECK(cut.size() == 10);
    for (const PrewarmRange &r : cut) CHECK(!r.willneed);
    // compact: the rows MADV_RANDOM, even with the full budget.
    plan = prewarm_plan(cd, csize, coded.set, true);
    CHECK(plan.size() == 10 && plan[9].offset == 0 && !plan[9].willneed && plan[6].willneed);
    // A flat coded index scans every code: the codes are asked for whatever the budget.
    TestSet fcoded;
    {
        SnapshotBuilder b(Metric::L2);
        std::vector<float> v(8);
        for (std::uint64_t i = 0; i < 500; ++i) {
            test_vector(i, 8, 6, true, v.data());
            b.add(static_cast<std::int64_t>(1 + i), v.data(), 8);
        }
        const CodeSection c = sq8_code_section();
        b.finish(0, fcoded.buffer, nullptr, &c);
        fcoded.set = snapshot_open(fcoded.buffer.data(), fcoded.buffer.size(), true);
    }
    cut = prewarm_plan(fcoded.buffer.data(), fcoded.buffer.size(), fcoded.set, false, 0);
    CHECK(cut.size() == 5 && !cut[0].willneed && cut[1].willneed && cut[2].willneed && cut[3].willneed && !cut[4].willneed);
}

int main()
{
    char tmpl[] = "/tmp/vvector_test_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    test_prewarm_plan();

    CHECK(valid_index_name("docs_2") && !valid_index_name("") && !valid_index_name("../x") &&
          !valid_index_name("a/b") && !valid_index_name("a b"));

    // Big enough for three chunks: 200,000 vectors of 32 floats.
    TestSet big;
    const std::uint64_t n = 200000;
    build(big, n, 32, Order::Ascending, Metric::Dot, 77);
    CHECK(big.buffer.size() > 2 * CHUNK_BYTES);

    std::int64_t active = 0;
    CHECK(!read_active(dir, "g", active));
    bool thrown = false;
    try { MappedSnapshot m; m.open_active(dir, "g"); } catch (const std::runtime_error &) { thrown = true; }
    CHECK(thrown);

    load(dir, "g", 1, big.buffer);
    CHECK(read_active(dir, "g", active) && active == 1);
    {
        MappedSnapshot m;
        m.open_active(dir, "g");
        CHECK(m.snapshot_id() == 1 && m.vectors().max_ver == 77 && m.vectors().count == n && m.vectors().dims == 32);
        CHECK(m.size() == big.buffer.size());
        // Just written and read back by the load: in the page cache.
        CHECK(m.resident_bytes() > 0 && m.resident_bytes() <= m.size());
    }

    // A load with a missing chunk (one with data) fails and leaves the cache as it was.
    thrown = false;
    try { load(dir, "g", 2, big.buffer, 1); } catch (const std::runtime_error &) { thrown = true; }
    CHECK(thrown);
    CHECK(read_active(dir, "g", active) && active == 1);
    CHECK(!exists(snapshot_path(dir, "g", 2)));
    // Without the header piece nothing can be sized.
    CHECK(throws([&] { load(dir, "g", 2, big.buffer, 0); }, "no header received"));

    // Reload of the same snapshot is fine (vload is idempotent).
    load(dir, "g", 1, big.buffer);

    // A whole copy leaves all-zero chunks out (the room to grow); the file is sized from the header
    // and the missing chunks read as zeros (milestone M7).
    {
        SnapshotBuilder sb(Metric::L2);
        sb.set_growth(1000);                    // 10 x the count of room: chunks of zeros in the middle
        std::vector<float> v(32);
        for (std::int64_t i = 0; i < 20000; ++i) { for (float &x : v) x = static_cast<float>(i % 7 + 1); sb.add(i, v.data(), 32); }
        SnapshotBuffer roomy;
        sb.finish(5, roomy);
        std::int64_t zero_chunks = 0;
        for (std::uint64_t off = CHUNK_BYTES; off < roomy.size(); off += CHUNK_BYTES)
            zero_chunks += all_zero(roomy.data() + off, std::min<std::uint64_t>(CHUNK_BYTES, roomy.size() - off));
        CHECK(zero_chunks >= 2);
        load(dir, "roomy", 1, roomy, -1, true);
        MappedSnapshot m;
        m.open(snapshot_path(dir, "roomy", 1), true);
        CHECK(m.size() == roomy.size() && std::memcmp(m.data(), roomy.data(), roomy.size()) == 0);
        CHECK(m.vectors().capacity == 20000 + 200000 && m.vectors().count == 20000);
        std::system(("rm -rf " + dir + "/roomy").c_str());
    }

    // A patch (milestone M7): the pieces are the changed bytes, written over a copy of the base.
    {
        SnapshotBuffer changed;
        changed.allocate(big.buffer.size());
        std::memcpy(changed.data(), big.buffer.data(), big.buffer.size());
        float *row50 = reinterpret_cast<float *>(changed.data() + 256) + 50 * 32;
        row50[3] += 1.0f;                       // one row of the vectors, away from the header's block
        std::int64_t mv = 78;
        std::memcpy(changed.data() + offsetof(SnapshotHeader, max_ver), &mv, 8);
        const std::uint64_t sum = snapshot_checksum(changed.data(), changed.size());
        std::memcpy(changed.data() + offsetof(SnapshotHeader, checksum), &sum, 8);
        std::vector<ByteRange> runs = snapshot_diff(big.buffer.data(), changed.data(), changed.size(), {{0, changed.size()}}, CHUNK_BYTES);
        CHECK(runs.size() == 2 && runs[0].offset == 0 && runs[0].bytes == DIFF_BLOCK && runs[1].offset == (256 + 50 * 128) / DIFF_BLOCK * DIFF_BLOCK);
        auto patch = [&](std::int64_t id, std::int64_t base) {
            CacheWriter w;
            w.begin(dir, "g", id, base);
            for (const ByteRange &r : runs) w.write_at(static_cast<std::int64_t>(r.offset), reinterpret_cast<const char *>(changed.data()) + r.offset, r.bytes);
            return w.commit();
        };
        CHECK(patch(12, 1) == changed.size());
        MappedSnapshot m;
        m.open(snapshot_path(dir, "g", 12), true);
        CHECK(m.vectors().max_ver == 78 && std::memcmp(m.data(), changed.data(), changed.size()) == 0);
        // A missing base, and a base that is not the one the patch was made for (the checksum refuses).
        CHECK(throws([&] { patch(13, 999); }, "base snapshot 999 is not in the cache"));
        TestSet other;
        build(other, 3, 2);
        load(dir, "g", 14, other.buffer);
        CHECK(throws([&] { patch(15, 14); }, "checksum"));
        CHECK(read_active(dir, "g", active) && active == 14 && !exists(snapshot_path(dir, "g", 15)));
        load(dir, "g", 1, big.buffer);          // back to the state the tests below expect
        // clone_file's fallbacks all work: a plain copy through read and write.
        const int from = ::open(snapshot_path(dir, "g", 1).c_str(), O_RDONLY);
        const std::string to_path = dir + "/g/clone.tmp";
        const int to = ::open(to_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        CHECK(from >= 0 && to >= 0);
        const std::string how = clone_file(from, to, to_path);
        struct stat st;
        CHECK(fstat(to, &st) == 0 && static_cast<std::uint64_t>(st.st_size) == big.buffer.size());
        std::printf("  clone_file: %s\n", how.c_str());
        // Without reflink the copy is by read and write, never copy_file_range (a reflink on xfs).
        CHECK(std::string(clone_file(from, to, to_path, false)) == "copy");
        CHECK(fstat(to, &st) == 0 && static_cast<std::uint64_t>(st.st_size) == big.buffer.size());
        ::close(from); ::close(to); ::unlink(to_path.c_str());
    }

    // A load in passes (a large snapshot, vvector.load_on_nodes): each pass writes its chunks into
    // the partial file, the last one verifies and activates it. Chunk c goes in pass c % passes + 1.
    {
        const std::int64_t chunks = static_cast<std::int64_t>((big.buffer.size() + CHUNK_BYTES - 1) / CHUNK_BYTES);
        auto pass = [&](std::int64_t id, int p, int passes, bool drop_last_chunk) {
            CacheWriter w;
            w.begin_part(dir, "g", id, "load7", p > 1);
            for (std::int64_t c = chunks - 1; c >= 0; --c) {
                if (c % passes != p - 1 || (drop_last_chunk && c == chunks - 1)) continue;
                const std::uint64_t off = c * CHUNK_BYTES;
                const std::uint64_t len = std::min<std::uint64_t>(CHUNK_BYTES, big.buffer.size() - off);
                w.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(big.buffer.data()) + off, len);
            }
            if (p < passes) w.keep(); else w.commit();
        };
        const std::string partial = snapshot_path(dir, "g", 9) + ".part.load7";
        pass(9, 1, 3, false);
        pass(9, 2, 3, false);
        CHECK(exists(partial) && read_active(dir, "g", active) && active == 1);   // not active before the last pass
        pass(9, 3, 3, false);
        CHECK(read_active(dir, "g", active) && active == 9 && !exists(partial));
        MappedSnapshot m;              // by path: open_active trusts the ACTIVE it read above for ACTIVE_CHECK_MS
        m.open(snapshot_path(dir, "g", 9), true);
        CHECK(m.vectors().count == n && m.size() == big.buffer.size());
        // A pass that is not the first needs the partial file of the earlier passes.
        CHECK(throws([&] { pass(10, 2, 2, false); }, "cannot continue the partial file"));
        // A chunk missing in any pass: the last pass refuses the file and removes it; ACTIVE stays.
        pass(10, 1, 2, true);
        CHECK(throws([&] { pass(10, 2, 2, true); }));
        CHECK(read_active(dir, "g", active) && active == 9);
        CHECK(!exists(snapshot_path(dir, "g", 10)) && !exists(snapshot_path(dir, "g", 10) + ".part.load7"));
        CacheWriter w;
        CHECK(throws([&] { w.begin_part(dir, "g", 11, "../x", false); }, "part must be 1 to 64 letters and digits"));
    }

    // A foreign file is never removed; old snapshots are: only active and previous stay.
    const std::string foreign = dir + "/g/notes.txt";
    std::ofstream(foreign) << "not a snapshot";
    TestSet small;
    build(small, 3, 2);
    load(dir, "g", 2, small.buffer);
    load(dir, "g", 3, small.buffer);
    CHECK(read_active(dir, "g", active) && active == 3);
    CHECK(!exists(snapshot_path(dir, "g", 1)));
    CHECK(exists(snapshot_path(dir, "g", 2)) && exists(snapshot_path(dir, "g", 3)));
    CHECK(exists(foreign));

    // Only directories with an ACTIVE file are listed.
    std::system(("mkdir -p " + dir + "/not_an_index").c_str());
    CHECK(list_cached_indexes(dir) == std::vector<std::string>({"g"}));

    // Only regular files of this process's user are mapped: a search checks the header of a cache
    // file, not every link, so it must never map a file someone else could have written.
    {
        MappedSnapshot m;
        CHECK(throws([&] { m.open(dir + "/g", false); }, "not a regular file"));
        const std::string fifo = dir + "/g/99.vv";
        CHECK(mkfifo(fifo.c_str(), 0600) == 0);
        CHECK(throws([&] { m.open(fifo, false); }, "not a regular file"));      // and does not block
        ::unlink(fifo.c_str());
        if (geteuid() == 0) {
            const std::string other = dir + "/other";
            std::system(("mkdir -p " + other + "/h && cp " + snapshot_path(dir, "g", 3) + " " + other + "/h/1.vv && echo 1 > " + other + "/h/ACTIVE").c_str());
            CHECK(!throws([&] { m.open(other + "/h/1.vv", false); }));
            CHECK(chown((other + "/h/1.vv").c_str(), 65534, 65534) == 0);
            CHECK(throws([&] { m.open(other + "/h/1.vv", false); }, "not owned by the database's operating system user"));
            CHECK(list_cached_indexes(other) == std::vector<std::string>({"h"}));
            CHECK(chown((other + "/h").c_str(), 65534, 65534) == 0);
            CHECK(list_cached_indexes(other).empty());
            CHECK(throws([&] { m.open_active(other, "h"); }, "the directory is not owned by the database's operating system user"));
        } else {
            std::printf("  owner checks: skipped (they need root to make a file of another user)\n");
        }
    }

    // What ACTIVE says is trusted for ACTIVE_CHECK_MS, unless the caller needs a newer snapshot.
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTIVE_CHECK_MS + 50));
        MappedSnapshot m;
        m.open_active(dir, "g");
        CHECK(m.snapshot_id() == 3);
        load(dir, "g", 4, big.buffer);
        m.open_active(dir, "g");
        CHECK(m.snapshot_id() == 3);                 // within the check interval: no file system work
        m.open_active(dir, "g", 4);
        CHECK(m.snapshot_id() == 4 && m.vectors().count == n);      // needs 4: reads ACTIVE at once
        load(dir, "g", 5, small.buffer);
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTIVE_CHECK_MS + 50));
        MappedSnapshot other;
        other.open_active(dir, "g");
        CHECK(other.snapshot_id() == 5 && other.vectors().count == 3);
        CHECK(m.vectors().count == n);               // the older mapping stays valid while it is used
    }

    // Kept mappings of indexes without queries are given back; one in use stays valid.
    {
        MappedSnapshot in_use;
        in_use.open_active(dir, "g");
        CHECK(release_idle_mappings(0) >= 1);
        const std::int64_t last_id = in_use.vectors().ids[in_use.vectors().count - 1];     // still mapped: no fault
        CHECK(in_use.vectors().count > 0 && last_id > 0);
        CHECK(release_idle_mappings(0) == 0);
        MappedSnapshot again;
        again.open_active(dir, "g");
        CHECK(again.snapshot_id() == in_use.snapshot_id());
    }

    // Index options: written by vconfig, read with the snapshot.
    {
        CHECK(parse_index_options("precision=best, freshness=exact,ef_search=,threads=4").size() == 3);
        CHECK(parse_index_options("").empty());
        CHECK(throws([] { parse_index_options("colour=red"); }, "unknown index option 'colour'"));
        CHECK(throws([] { parse_index_options("threads"); }, "is not name=value"));
        CHECK(throws([] { parse_index_options("precision=../x"); }, "is not valid"));
        write_index_options(dir, "g", parse_index_options("precision=balanced,threads=2"));
        CHECK(read_index_options(dir, "g").at("precision") == "balanced");
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTIVE_CHECK_MS + 50));
        MappedSnapshot m;
        m.open_active(dir, "g");
        CHECK(m.options().size() == 2 && m.options().at("threads") == "2");
        load(dir, "g", 6, small.buffer);            // a load keeps OPTIONS
        CHECK(read_index_options(dir, "g").size() == 2);
        write_index_options(dir, "fresh_index", IndexOptions());        // creates the directory
        CHECK(read_index_options(dir, "fresh_index").empty());
    }

    // The index option cache_dir: OPTIONS in one directory names the directory that holds the index;
    // a query reads ACTIVE, OPTIONS and the snapshot there. One hop, never a chain.
    {
        CHECK(valid_cache_dir("/data/vvector") && valid_cache_dir("/a/.b/c-d_e/") && valid_cache_dir("/tmp/vvector/.alt"));
        CHECK(!valid_cache_dir("") && !valid_cache_dir("/") && !valid_cache_dir("data") && !valid_cache_dir("/a/../b") &&
              !valid_cache_dir("/a/./b") && !valid_cache_dir("/a//b") && !valid_cache_dir("/a b") && !valid_cache_dir("/a/..") &&
              !valid_cache_dir("/a;b") && !valid_cache_dir(std::string(1001, 'a').insert(0, "/")));
        CHECK(parse_index_options("cache_dir=/data/vv,threads=2").at("cache_dir") == "/data/vv");
        CHECK(throws([] { parse_index_options("cache_dir=../x"); }, "index option cache_dir"));
        const std::string home = dir + "/home";
        TestSet small2;
        build(small2, 7, 3);
        load(home, "r", 11, small2.buffer);
        write_index_options(home, "r", parse_index_options("threads=3"));
        write_index_options(dir, "r", parse_index_options("threads=3,cache_dir=" + home));
        CHECK(list_cached_indexes(dir) == std::vector<std::string>({"fresh_index", "g", "r"}));     // an OPTIONS file lists it
        MappedSnapshot m;
        m.open_active(dir, "r");
        CHECK(m.snapshot_id() == 11 && m.vectors().count == 7 && m.path() == snapshot_path(home, "r", 11));
        CHECK(m.options().count("cache_dir") == 0 && m.options().at("threads") == "3");
        // One hop: an OPTIONS file in the home that names a third directory is not followed.
        write_index_options(home, "r", parse_index_options("cache_dir=" + dir + "/third"));
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTIVE_CHECK_MS + 50));
        MappedSnapshot again;
        again.open_active(dir, "r");
        CHECK(again.path() == snapshot_path(home, "r", 11));
        // A redirect to a directory without the index: "no snapshot cache ... in <that directory>".
        write_index_options(dir, "q", parse_index_options("cache_dir=" + dir + "/nowhere"));
        CHECK(throws([&] { MappedSnapshot x; x.open_active(dir, "q"); }, ("no snapshot cache for index 'q' in " + dir + "/nowhere").c_str()));
    }

    std::system(("rm -rf " + dir).c_str());
    return finish("test_cache");
}
