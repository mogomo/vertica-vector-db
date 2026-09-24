// Snapshot cache: chunked write in any order, ACTIVE flip, cleanup rules, damage detection.
#include "check.h"

#include "../../src/engine/cache.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

using namespace vvector;

static bool exists(const std::string &path) { return access(path.c_str(), F_OK) == 0; }

// Loads the buffer as snapshot `id`, chunks in reverse order. chunk_bytes is small to get many chunks.
static void load(const std::string &dir, const std::string &name, std::int64_t id, const SnapshotBuffer &b,
                 bool drop_last_chunk = false)
{
    CacheWriter w;
    w.begin(dir, name, id);
    const std::int64_t chunks = static_cast<std::int64_t>((b.size() + CHUNK_BYTES - 1) / CHUNK_BYTES);
    for (std::int64_t c = chunks - 1 - (drop_last_chunk ? 1 : 0); c >= 0; --c) {
        const std::uint64_t off = c * CHUNK_BYTES;
        const std::uint64_t len = std::min<std::uint64_t>(CHUNK_BYTES, b.size() - off);
        w.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(b.data()) + off, len);
    }
    w.commit();
}

int main()
{
    char tmpl[] = "/tmp/vvector_test_XXXXXX";
    const std::string dir = mkdtemp(tmpl);

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
    }

    // A load with a missing chunk fails and leaves the cache as it was.
    thrown = false;
    try { load(dir, "g", 2, big.buffer, true); } catch (const std::runtime_error &) { thrown = true; }
    CHECK(thrown);
    CHECK(read_active(dir, "g", active) && active == 1);
    CHECK(!exists(snapshot_path(dir, "g", 2)));

    // Reload of the same snapshot is fine (vload is idempotent).
    load(dir, "g", 1, big.buffer);

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

    std::system(("rm -rf " + dir).c_str());
    return finish("test_cache");
}
