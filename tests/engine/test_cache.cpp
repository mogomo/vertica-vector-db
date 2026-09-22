// Snapshot cache: chunked write in any order, ACTIVE flip, cleanup rules, damage detection.
#include "check.h"

#include "../../src/engine/cache.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
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
    build(big, n, 32, false, Metric::Dot, 77);
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

    CHECK(list_cached_indexes(dir) == std::vector<std::string>({"g"}));

    std::system(("rm -rf " + dir).c_str());
    return finish("test_cache");
}
