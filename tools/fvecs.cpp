// fvecs: reads the vector files of the TEXMEX corpus (SIFT1M, GIST1M) and of other
// ANN benchmarks, and writes text that Vertica's COPY loads.
//
//   fvecs info FILE                          count and dimensions
//   fvecs rows FILE [--first=N] [--limit=N] [--id_base=N]
//        one line per vector: id|[x1,x2,...]   (ids from id_base, default 0: the position);
//        --first seeks (every vector of a file has the same length)
//   fvecs gt FILE [--k=N]                    ground truth (.ivecs): qid|rank|id, rank from 1
//   fvecs gen --dims=D --limit=N [--first=N] [--seed=S] [--clusters=C] [--queries]
//        generated vectors, same line format, ids first .. first+N-1: a mixture of C Gaussian
//        clusters (centres N(0,1) per element, spread 0.5 around them). Row r depends only on
//        (seed, r), so the rows are the same however a load splits them with --first/--limit.
//        --queries draws from another stream (same centres): query vectors for the base rows.
//
// FILE: .fvecs (float32), .ivecs (int32) or .bvecs (uint8). Each vector is stored as an int32
// d followed by d values, little-endian. Floats are written with 9 significant digits, so the
// text reads back as the same float32.
// COPY: tools/fvecs rows base.fvecs | vsql -c "COPY t (id, vec) FROM STDIN DELIMITER '|' ABORT ON ERROR DIRECT"
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/types.h>

namespace {

[[noreturn]] void die(const std::string &msg)
{
    std::fprintf(stderr, "fvecs: %s\n", msg.c_str());
    std::exit(1);
}

enum class Kind { F32, I32, U8 };

Kind kind_of(const std::string &path)
{
    auto ends = [&](const char *s) { const std::size_t n = std::strlen(s); return path.size() >= n && path.compare(path.size() - n, n, s) == 0; };
    if (ends(".fvecs")) return Kind::F32;
    if (ends(".ivecs")) return Kind::I32;
    if (ends(".bvecs")) return Kind::U8;
    die("file name must end in .fvecs, .ivecs or .bvecs: " + path);
}

struct Reader {
    std::FILE *f = nullptr;
    Kind kind;
    std::string path;
    std::vector<unsigned char> raw;
    explicit Reader(const std::string &p) : kind(kind_of(p)), path(p)
    {
        f = std::fopen(p.c_str(), "rb");
        if (!f) die("cannot open " + p + ": " + std::strerror(errno));
    }
    ~Reader() { if (f) std::fclose(f); }
    std::size_t value_bytes() const { return kind == Kind::U8 ? 1 : 4; }
    // Reads the next vector into out (as double). False at the end of the file.
    bool next(std::vector<double> &out)
    {
        std::int32_t d;
        const std::size_t got = std::fread(&d, 1, 4, f);
        if (got == 0) return false;
        if (got != 4 || d <= 0 || d > 1000000) die("bad vector header in " + path);
        raw.resize(std::size_t(d) * value_bytes());
        if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) die("file ends inside a vector: " + path);
        out.resize(d);
        for (std::int32_t i = 0; i < d; ++i) {
            if (kind == Kind::F32) { float x; std::memcpy(&x, &raw[i * 4], 4); out[i] = x; }
            else if (kind == Kind::I32) { std::int32_t x; std::memcpy(&x, &raw[i * 4], 4); out[i] = x; }
            else out[i] = raw[i];
        }
        return true;
    }
};

long long option(int argc, char **argv, const char *name, long long fallback, int from = 3)
{
    const std::string prefix = std::string("--") + name + "=";
    for (int i = from; i < argc; ++i)
        if (std::strncmp(argv[i], prefix.c_str(), prefix.size()) == 0) {
            char *end = nullptr;
            const long long v = std::strtoll(argv[i] + prefix.size(), &end, 10);
            if (*end || v < 0) die(std::string("bad value for --") + name);
            return v;
        }
    return fallback;
}

void usage()
{
    std::fprintf(stderr,
                 "usage: fvecs info FILE\n"
                 "       fvecs rows FILE [--first=N] [--limit=N] [--id_base=N]\n"
                 "       fvecs gt FILE [--k=N]\n"
                 "       fvecs gen --dims=D --limit=N [--first=N] [--seed=S] [--clusters=C] [--queries]\n");
    std::exit(2);
}

std::uint64_t splitmix(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// A deterministic stream of standard normal numbers (Box-Muller on splitmix64).
struct Normal {
    std::uint64_t state;
    explicit Normal(std::uint64_t seed) : state(splitmix(seed)) {}
    double uniform() { state = splitmix(state); return (double(state >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double next() { return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()); }
};

bool flag(int argc, char **argv, const char *name)
{
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

int generate(int argc, char **argv)
{
    for (int i = 2; i < argc; ++i)
        if (std::strncmp(argv[i], "--", 2) != 0) usage();
    const long long dims = option(argc, argv, "dims", 0, 2), limit = option(argc, argv, "limit", 0, 2);
    const long long first = option(argc, argv, "first", 0, 2), clusters = option(argc, argv, "clusters", 1000, 2);
    const std::uint64_t seed = static_cast<std::uint64_t>(option(argc, argv, "seed", 1, 2));
    const std::uint64_t stream = flag(argc, argv, "--queries") ? 2 : 1;
    if (dims < 1 || dims > 32768) die("gen needs --dims between 1 and 32768");
    if (limit < 1) die("gen needs --limit=N (the number of vectors)");
    if (clusters < 1 || clusters > 100000) die("--clusters must be between 1 and 100000");
    std::vector<double> centres(static_cast<std::size_t>(clusters * dims));
    Normal c(splitmix(seed) ^ 0x63656e7472657300ULL);
    for (double &x : centres) x = c.next();
    static char buf[1 << 16];
    std::setvbuf(stdout, buf, _IOFBF, sizeof(buf));
    for (long long r = first; r < first + limit; ++r) {
        Normal n(splitmix(seed) ^ splitmix(static_cast<std::uint64_t>(r) * 4 + stream));
        const double *centre = &centres[static_cast<std::size_t>((n.state >> 7) % static_cast<std::uint64_t>(clusters) * dims)];
        std::printf("%lld|[", r);
        for (long long i = 0; i < dims; ++i)
            std::printf(i ? ",%.9g" : "%.9g", static_cast<double>(static_cast<float>(centre[i] + 0.5 * n.next())));
        std::fputs("]\n", stdout);
    }
    if (std::fflush(stdout) != 0) die("cannot write the output");
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2 && std::strcmp(argv[1], "gen") == 0) return generate(argc, argv);
    if (argc < 3) usage();
    const std::string cmd = argv[1];
    for (int i = 3; i < argc; ++i)
        if (std::strncmp(argv[i], "--", 2) != 0) usage();
    Reader r(argv[2]);
    std::vector<double> v;
    static char buf[1 << 16];
    std::setvbuf(stdout, buf, _IOFBF, sizeof(buf));

    if (cmd == "info") {
        long long count = 0, dims = -1;
        while (r.next(v)) {
            if (dims < 0) dims = static_cast<long long>(v.size());
            else if (dims != static_cast<long long>(v.size())) die("vectors of different lengths");
            ++count;
        }
        std::printf("%lld vectors of %lld dimensions\n", count, dims);
    } else if (cmd == "rows") {
        const long long first = option(argc, argv, "first", 0), limit = option(argc, argv, "limit", -1);
        const long long id_base = option(argc, argv, "id_base", 0);
        long long pos = 0, written = 0;
        if (first > 0) {   // every vector of a file has the same length: seek to the first one
            std::int32_t d;
            if (std::fread(&d, 1, 4, r.f) != 4 || d <= 0 || d > 1000000) die("bad vector header in " + r.path);
            const long long record = 4 + static_cast<long long>(d) * static_cast<long long>(r.value_bytes());
            if (fseeko(r.f, static_cast<off_t>(first * record), SEEK_SET) != 0) die("cannot seek in " + r.path);
            pos = first;
        }
        while ((limit < 0 || written < limit) && r.next(v)) {
            if (pos++ < first) continue;
            std::printf("%lld|[", id_base + pos - 1);
            for (std::size_t i = 0; i < v.size(); ++i)
                std::printf(i ? ",%.9g" : "%.9g", v[i]);
            std::fputs("]\n", stdout);
            ++written;
        }
    } else if (cmd == "gt") {
        if (r.kind != Kind::I32) die("gt needs an .ivecs file");
        const long long k = option(argc, argv, "k", 100);
        long long q = 0;
        while (r.next(v)) {
            for (long long i = 0; i < k && i < static_cast<long long>(v.size()); ++i)
                std::printf("%lld|%lld|%lld\n", q, i + 1, static_cast<long long>(v[i]));
            ++q;
        }
    } else {
        usage();
    }
    if (std::fflush(stdout) != 0) die("cannot write the output");
    return 0;
}
