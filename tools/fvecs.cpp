// fvecs: reads the vector files of the TEXMEX corpus (SIFT1M, GIST1M) and of other
// ANN benchmarks, and writes text that Vertica's COPY loads.
//
//   fvecs info FILE                          count and dimensions
//   fvecs rows FILE [--first=N] [--limit=N] [--id_base=N]
//        one line per vector: id|[x1,x2,...]   (ids from id_base, default 0: the position)
//   fvecs gt FILE [--k=N]                    ground truth (.ivecs): qid|rank|id, rank from 1
//
// FILE: .fvecs (float32), .ivecs (int32) or .bvecs (uint8). Each vector is stored as an int32
// d followed by d values, little-endian. Floats are written with 9 significant digits, so the
// text reads back as the same float32.
// COPY: tools/fvecs rows base.fvecs | vsql -c "COPY t (id, vec) FROM STDIN DELIMITER '|' ABORT ON ERROR DIRECT"
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

long long option(int argc, char **argv, const char *name, long long fallback)
{
    const std::string prefix = std::string("--") + name + "=";
    for (int i = 3; i < argc; ++i)
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
                 "       fvecs gt FILE [--k=N]\n");
    std::exit(2);
}

} // namespace

int main(int argc, char **argv)
{
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
