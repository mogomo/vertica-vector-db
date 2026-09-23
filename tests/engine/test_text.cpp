// Vectors as text: the query parameter of vsearch.
#include "check.h"

#include "../../src/engine/text.h"

#include <clocale>
#include <string>

using namespace vvector;

static std::vector<float> parse(const std::string &s) { return parse_vector_text(s.data(), s.size()); }

int main()
{
    CHECK(parse("[1]") == std::vector<float>({1.0f}));
    CHECK(parse(" [ 0.5 ,-2, 3e-2,+4 , .25 ] ") == std::vector<float>({0.5f, -2.0f, 0.03f, 4.0f, 0.25f}));
    CHECK(parse("[\n1,\t2\r\n]") == std::vector<float>({1.0f, 2.0f}));
    CHECK(parse("[0.1]")[0] == 0.1f);                                    // rounded like a float32 cast
    CHECK(throws([] { parse("1, 2"); }, "expected '[' at character 1"));
    CHECK(throws([] { parse("[]"); }, "the list is empty"));
    CHECK(throws([] { parse("[1, 2"); }, "expected ',' or ']' at character 6"));
    CHECK(throws([] { parse("[1 2]"); }, "expected ',' or ']' at character 4"));
    CHECK(throws([] { parse("[1,,2]"); }, "expected a number at character 4"));
    CHECK(throws([] { parse("[1, abc]"); }, "expected a number at character 5"));
    CHECK(throws([] { parse("[1] x"); }, "unexpected text after ']' at character 5"));
    CHECK(throws([] { parse("[1e39]"); }, "not a finite float32"));
    CHECK(throws([] { parse("[nan]"); }, "expected a number"));
    CHECK(throws([] { parse("[inf]"); }, "expected a number"));
    CHECK(throws([] { parse(""); }, "expected '['"));
    // The server's locale does not matter: '.' is the decimal point.
    if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") || std::setlocale(LC_NUMERIC, "de_DE"))
        CHECK(parse("[1.5]")[0] == 1.5f);
    std::setlocale(LC_NUMERIC, "C");
    // 1536 numbers, as a large embedding.
    std::string big = "[";
    for (int i = 0; i < 1536; ++i) big += (i ? ", " : "") + std::to_string(i) + ".5";
    big += "]";
    const std::vector<float> v = parse(big);
    CHECK(v.size() == 1536 && v[1535] == 1535.5f);
    return finish("test_text");
}
