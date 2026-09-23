#include "text.h"

#include <cfloat>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(__APPLE__)
#include <xlocale.h>
#else
#include <locale.h>
#endif

namespace vvector {

namespace {

locale_t c_locale()
{
    static const locale_t loc = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(0));
    return loc;
}

[[noreturn]] void bad(const std::string &why, std::size_t at)
{
    throw std::runtime_error("query vector text: " + why + " at character " + std::to_string(at + 1));
}

} // namespace

std::vector<float> parse_vector_text(const char *text, std::size_t len)
{
    const std::string s(text, len);              // strtod needs a terminated string
    std::size_t at = 0;
    auto blanks = [&] { while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\n' || s[at] == '\r')) ++at; };
    std::vector<float> out;
    blanks();
    if (at >= s.size() || s[at] != '[') bad("expected '['", at);
    ++at;
    blanks();
    if (at < s.size() && s[at] == ']') bad("the list is empty", at);
    for (;;) {
        blanks();
        const char *begin = s.c_str() + at;
        char *end = nullptr;
        if (at >= s.size() || !(s[at] == '-' || s[at] == '+' || s[at] == '.' || (s[at] >= '0' && s[at] <= '9')))
            bad("expected a number", at);
        const double v = strtod_l(begin, &end, c_locale());
        if (end == begin) bad("expected a number", at);
        const float f = static_cast<float>(v);
        if (!(std::fabs(f) <= FLT_MAX)) bad("the number is not a finite float32 value", at);
        out.push_back(f);
        at += static_cast<std::size_t>(end - begin);
        blanks();
        if (at >= s.size()) bad("expected ',' or ']'", at);
        if (s[at] == ']') { ++at; break; }
        if (s[at] != ',') bad("expected ',' or ']'", at);
        ++at;
    }
    blanks();
    if (at != s.size()) bad("unexpected text after ']'", at);
    return out;
}

} // namespace vvector
