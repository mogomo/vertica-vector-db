// vvector engine: vectors written as text, as in the query parameter of vsearch: '[0.1, -2, 3e-5]'.
#ifndef VVECTOR_ENGINE_TEXT_H
#define VVECTOR_ENGINE_TEXT_H

#include <cstddef>
#include <vector>

namespace vvector {

// Parses '[' number (',' number)* ']' with optional blanks around every part. Numbers are read in
// the C locale (decimal point '.') and stored as float32. Throws std::runtime_error that names the
// position of the first problem: a missing bracket or comma, an empty list, text after ']', or a
// value that is not a finite float32.
std::vector<float> parse_vector_text(const char *text, std::size_t len);

} // namespace vvector

#endif
