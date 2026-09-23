// vvector engine: version constants.
// Pure C++17. No Vertica includes in src/engine.
#ifndef VVECTOR_ENGINE_VERSION_H
#define VVECTOR_ENGINE_VERSION_H

#include <cstdint>

namespace vvector {

// Library version. Change it here only.
constexpr const char *LIBRARY_VERSION = "0.1.0";

// Snapshot binary format version. See docs/format.md.
constexpr std::int32_t FORMAT_VERSION = 2;

// Build flags are passed by the Makefile. The fallback keeps other builds working.
#ifndef VVECTOR_BUILD_FLAGS
#define VVECTOR_BUILD_FLAGS "unknown"
#endif
constexpr const char *BUILD_FLAGS = VVECTOR_BUILD_FLAGS;

} // namespace vvector

#endif
