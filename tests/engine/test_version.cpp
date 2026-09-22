// Engine unit tests. No Vertica needed.
// Checks that the engine headers build and the constants are sane.
#include "../../src/engine/version.h"

#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main()
{
    CHECK(std::strlen(vvector::LIBRARY_VERSION) > 0);
    CHECK(vvector::FORMAT_VERSION >= 1);
    CHECK(std::strlen(vvector::BUILD_FLAGS) > 0);

    if (failures == 0)
        std::printf("test_version: OK\n");
    return failures == 0 ? 0 : 1;
}
