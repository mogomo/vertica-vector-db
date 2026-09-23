# vvector Makefile.
# Compiler flags and link line follow /opt/vertica/sdk/examples/makefile
# (Vertica 26.x): -shared -fPIC, _GLIBCXX_USE_CXX11_ABI=1, Vertica.cpp compiled in.
# Difference: -std=c++17 (the engine needs it; tested with the SDK headers).
# Arrays/Arrays.cpp (array support) is compiled as part of Vertica.cpp.
#
#   make                      build build/libvvector.so
#   make test [DATA_DIR=DIR]  engine unit tests, no Vertica needed; with DATA_DIR also the SIFT1M recall test
#   make tools                build/tools/fvecs (vector files to COPY text)
#   make bench [DATA_DIR=DIR] engine benchmark; with DATA_DIR on SIFT1M (DIR/sift_*.fvecs), else random data
#   make deploy [FENCED=yes|no|mixed]   install the library and functions (default: fenced)
#   make undeploy             remove functions and library
#   make clean

SDK_HOME ?= /opt/vertica/sdk
CXX      ?= g++
OPT      ?= -O3
FENCED   ?= yes

# Same C++ ABI as the Vertica server. Always 1 since Vertica 24.1.
VERTICA_CXX11_ABI ?= 1

BUILD_DIR := build
LIB       := $(BUILD_DIR)/libvvector.so

ENGINE_SRC := $(wildcard src/engine/*.cpp)
ENGINE_HDR := $(wildcard src/engine/*.h)
UDX_SRC    := $(wildcard src/udx/*.cpp)
TEST_SRC   := $(wildcard tests/engine/test_*.cpp)
TEST_BIN   := $(patsubst tests/engine/%.cpp,$(BUILD_DIR)/tests/%,$(TEST_SRC))
BENCH_SRC  := $(wildcard tests/engine/bench_*.cpp)
BENCH_BIN  := $(patsubst tests/engine/%.cpp,$(BUILD_DIR)/tests/%,$(BENCH_SRC))
DATA_DIR   ?=

# Reported by vversion().
BUILD_FLAGS := $(OPT) -ffp-contract=off -std=c++17 $(shell uname -m) $(notdir $(CXX))-$(shell $(CXX) -dumpfullversion 2>/dev/null || $(CXX) -dumpversion)

# -ffp-contract=off: no fused multiply-add, so every node and every thread count computes
# bit-identical scores (see src/engine/kernels.h). Never -ffast-math, never -march=native:
# one .so must run on every node of a cluster.
COMMON_FLAGS := -std=c++17 -g $(OPT) -ffp-contract=off -Wall -pthread -DVVECTOR_BUILD_FLAGS='"$(BUILD_FLAGS)"'
UDX_FLAGS    := $(COMMON_FLAGS) -DNDEBUG -fno-plt -I $(SDK_HOME)/include -Wno-unused-value -shared -fPIC \
                -D_GLIBCXX_USE_CXX11_ABI=$(VERTICA_CXX11_ABI)

.PHONY: all test bench tools deploy undeploy clean

all: $(LIB)

$(LIB): $(UDX_SRC) $(wildcard src/udx/*.h) $(ENGINE_SRC) $(ENGINE_HDR) $(SDK_HOME)/include/Vertica.cpp Makefile
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(UDX_FLAGS) -o $@ $(UDX_SRC) $(ENGINE_SRC) $(SDK_HOME)/include/Vertica.cpp

# Each file in tests/engine is one test program, linked with the engine only.
$(BUILD_DIR)/tests/%: tests/engine/%.cpp $(wildcard tests/engine/*.h) $(ENGINE_SRC) $(ENGINE_HDR) Makefile
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(COMMON_FLAGS) -o $@ $< $(ENGINE_SRC)

test: $(TEST_BIN)
	@for t in $(TEST_BIN); do echo "== $$t"; $$t $(if $(DATA_DIR),--dir=$(DATA_DIR)) || exit 1; done
	@echo "All engine tests passed."

bench: $(BENCH_BIN)
	@for b in $(BENCH_BIN); do echo "== $$b"; $$b $(if $(DATA_DIR),--dir=$(DATA_DIR)) || exit 1; done

tools: $(BUILD_DIR)/tools/fvecs

$(BUILD_DIR)/tools/%: tools/%.cpp Makefile
	@mkdir -p $(BUILD_DIR)/tools
	$(CXX) $(COMMON_FLAGS) -o $@ $<

deploy: $(LIB)
	scripts/deploy.sh --fenced=$(FENCED)

undeploy:
	scripts/deploy.sh --undeploy

clean:
	rm -rf $(BUILD_DIR)
