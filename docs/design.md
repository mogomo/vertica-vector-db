# vvector design notes

This file explains the decisions and records the measurements behind them.
The user's view is in the README.

## Pieces

    vector table --vbuild--> vvector.snapshot --vload--> node cache file --mmap--> vsearch
    (customer)               (segmented, broadcast       (/tmp/vvector/<index>/)
                              to every node by vload)

- `src/engine` is plain C++17 without Vertica includes. `src/udx` holds thin
  Vertica adapters only: they read rows and parameters, call the engine and
  write rows.
- The snapshot table is segmented (one copy per refresh, plus the buddy
  copies of K-safety). It is backed up and protected with the database. The
  cache file on every node is only a copy of it.
- A missing or damaged cache file is never a data loss: run vload again
  (`CALL vvector.load_all('<index>')`).

## Why an index outside SQL

Vertica 26.2 has no vector type and no vector index. Vectors are stored as
`ARRAY[FLOAT]` (or INT, NUMERIC), and the built-in functions
`COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and `VECTOR_MAGNITUDE` compare
two arrays. A nearest-neighbour query is therefore a full scan with a top-k
sort (verified with EXPLAIN: STORAGE ACCESS, then SORT [TOPK]); on SIFT1M
(1,000,000 vectors of 128 dimensions) one query takes 7.3 seconds on the test
VM. vvector keeps a copy of the vectors and an HNSW graph over them as a
snapshot, loads it on every node and answers the same query from it in a few
milliseconds (below).

## Search engine

- Distance kernels (`src/engine/kernels.h`): l2 (squared distance, then one
  square root per result), l1, dot; cosine is dot on unit vectors. Portable
  SIMD with GCC vector extensions, no intrinsics; on x86_64 `target_clones`
  gives one copy per instruction set in one .so.
- Bit-identical scores: a row is processed 16 elements at a time into 16 lane
  accumulators (lane j holds elements j, j+16, ...) and the lanes are added in
  one fixed order; `-ffp-contract=off` forbids fused multiply-add. So a score
  is the same on every CPU, with any number of threads, alone or in a batch.
  `tests/engine/test_kernels.cpp` pins a fingerprint of the result bits; it
  is the same with clang on macOS arm64 and g++ 11.5 on Linux aarch64.
- The 16 lanes are two groups of 8, each made of native vectors: two 16-byte
  vectors on aarch64, one 32-byte vector on x86_64. A single 64-byte vector
  type is simpler to write, but g++ keeps such a variable in memory on aarch64
  (the hot loop stored and reloaded it through the stack): changing the type
  made a single query 3.2 times faster with the same result bits. Until
  milestone M4 x86_64 also used four 16-byte vectors; GCC never widens vector
  code, so its avx2 and avx512f copies computed 4 floats per instruction (seen
  in the disassembly on the 4-node cluster). The width changes no result: the
  pinned fingerprint is the same.
- Two g++ 8.5 traps found on the way (both invisible with clang and g++ 11):
  a `memcpy` into a 32-byte vector goes through the stack (the loads now use a
  vector type with 4-byte alignment: one unaligned load), and a helper that
  the compiler does not inline into a `target_clones` function is compiled
  once for the default target (g++ 8.5 stopped inlining the 4-query kernel
  and batches ran on SSE at half the speed; every helper is now
  `always_inline`). Check both with `objdump -d` of the .so: the avx2 and
  avx512f copies must call nothing and use `ymm` registers.
- Flat search (`src/engine/flat.h`): candidates are ordered by (key, id), so
  ties go to the smaller id and the result does not depend on the order of
  the work. With few queries the rows are split over the threads and the
  per-thread top-k lists are merged; with many queries (or a large k) the
  queries are split. A tile of about 256 KB of rows stays in the L2 cache
  while every group of 4 queries is scored against it, so each row is read
  from memory once per tile, not once per query. A row whose key cannot enter
  the full top-k list is dropped before its id is read (milestone M6: 4 to 5%
  more queries per second in batches on the 4-node cluster's x86 nodes, 3%
  less time for one query on 10 threads; SIFT1M, the same results).
- Journal overlay: vsearch keeps the latest journal row per id (a delete wins
  a version tie), masks those ids in the snapshot with a bitset, and searches
  the live journal vectors with the same kernels beside the snapshot.
- Threads are started per call and joined before the function returns; they
  never call the SDK. Only the calling thread checks `isCanceled()` between
  work units. Below about 2 million multiply-adds the search runs on the
  calling thread only (thread start costs more than it saves).

## HNSW

`src/engine/hnsw.h`, after Malkov and Yashunin (TPAMI 2018); the mechanics
follow hnswlib, the code is our own.

- Built in place: the snapshot builder sorts the rows, lays out the graph
  section (its size follows from the ids alone, because the level of a node
  is a hash of its id) and the graph is written straight into the final
  buffer. There is no second copy of vectors or links. The graph refers to
  positions; the search reads the rows of the vectors section, so flat
  search, graph search and the journal overlay share one copy of the vectors.
- Insert (hnswlib's addPoint): greedy descent on the upper levels, a beam
  search with ef_construction on each level of the new node, the neighbour
  heuristic (Algorithm 4: a candidate is kept only when it is nearer to the
  new node than to every neighbour kept before it; pruned candidates are not
  kept, as in hnswlib), back links, and the heuristic again when a list is
  full. The new node gets m links on every level; lists hold m (upper levels)
  or 2 x m (level 0).
- Parallel build: a thread takes the next 64 positions at a time. Link lists
  are protected by 65536 striped mutexes (never two held at once); the global
  mutex is held only by an insert whose node becomes the new entry point.
- Two phases per insert (milestone M6): first the searches of every level of
  the new node, top down, keeping the selected neighbours of each level; then
  the links, bottom up. hnswlib links each level right after its search, so
  another thread can reach the new node on an upper level while its lower
  levels are still empty, start its own lower-level search there and end up
  linked only to the new node and other such late nodes (a first version of
  ours even lost those links: 0.5% of the nodes of a 4-thread build had no
  incoming link). With two phases nothing links to a node while it searches,
  so it can never find itself, and when a level is linked the levels below
  are complete. Recall and build time on SIFT1M are the same as before
  (recall@10 at ef 100 0.9829 against 0.9827 to 0.9830, 33.9 s on the VM).
  Defences on top: a search never keeps the node being inserted, a back link
  is not added twice, and the full check of a graph (vload, the base of an
  incremental build) refuses a self link and a position listed twice (+35 ms
  per million vectors). `test_hnsw` and `test_delta` build 200,000 vectors in
  clusters of equal and near-equal ones on 8 threads, full and incremental
  with tombstones, and check every list.
- Reachability repair: the heuristic can prune the last link to a node; such
  a node can never be found (hnswlib has the same effect, mostly with small m
  or many equal vectors). After the inserts, one thread walks level 0 from
  the entry point and links every node it misses from its nearest reachable
  node (a search for it), preferring a list with room. With it, a search with
  ef = count finds exactly what the flat search finds; `test_hnsw` checks
  that for every metric, for clustered data and for 1500 equal vectors.
- Search: greedy descent from the entry point, then a beam search on level 0
  with ef = max(ef_search, k). The neighbours of a node are scored in one
  kernel call (`keys_gather`) that prefetches the next rows while it scores
  one; the visited marks are prefetched too. Visited marks are 16-bit stamps
  per position (a new search takes the next stamp; the array is cleared only
  when the stamp wraps), kept by the process between calls, 2 bytes per
  vector per search thread. Masked and tombstoned positions (the journal
  overlay) are traversed but never returned; the journal's live vectors are
  searched exactly and merged.
- Determinism: candidates are ordered by (key, position), a total order, so a
  search on a given snapshot gives the same result on every thread count and
  CPU. The graph of a parallel build depends on the interleaving of the
  threads; with one build thread it depends on the data only.
- Precision levels are presets of ef_search: fast max(2 x k, 32), balanced
  100, best 400; exact (or `exact=true`) is the flat search.
- Graph memory: (2m + 1) x 4 bytes per vector on level 0, 5 bytes for the
  level and the upper index, and on average 1/(m - 1) upper blocks of
  (m + 1) x 4 bytes: 141 bytes per vector with m = 16 (135 MB for SIFT1M).

Engine against hnswlib, SIFT1M (1M x 128, l2), m 16, ef_construction 200,
10,000 queries, k 10, same VM, same threads (`make bench DATA_DIR=...
HNSWLIB_DIR=...`; hnswlib v0.10.0-rc.2 built with its own flags `-Ofast
-march=native`, vvector with `-O3 -ffp-contract=off`):

| | vvector | hnswlib |
|---|---:|---:|
| build, 1 thread | 197.5 s | 229.0 s |
| build, 8 threads | 34.0 s | 39.5 s |

| ef_search | recall@10 vvector | recall@10 hnswlib | q/s 1 thread vvector | q/s 1 thread hnswlib | q/s 8 threads vvector | q/s 8 threads hnswlib |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.7084 | 0.7079 | 42,752 | 38,830 | 284,203 | 224,250 |
| 16 | 0.8001 | 0.8016 | 30,902 | 28,162 | 209,882 | 172,439 |
| 32 | 0.9031 | 0.9034 | 20,444 | 18,106 | 127,755 | 106,988 |
| 64 | 0.9632 | 0.9634 | 11,867 | 10,396 | 73,476 | 62,513 |
| 100 | 0.9828 | 0.9826 | 8,042 | 7,198 | 50,294 | 43,092 |
| 200 | 0.9956 | 0.9956 | 4,473 | 3,967 | 27,169 | 23,927 |
| 400 | 0.9987 | 0.9987 | 2,364 | 2,179 | 14,767 | 13,049 |

Equal recall, 8 to 27% more queries per second, 14% faster build. One search
call (the unit vsearch makes) takes 0.13 ms at ef 100. hnswlib has no NEON
code: on aarch64 its distances come from the compiler's vectorisation of a
plain loop; on x86_64 it uses AVX intrinsics, so the comparison is repeated
on the x86 cluster when it has the memory for SIFT1M.

## int8 quantisation (milestone M4)

`quantization='sq8'` adds one byte per element to the snapshot (docs/format.md, "sq8 section"):
a code from 0 to 255 in one range for the whole index, `offset + scale x code`. The float vectors
stay: they are needed for rescoring and for `precision='exact'`.

- Training: the 0.001 and 0.999 quantiles of about 100,000 elements of evenly spaced rows (all
  dimensions pooled, as Qdrant does). One global range is enough for data whose dimensions have
  similar ranges (SIFT, normalised embeddings); per-dimension ranges would cost a float pair per
  dimension and a multiply per element in every distance, and are not planned.
- Codes are unsigned bytes; padding codes are 0 (not the code of 0.0), so the padding adds nothing.
  Every row stores the sum of its codes.
- Distances from codes are integer sums: sum (a - b)^2 for l2, sum abs(a - b) for l1, sum a x b
  for dot and cosine, in uint32 (255^2 x 32768 fits). They are exact, so every CPU, thread count
  and vector width gives the same candidates; the integer loops are plain C++ and the compiler
  vectorises them as it likes. The key adds the terms of the range in double, in one fixed order,
  so it is the approximate score itself (for dot and cosine: -(scale^2 x sum a b + scale x offset
  x (sum a + sum b) + dims x offset^2)).
- A search (src/engine/search.cpp): the queries are coded with the same range; the graph walk or
  the flat scan runs on the codes and keeps k x oversampling candidates (at least k); with
  `rescore` the exact float key of every candidate is computed from its row and the k best are
  returned, so the scores are the float scores; without, the k best candidates are returned with
  their approximate scores. The radius acts on the returned scores. The journal's rows are always
  searched exactly and merged afterwards. The graph is built on the float vectors: the codes change
  the search, not the graph.
- Speed details, each found by measuring on the x86 cluster (g++ 8.5): the candidates carry their
  positions (the flat scan runs on a block without ids, the graph walk reports positions), so
  rescoring reads the row directly instead of searching the id among a million (about 20 cache
  misses each), and it runs in parallel over the queries; before, a batch of 1000 queries was
  slower with sq8 than without. The flat scan scores 4 queries per pass over a tile of codes, as
  the float scan does. The integer loops use 16-bit factors, which GCC turns into pmaddwd (one
  multiply-add of 16-bit pairs); g++ 8.5 takes that form for a dot product only when a factor is
  signed, so the row code is shifted by 128 and the sum corrected exactly with the query's code
  sum. With 32-bit factors it used 32-bit multiplies.
- Presets of `precision`: fast = codes only (no rescoring), balanced = 2 x k rescored (4 x k from
  512 dimensions on, see "768 and 1536 dimensions"), best = 4 x k; each can be overridden by
  `rescore` and `oversampling` (per call or per session: a value set there replaces the preset at
  every dimension).
- Incremental refresh keeps the base's range and codes the appended rows with it; a full build
  trains anew. A change of the option is a full build (the options are part of active_options).
- `memory_mode compact`: a query mapping asks the kernel to read ahead only what follows the float
  rows (ids, codes, graph: the float rows are the first section) and marks the float rows
  MADV_RANDOM; rescoring reads a few rows per query. The mode reaches the nodes in the OPTIONS
  file and applies to the next mapping of a snapshot.
- Tests: tests/engine/test_sq8.cpp proves that rescoring every candidate gives the float result bit
  for bit (four metrics, flat and HNSW, journal rows, masks, radius, incremental snapshots), bounds
  the error of the approximate scores, checks thread independence and the SIFT1M recall loss;
  tests/sql/test_sq8.sh does the same through SQL.

SIFT1M recall@10 through SQL (1000 queries, the VM): fast 0.8839, balanced 0.9797, best 0.9989;
the float index: 0.8926, 0.9804, 0.9987. Rescoring 2 x k candidates loses 0.0007 of recall.

## Filtered search and range search (milestone M5)

**Filtered search.** Allow-list rows (id set, vec and del NULL) reach vsearch with the queries. It
sorts their ids, drops repeats, finds their positions in the snapshot (ids that are not there are
ignored) and removes the positions the journal or the tombstones mask. Journal rows count only
when their id is allowed. Then one of two paths (src/engine/search.cpp):

- *Exact*: the allowed rows are copied into one block (in parallel, into memory that is not
  cleared first) and searched with the tiled flat kernels, the journal beside them. Cost: the
  allowed rows.
- *Masked*: every position outside the list joins the skip mask, and the search runs as without a
  filter (graph or flat scan, float rows or sq8 codes). The graph walk passes through masked
  nodes but keeps only allowed ones in its result list, so it walks until it has ef of them:
  about ef x count / allowed nodes. ef is not raised further.

On an index without a graph (milestone M6) the masked path scores every row for every query (with
sq8 codes a quarter of the bytes), the exact path copies the allowed rows once (about the cost of
scoring them for 8 queries) and scores only those: the exact path is taken when allowed x (queries +
8) < rows x queries (with codes: 4 x allowed x (queries + 8)). On flat SIFT1M (VM, 8 threads):

| allowed | queries | masked ms | exact ms | chosen ms |
|---|---|---|---|---|
| 1% | 1 | 2.38 | 0.39 | 0.32 |
| 1% | 1000 | 1,070 | 15.0 | 14.7 |
| 10% | 1 | 2.78 | 2.74 | 2.85 |
| 10% | 1000 | 1,117 | 121 | 121 |
| 50% | 1 | 3.69 | 7.08 | 3.62 |
| 50% | 1000 | 1,138 | 587 | 592 |

Before, a flat index used the HNSW cut-over below and the masked path above it (80,000 rows here).
On an HNSW index the exact path is taken below max(10000, sqrt(64 x ef x count)) allowed rows: the
two costs grow
as allowed and as 1 / allowed, so they meet near sqrt(c x ef x count). The plan's first rule,
max(10 x ef, 10000), was far too low: measured on SIFT1M (VM, 8 threads, ef 100, float; recall of
the masked graph against the exact path; tests/engine/bench_filter.cpp):

| filter | allowed | exact path, batch q/s | masked graph, batch q/s | recall | exact, 1 query ms | graph, 1 query ms |
|---|---|---|---|---|---|---|
| 0.01% | 89 | 831,049 | 17.1 | 1.0000 | 0.002 | 320.8 |
| 0.1% | 998 | 205,978 | 130.7 | 1.0000 | 0.026 | 41.8 |
| 1% | 9,919 | 75,687 | 911.6 | 1.0000 | 0.246 | 6.10 |
| 2% | 19,782 | 40,641 | 1,610.7 | 1.0000 | 0.395 | 3.51 |
| 5% | 50,167 | 16,498 | 3,486.7 | 0.9999 | 1.104 | 1.77 |
| 10% | 100,612 | 8,303 | 6,377.8 | 0.9994 | 2.489 | 1.18 |
| 20% | 200,066 | 4,564 | 11,595.8 | 0.9981 | 4.276 | 0.99 |
| 50% | 500,657 | 1,770 | 24,430.0 | 0.9925 | 7.009 | 1.44 |

A batch is 1000 queries in one call; a single query is one call. The paths meet at about 68,000
allowed rows for single queries and 120,000 for batches; the rule gives 80,000 for this index
(c = 64). With sq8 codes the graph is faster (1%: 1,288 q/s; 50%: 35,832 q/s), which moves the
meeting point down by about a quarter; one rule serves both. Before the parallel copy the exact
path of a single query took 4.8 ms at 5% (a serial copy into zero-filled memory). On the 4-node
cluster's x86 nodes (10 threads) both paths are slower, but they meet at the same place: about
67,000 allowed rows for one query and 95,000 for a batch.

Through SQL (VM, index sift_hnsw, one query, median at the client over 100 runs; the allow-list
rows come from a table; scripts/latency.sh shapes filter100, filter10k, filter100k):

| statement | fenced | mixed |
|---|---|---|
| no filter (`_snap`) | 7.4 ms | 1.9 ms |
| 100 allowed ids | 8.6 ms | 3.0 ms |
| 10,000 allowed ids | 12.5 ms | 5.3 ms |
| 100,000 allowed ids | 29.8 ms | 13.9 ms |

The engine's part is under 1.5 ms in every row; the rest is Vertica reading the allow-list rows and
passing them to vsearch (about 0.12 microseconds per row mixed, twice that fenced). Finding the
allowed ids in the snapshot with one galloping pass over the sorted ids (`VectorSet::find_sorted`)
instead of one binary search per id saved 3 ms at 100,000 ids.

On the 4-node cluster (Enterprise mode, SIFT1M index sift_hnsw loaded on every node, one query,
client median over 200 runs; scripts/latency.sh shapes filter* and filter*seg) the source of the
allow-list rows decides the cost. vsearch runs on the initiator (`OVER()`); rows of a segmented
table are scanned on every node and sent there, which costs about 24 ms whatever their number:

| statement | fenced, unsegmented | fenced, segmented | mixed, unsegmented | mixed, segmented |
|---|---:|---:|---:|---:|
| no filter (`_snap`) | 15.0 ms | 15.0 ms | 6.8 ms | 6.8 ms |
| 100 allowed ids | 17.4 ms | 40.5 ms | 8.7 ms | 32.9 ms |
| 10,000 allowed ids | 31.3 ms | 55.5 ms | 14.2 ms | 43.3 ms |
| 100,000 allowed ids | 76.7 ms | 95.3 ms | 41.2 ms | 68.9 ms |

On the VM (one node) both tables cost the same (mixed 2.7 / 4.4 / 15.1 ms). This is the delta
view's gather again (lever 3); the remedy is the same, a table or projection of the filter columns
that is `UNSEGMENTED ALL NODES` (README "Filtered search"). Per allowed id the x86 nodes need
about 0.35 microseconds mixed, three times the VM.

**Range search on HNSW.** Before M5 the graph search kept ef = max(ef_search, k) candidates and
cut them by the radius: to get "everything within r" one set k large and paid a walk with a
candidate list of k. Now, with a radius, the walk starts with the preset ef (fast: 32) and makes
the list four times longer, up to k, while more than a quarter of the list is within the radius
(hnsw.cpp `walk`). A walk cannot be resumed with a longer list (it has dropped what did not fit),
so each step walks again from the start; with a factor of four the earlier steps cost at most a
third of the last one. The quarter keeps the list at least four times the answer, where recall is
high. SIFT1M, k 16384, 100 queries, radius = the median over the queries of their r-th neighbour
distance (so the mean number within the radius is larger than r):

| r | mean within the radius | growing walk q/s | list of k q/s | recall | flat q/s |
|---|---|---|---|---|---|
| 10 | 407 | 2,674 | 485 | 0.9999 | 639 |
| 100 | 1,163 | 1,268 | 489 | 0.9999 | 618 |
| 1000 | 3,846 | 687 | 494 | 0.9999 | 558 |

On the x86 cluster nodes: 1,752 / 964 / 552 queries per second against 426 / 425 / 423. Through SQL
the range statement (k 16384, a radius holding the query's 10 nearest) costs what a plain search
costs: 7.3 ms fenced, 1.9 ms mixed on the VM; 15.6 and 7.1 ms on the 4-node cluster (plain search
12.8 and 5.9 ms in the same run).

A first version doubled the list while the whole list was within the radius: recall on random l1
data fell to 0.93 (the answers sat at the end of the list, where the walk is least complete), and
restarting at every doubling cost up to twice the last walk.

**Vector functions.** src/engine/vecmath.cpp computes in double, element by element in index
order. A C++ aggregate cannot read an ARRAY argument in Vertica 26.2 (the server crashed in
`BlockReader::getArrayRef` inside `aggregate()`; VERTICA_NOTES), so vector_sum and vector_avg are
transform functions, which can also run fenced.

## vload on every node

`vload(...) OVER(PARTITION NODES)` runs one function instance on every node
that receives input rows. Measured on Vertica 26.2 (in the earlier graph
project this code comes from):

- With `vvector.snapshot` as the only input, Vertica does not run the load on
  every node (an unsegmented table was read on one node only: on a 3-node Eon
  cluster only one node ran the load). On a single node this cannot be seen.
- So the input is driven by a segmented table. `vvector.probe(k)` holds 8192
  rows, segmented by hash, so every node has some. `vvector_admin.vnode(k)
  OVER(PARTITION NODES)` returns the smallest k stored on each node. The chunks
  are cross joined to those probe rows: one probe row per node, so every node
  gets every chunk exactly once. The snapshot table is segmented, so the join
  broadcasts the chunks to the probe rows: `SELECT /*+SYNTACTIC_JOIN*/ ... FROM
  vvector.probe p JOIN /*+DISTRIB(L,B)*/ vvector.snapshot s ON TRUE` (join
  hints are ignored without SYNTACTIC_JOIN). load_on_nodes leaves the hints
  out on one node, where Vertica only warns that they are not feasible.
- The mapping is computed inside the same statement, so it follows node and
  shard changes. vload returns one row per node that loaded.
- The join holds its inner, the broadcast chunks, in memory. At 100M x 128 (a 49.6 GB flat snapshot,
  4 nodes of 78 GB) the one statement failed: "Join inner did not fit in memory", the plan asked for
  32 GB more with 28 GB of the general pool free (milestone M6). load_on_nodes(index, snapshot,
  pass_mb) therefore loads in passes of pass_mb (2048 by default) of byte offsets, one statement each;
  vload (parameters part, pass, passes) writes every pass into `<id>.vv.part.<part>` at the chunks'
  offsets (O_NOFOLLOW, a regular file of the process user), and the last pass checks the whole file
  (size, structure, checksum) and makes it active; a failed pass removes the partial file, so the
  next pass fails too and the procedure names the pass. A snapshot up to one pass loads as before.
- vinfo and vconfig read `vvector.probe` for the same reason and answer once
  per node.

## Cache rules

- Layout: `<cache_dir>/<index>/<snapshot_id>.vv`, `<cache_dir>/<index>/ACTIVE`
  and `<cache_dir>/<index>/OPTIONS` (the index defaults of set_index_options).
- cache_dir: function parameter, then session parameter
  (`ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'`), then the
  index option `cache_dir` (milestone M6), then `/tmp/vvector`. A query cannot
  read the manifest, so the option reaches it as a file: vconfig writes the
  index defaults into the index's directory, and into the default directory
  (and the calling session's) an OPTIONS file that also holds
  `cache_dir=<dir>`. `open_active` reads OPTIONS in the directory it resolved;
  when that names another directory it takes ACTIVE, OPTIONS and the snapshot
  from there: one hop, never a chain, both directories owned by the database
  user. The redirect is read with ACTIVE, inside the 200 ms trust window, so a
  warm query still does no file system work. The procedures pass the option to
  vbuild, vload, vinfo and vconfig as the function parameter (vload must write
  there whatever a session says). A new value loads the active snapshot into
  the new directory in the same transaction and is committed only after every
  node loaded it; a failure rolls it back (no exception handler: around a
  failing multi-node UDx query PL/vSQL gets "Operation canceled").
- vload writes to a temporary file, verifies size, structure, checksum and id
  order, renames it into place, replaces ACTIVE by rename and syncs the
  directory. A failed load leaves the cache as it was.
- vload keeps the new and the previously active snapshot file. It removes
  other files in the index's directory only if they start with the vvector
  magic. It never touches anything else. Index names are limited to letters,
  digits and underscore, so a name cannot point outside cache_dir.
- Index defaults: a query cannot read the manifest table, so
  `set_index_options`, `refresh_index` and `load_all` call `vconfig` on every
  node, which writes the OPTIONS file (one file, fixed name, admin only).
- Two vload runs for the same index at the same time are not supported.
- The mapping of the active file is kept by the process and shared by later
  calls: a new mapping pays a page fault for every page a search touches. What
  ACTIVE and OPTIONS say is trusted for 200 ms (`ACTIVE_CHECK_MS`): a warm query
  does no file system work at all. When a query brings a newer snapshot id
  (the view of a refresh that just finished), ACTIVE is read again at once, so
  the stale check never fires by mistake. A changed index default is seen at
  the latest 200 ms later. Unfenced, the mapping lives in the Vertica process
  (address space only: the pages are file cache of the operating system).
  Fenced, it lives in the session's fenced process.
- Directories are created with mode 0700, files with 0600.
- A query maps a cache file after checking its header only (a full check of a
  large file per new mapping is impossible), so its safety rests on "vload
  wrote this file". The search functions are PUBLIC and cache_dir can be set
  by any user, so the mapping refuses a file that is not a regular file or
  not owned by the database's operating system user, and an index directory
  owned by someone else counts as no cache (vinfo does not list it). Opens are
  non-blocking (a FIFO in place of a file cannot hang a query). Temp files
  are named `<file>.tmp.<pid>.<n>`: two sessions of one unfenced process that
  write the same OPTIONS file at once no longer share a temp file.

## Memory

- vbuild: `4 x row_stride + 12` bytes per vector (the rows, the ids, the sort
  order), where row_stride is dims rounded up to a multiple of 16. The rows are
  written straight into the final buffer, which grows by remapping (Linux:
  `mremap`, no copy); pages that are never written cost nothing. Unordered
  input is sorted by moving the rows in place. Example: 1,000,000 vectors of
  128 dimensions: 500 MB; of 768 dimensions: 2.9 GB. Fenced, this memory
  belongs to the fenced process; its limit is `FencedUDxMemoryLimitMB`
  (-1 = no limit). `vvector.sizing()` and `vvector.status()` compute it.
- HNSW build: in addition, the graph section (below) and, per build thread,
  2 bytes per vector of visited marks; 5 bytes per vector for a moment while
  the levels are laid out.
- Incremental build: the new snapshot (base plus appended rows, graph, id_index
  when needed), `4 x row_stride` bytes per changed row while they are collected,
  one bit per position for the tombstones, and up to 2 bytes per position per
  insert thread for the visited marks. The base is the mapped cache file (page
  cache, shared).
- The snapshot and every cache file: 256 bytes plus `4 x row_stride + 8` bytes
  per vector, plus the graph: about 141 bytes per vector with m = 16 (codes
  from M4).
- A process keeps the mapping of every index it searched (`open_active`) and
  gives it back when the index had no query for 10 minutes (milestone M6), so
  an unregistered index or an old cache directory does not stay mapped for
  the life of an unfenced Vertica process. The visited arrays of graph
  searches are pooled up to 64 arrays and 512 MB; beyond that an array is
  freed when its search ends (a 100M index has 200 MB arrays).
- vsearch maps the snapshot and copies nothing from it. It holds the queries
  and the live journal vectors: `4 x row_stride` bytes each, and a bitset of
  one bit per snapshot vector when the journal has rows. A graph search keeps
  2 bytes per vector per search thread of visited marks in the process
  between calls (2 MB per thread for 1M vectors).

## Memory placement (milestone M6)

Measured on the VM (Vertica 26.2.0-1, 34 GB, SIFT1M HNSW index of 632 MB,
fenced, single searches from one session and batches of 1000 queries; local
results of session 12).

- The search side is lazy by construction: the cache file is mapped
  read-only and shared, pages come in when a search touches them and are
  file cache, which the kernel can reclaim. `vinfo` reports `resident_mb`
  (mincore over the file): how much of it is in memory now. vinfo maps the
  file without read-ahead advice, so asking does not load it.
- Vertica reads its own storage through the page cache: a scan of a 269 MB
  table (after the cache was emptied) grew the page cache by 277 MB. So large
  scans compete with the index pages under the kernel's LRU.
- Worst case, the whole page cache emptied (`drop_caches`): `resident_mb`
  4 MB; the first search 321 ms, the next four 15 ms on average, then normal
  again (median 7 ms); after the series the file was resident again (630
  MB): the read-ahead advice of the new mapping reloads the whole file at
  disk speed (here about 2 GB/s). At 100M vectors (66 GB for HNSW) the same
  reload takes the better part of a minute, and the searches during it are
  slow: that is where the next two points matter.
- A cache directory on tmpfs (`/dev/shm`, the index option `cache_dir`):
  the same warm speed (median 5 to 6 ms) and immune to eviction: after the
  page cache was emptied the index stayed resident and the first search took
  10 ms. The price: the memory is taken for good, and after a reboot the
  cache is empty until `load_all` (or the next refresh) fills it.
- Huge pages: a tmpfs mounted with `huge=always` (a DBA's mount; the test
  used 2 GB at /mnt/vvhuge, removed afterwards) held the file on 2 MB pages
  (ShmemHugePages 651 MB). Batches of 1000 queries took 28 to 29 ms instead
  of 35 to 37 ms (22% less), single searches a median of 4 instead of 5 ms:
  the graph walk's random reads miss the TLB less. A file on xfs cannot get
  huge pages for a read-only mapping on these kernels, and copying the
  snapshot into anonymous memory would break the "mmap, no copies" rule, so
  this is a deployment recommendation (README), not code.
- `FencedUDxMemoryLimitMB` is enforced as the address-space limit (RLIMIT_AS)
  of each fenced process (VERTICA_NOTES): with 400 MB, a SIFT1M build fails at
  its first 257 MB mapping, in memory or in a file alike.
- The build buffer can live in a file (vbuild `build_in='file'`: an unlinked
  file in the index's cache directory, mapped shared, grown with ftruncate
  and mremap): its pages are page cache that the kernel writes out and
  reclaims, so a build larger than the free memory finishes, slower, instead
  of running the node out of memory. Cost where memory is plentiful: flat
  SIFT1M 3.64 to 3.88 s (+7%), HNSW 36.7 to 41.6 s (+13%); the bytes are
  identical. `refresh_index` chooses it by itself when the build estimate
  (the one `status` prints) exceeds half of the smallest node's free memory
  plus page cache, and says so in the refresh note. It does not get around
  `FencedUDxMemoryLimitMB` (address space).
  The 100M proof (milestone M6) showed that a file must never be written in
  random order: the rows arrive in random id order, and the in-place sort
  (one permutation cycle at a time) dirtied one 4 KB page per row. The node
  reached its dirty-page limit (16 GB) and the kernel wrote the pages back
  one by one: 8.5 MB/s on the cluster's virtual disk, 3.5 CPU minutes in 46
  minutes of build, an estimated 10 hours for 100M x 128. A build in a file
  therefore sorts by copying the rows in position order into a second
  unlinked file (written front to back, the first file only read, then
  dropped: the disk holds the vectors twice for a moment), and builds the
  HNSW graph in anonymous memory and copies it into the file in one pass.
  The in-memory build still sorts in place (no second copy in memory).
- A separate `hot_dir` for the ids, codes and graph (PLAN 17.1 d) is not
  built (decided with the 100M proof, "100 million vectors" below): at 1M
  and 10M one slow search per eviction is the whole cost, `cache_dir` on a
  tmpfs pins a whole index that fits, and at 100M the float rows (half of
  every search's reads) would still come from disk. The 100M numbers showed
  instead that the whole-file read-ahead of a new mapping stalls the first
  search for as long as the disk needs for the file (45 s for 63 GB); the
  advice is bounded since milestone M7 (below).

## Freshness: exact results between refreshes

The vector table is a journal: rows are only inserted, a delete is a row with
the delete flag, and for every id the row with the latest version wins. The
version is a column of the customer's table (insertion time or an increasing
INT). Vertica's `epoch` pseudo-column is deliberately not used: it is not
unique, it can change, and it cannot be part of a projection.

- The rows a query must apply come from the view `<schema>.<index>_delta`:
  `ver_col > boundary` plus one sentinel row, because Vertica does not call a
  transform function on empty input. Every row carries the id of the snapshot
  the view belongs to. `<schema>.<index>_snap` is the sentinel alone: the input
  of a snapshot-only search that keeps the stale check.
- The boundary is a literal, written into the view by refresh_index, so
  Vertica can prune partitions and storage containers.
- Boundary for a timestamp version: LEAST(clock at refresh start, earliest
  lock request of an open writer on the table, from `v_monitor.locks`) minus
  the margin (default 60 s). A row is missing from the snapshot only if it was
  committed after the build started reading, so it was written either after
  the refresh started or by a transaction that was open at that moment, and an
  open writer holds an insert lock. With `CLOCK_TIMESTAMP()` versions the
  argument is exact; the margin covers clock skew and `SYSDATE()` versions.
  For an INT version: highest version minus the margin, and the margin must
  cover open writers. `tests/sql/test_freshness.sh` has the two-session case
  with margin 0.
- Consolidation for the build: the latest row per id by version (a delete
  wins a tie), kept if it is not a delete. Without a version column every id
  must appear once.
- vsearch with `freshness='exact'`: ids that appear in the delta are masked
  in the snapshot; their latest delta row, if it is not a delete, is searched
  exactly next to the snapshot; the two are merged into one top-k. Applying
  is idempotent. `freshness='snapshot'` (the default) ignores journal rows.
- The build reads the journal rows up to the boundary only (milestone M6), so
  the snapshot and the delta never overlap and the snapshot holds exactly the
  rows the journal digest covers. Before, rows inside the margin went into
  the snapshot too; a physical DELETE or UPDATE of such a row before the next
  refresh was then seen by no verification (the digest covers the rows up to
  the boundary) and stayed in the index until a full build. The price: a
  snapshot-only query does not see the rows of the last margin seconds before
  the refresh (60 s by default), and a refresh that finds no live vector up
  to the boundary (the first refresh right after a load) stops with a message
  that says so. An open writer holds the boundary back; when it is older than
  ten times the margin (at least 60 s), the refresh note says so.
  `tests/sql/test_search.sh` checks the result against the full scan of the
  live rows after 1000 adds, 1000 deletes and 500 replacements.
- Stale cache: if the node's active snapshot is older than the snapshot id on
  the view's rows, vsearch fails with "snapshot cache stale on <node>: run
  vload". Newer is fine (a refresh is in flight).
- Refresh order, never changed: boundary, build, insert chunks, vload on all
  nodes, index defaults on all nodes, update manifest and views, delete older
  snapshots. An incremental refresh (next section) keeps the order; only the
  build reads less.
- One refresh per index at a time: a refresh marks the manifest row with one
  conditional UPDATE (`refresh_started_at`, `refresh_started_by` = user and
  session) and removes the mark at the end, also on error (as far as PL/vSQL
  can catch it: a UDx query that fails on several nodes ends the CALL without
  the handler, VERTICA_NOTES; a mark of the caller's own session is therefore
  also ignored). A mark whose session is gone from `v_monitor.sessions` is
  ignored at once (milestone M6),
  so a killed refresh no longer blocks the index for hours. A superuser sees
  every session there, another user only its own, so the check covers what
  the caller can see. A refresh run by a schedule trigger runs in a session
  that `v_monitor.sessions` never shows (VERTICA_NOTES): its mark says
  "scheduled" and is judged by its age only. The age limit is 6 hours or 4
  times the index's last build time, whichever is longer, so a 100M build of
  3 hours keeps its mark for 12.

## Incremental refresh (milestone M3)

`refresh_index` builds a new snapshot from the active one and the journal
rows after the previous boundary, the same rows the delta view shows,
consolidated per id (latest row, a delete wins a tie; deletes are passed to
vbuild with `del = true`). vbuild gets `base_snapshot` and reads the base from
the cache of its node, verified like a vload (checksum, ids, graph links): a
damaged base would otherwise be copied into every later snapshot with a fresh
checksum. The price is a read of the whole base at every incremental refresh
(0.1 to 0.4 s at 1M vectors, seconds at 100M); kept on purpose. Since
milestone M6 the rows are the ones up to the new boundary only (see
Freshness).

The engine (`src/engine/delta.cpp`, `IncrementalBuilder`):

- Copies vectors, ids and tombstones of the base into a new buffer. A delete
  of a live id sets the tombstone bit of its position. An add of an id that is
  live with a different vector tombstones the old position; every new or
  changed vector is appended as a new position, in id order. An add whose
  vector is bit for bit the live one changes nothing: the delta margin brings
  back rows the base already has, and without this every refresh would
  tombstone and append them again. A delete of an id that is not live changes
  nothing. When nothing changes, vbuild returns no rows and the refresh keeps
  the snapshot and moves only the boundary.
- An id can then be at several positions. The id_index sorts by id and, for
  one id, by position from high to low; the live position is always the
  highest (every change appends), so `VectorSet::find` looks at the first
  entry only. Without appended ids below the largest base id (ids that only
  grow, the usual case) the ids stay ascending and no id_index is written.
- HNSW (`hnsw_extend`): the base graph is copied to the same positions; the
  base's upper blocks come first, so no block index changes; the new
  positions are inserted with the insert code of the full build, in parallel,
  starting from the base's entry point. As in hnswlib, tombstoned nodes are
  passed through during the insert but never chosen as neighbours; they stay
  in other nodes' lists as bridges until the next full build. The
  reachability repair pass of the full build runs again (it skips tombstoned
  nodes), because a link list that overflows is shrunk by the heuristic and
  can drop the last link to a node.
- Tested (`tests/engine/test_delta.cpp`): after every round of random changes
  the new snapshot passes the full verification, holds exactly the live
  vectors, and a flat search, and a graph search with ef = count, give bit
  for bit the result of a full build of the same vectors; every live node is
  reachable. SIFT1M (`make test DATA_DIR=...`, VM, 8 threads): a 900,000-vector
  base, then 100 rounds of 1000 adds and 500 deletes, 0.25 s per round (the
  slowest 0.41 s; no Vertica). Recall@10 of the result against the exact
  answer of the live set, next to a full build of the same 950,000 vectors
  (32.3 s):

  | ef_search | after 100 incremental rounds | full build |
  |---:|---:|---:|
  | 32 | 0.9072 | 0.9056 |
  | 64 | 0.9658 | 0.9650 |
  | 100 | 0.9845 | 0.9837 |
  | 200 | 0.9965 | 0.9962 |

  Inserting into a finished graph is what the HNSW build does all along, so
  the graph does not get worse; 50,000 tombstones (5%) did not lower recall
  either.

When the refresh builds in full instead (`refresh_index` in
`sql/procedures.sql`, the reason goes to `manifest.refresh_note`): mode
`full`; the first build; a static index; changed build options
(`manifest.active_options` records those of the active snapshot) or format;
tombstones above `tombstone_ratio` of the positions or `rebuild_every`
incremental refreshes reached (mode `auto` only); a node whose cache does not
hold the active snapshot (checked with `vinfo` before the build); and a
journal whose rows up to the previous boundary are not the rows the last
refresh saw. A row committed later always has a newer version (that is the
boundary rule), so those rows only change when someone changes them
physically: DELETE, UPDATE, dropped partitions, rows written with old
versions by hand. The manifest keeps their number and a digest,
`SUM(HASH(id, vec, op, ver)::NUMERIC(38,0))` (`manifest.boundary_rows`,
`manifest.boundary_digest`). Every refresh carries both forward from the rows
between the previous and the new boundary (all of them, before the
consolidation per id): a scan of the new partitions only. The verification
recomputes both over every row up to the previous boundary and compares;
`verify_every` (index option) says how often: 1 at every refresh (the
default), N every N refreshes, 0 never. A full build stores exact values.
Vertica's UPDATE keeps the row count (verified), so the count alone missed
it; the digest sees a changed vector, delete flag or version. HASH of a
FLOAT array depends on every element and on their order (VERTICA_NOTES);
two different row sets with the same count and the same sum of 63-bit
hashes are possible in theory, but not by accident in practice. Row epochs
were not used: what mergeout does to them is not a contract. Cost: below.

## Where a single query spends its time

Vertica 26.2.0-1, single node VM (aarch64, 8 cores, 34 GB), SIFT1M index
(1,000,000 x 128, flat, l2), k = 10, 200 runs per row after 5 warm-up runs,
`scripts/latency.sh`. Client = vsql `\timing` (includes the round trip);
server = `v_monitor.query_requests` (end minus start timestamp). Medians in
milliseconds; p99 is within 1 ms of the median except where noted.

| Statement shape | Fenced client | Fenced server | Unfenced client | Unfenced server | What it adds |
|---|---:|---:|---:|---:|---|
| `SELECT 1` | 0.84 | 0.69 | 0.81 | 0.67 | round trip, parse, trivial plan |
| `vvector.vversion() OVER()` | 6.99 | 4.13 | 1.11 | 0.86 | a transform function: about 6 ms fenced, 0.3 ms unfenced |
| vsearch, `query` parameter, 16-vector index | 6.78 | 3.80 | 1.78 | 1.34 | vsearch setup, 12 parameters, cache check: about 0.5 ms |
| vsearch, `query` parameter, `FROM sift_snap` | 12.28 | 7.92 | 5.02 | 4.31 | the search of 1M vectors, 8 threads: about 3.2 ms |
| same, `FROM dual` | 12.15 | 7.80 | 4.88 | 4.22 | the `_snap` view costs nothing measurable |
| same, query as an ARRAY literal row | 19.61 | 12.01 | 12.24 | 8.45 | Vertica parsing a 128-number literal: about 7 ms |
| `query` parameter, `FROM sift_delta`, `freshness='exact'`, empty delta | 13.47 | 8.80 | 6.18 | 5.17 | the journal view: about 1.1 ms |
| same with 1000 journal rows | 16.36 | 11.71 | 7.77 | 6.78 | 1000 rows of 128 numbers in and searched: about 1.7 ms |
| `FROM sift_snap`, `threads=1` | 20.07 | 15.62 | 12.71 | 12.01 | one thread instead of eight (p99 18.7 unfenced) |

The engine alone (`make bench`, no Vertica) needs 2.3 ms for this search with
8 threads and 8.6 ms with one; the statement adds about 2.7 ms unfenced and
about 10 ms fenced.

The levers of the plan, applied and measured:

| Lever | Result |
|---|---|
| 1. No journal scan for snapshot-only queries | `FROM <index>_snap` with the `query` parameter: the fastest shape; the view costs nothing against `FROM dual` and keeps the stale check |
| 2. Cheap delta scan | journal partitioned by version date: the empty delta costs 1.1 ms. Journal rows cost about 1.7 ms per 1000 |
| 3. No gather across nodes | single node: does not apply; on the Eon cluster see M2 below |
| 4. No file system work per call | ACTIVE and OPTIONS trusted for 200 ms, the mapping kept: a warm call makes no system call for the cache |
| 5. Function setup | 0.5 ms more than an empty transform function (unfenced); parameters are read once, the query parsed once |
| 6. Output | k rows written in one pass; no string work |
| 7. Lightest function shape | verified: with `isExploder` a transform function can be called without OVER() and beside other columns (VERTICA_NOTES); `vknn` measured at M2 below |
| 8. Prepared statements | not measured yet: the VM has the Vertica ODBC driver and JDBC jar but no driver manager and no Java; needs an install (asked) |
| 9. Resource pool | not measured: it needs a new pool (a database change); a statement this short shows no thread setup in the numbers above |
| 10. Fenced against unfenced | fenced adds about 6 ms per statement; see the recommendation below |
| Query as a parameter (decision 4) | saves about 7 ms per statement against an ARRAY literal: the literal is parsed by Vertica |

Recommendation from the numbers: fenced mode costs about 6 ms per statement,
more than the search itself. For a service that runs single searches, deploy
with `FENCED=mixed` (vbuild, vload and vconfig stay fenced; vsearch, vknn,
vinfo and vversion run in the Vertica process). For batch searches (1000 queries in
one statement) the difference is below 2% (957 against 944 queries per
second), and fenced is the safer choice.

### Repeated at milestone M2: HNSW, vknn, and the Eon cluster

VM as above (Vertica 26.2.0-1), `scripts/benchmark.sh` of 2026-09-23: the
same shapes on the HNSW index `sift_hnsw` (SIFT1M, m 16, ef_construction 200,
precision fast = ef_search 32), next to the flat index `sift`. Medians in ms,
client / server.

| Statement shape | Fenced | Unfenced | Mixed |
|---|---:|---:|---:|
| `SELECT 1` | 0.83 / 0.68 | 0.81 / 0.67 | 0.81 / 0.66 |
| flat, `FROM sift_snap` | 12.38 / 7.99 | 5.08 / 4.36 | 5.13 / 4.39 |
| flat, `FROM sift_delta`, empty delta | 13.55 / 8.87 | 6.26 / 5.24 | 6.24 / 5.21 |
| HNSW, `FROM sift_hnsw_snap` | 7.80 / 4.42 | 1.93 / 1.47 | 1.88 / 1.45 |
| HNSW, `FROM dual` | 7.71 / 4.39 | 1.66 / 1.29 | 1.74 / 1.35 |
| HNSW, `FROM sift_hnsw_delta`, empty delta | 8.86 / 5.24 | 2.69 / 2.08 | 2.68 / 2.06 |
| `vknn`, `query` parameter, `FROM dual` | 7.76 / 3.73 | 1.45 / 1.06 | 1.43 / 1.04 |
| `vknn` on the query row of a table | 7.94 / 3.91 | 1.61 / 1.16 | 1.58 / 1.13 |

(These rows were measured with the first default, precision fast. The default is now balanced,
ef_search 100: recall@10 0.98 instead of 0.89 on SIFT1M for 0.09 ms more engine time per search;
measured again at balanced, VM, 200 runs, client / server ms:

| Statement shape | Fenced | Mixed |
|---|---:|---:|
| HNSW, `FROM sift_hnsw_snap` | 7.05 / 4.01 | 1.85 / 1.42 |
| HNSW, `FROM dual` | 6.93 / 3.96 | 1.81 / 1.40 |
| HNSW, `FROM sift_hnsw_delta`, empty delta | 8.01 / 4.77 | 2.80 / 2.17 |
| `vknn`, `query` parameter, `FROM dual` | 7.36 / 3.65 | 1.56 / 1.14 |
| `vknn` on the query row of a table | 7.52 / 3.78 | 1.71 / 1.23 |

The 0.09 ms of engine time is inside the run-to-run noise of a statement.)

The HNSW search itself takes 0.05 ms (engine, one call at ef 32); the rest of
the 1.9 ms statement is Vertica: 0.8 ms for any statement, about 0.3 ms for a
transform function, about 0.5 ms for vsearch's setup, cache check and output,
and 0.1 to 0.3 ms for the `_snap` view against `FROM dual`. Fenced adds about
6 ms, as for the flat index.

Batches (one statement, 1000 queries of the SIFT1M query set, median of 3):

| Statement | Fenced | Unfenced | Mixed |
|---|---:|---:|---:|
| vsearch flat | 1076 ms | 1076 ms | 1098 ms |
| vsearch HNSW, precision fast | 22 ms | 13 ms | 12 ms |
| vsearch HNSW, precision balanced | 35 ms | 25 ms | 25 ms |
| vknn HNSW, precision fast (1000 rows) | 71 ms | 55 ms | 55 ms |

Recall@10 against the ground truth (1000 queries): flat 0.9994; HNSW fast
0.8926, balanced 0.9804, best 0.9987, exact 0.9994. `tests/sql/test_hnsw.sh
--sift=VVBENCH` asserts balanced >= 0.95 (it measured 0.9793 on its own build:
every parallel build gives a slightly different graph).

Eon cluster: 3 nodes, x86_64 with AVX-512, 2 cores and 15 GB per node,
Vertica 26.2.0-2; HNSW index of 100,000 random vectors of 128 dimensions
(l2), journal partitioned by version date and segmented by id. Medians in ms,
client / server, 200 runs.

| Statement shape | Fenced | Mixed | Fenced, replicated journal | Mixed, replicated journal |
|---|---:|---:|---:|---:|
| `SELECT 1` | 2.96 / 1.99 | 2.69 / 1.72 | 2.71 / 1.73 | 2.74 / 1.78 |
| `vversion() OVER()` | 10.02 / 6.42 | 3.63 / 2.35 | | |
| `FROM lat_snap` | 12.75 / 8.13 | 7.55 / 5.34 | 12.72 / 8.13 | 7.76 / 5.42 |
| `FROM dual` | 12.45 / 8.00 | 7.36 / 5.27 | | |
| `FROM lat_delta`, empty delta | 27.87 / 21.86 | 24.18 / 20.60 | 16.11 / 10.56 | 11.93 / 8.08 |
| same with 1000 journal rows | 38.84 / 32.62 | 33.05 / 29.43 | 22.04 / 16.12 | 15.01 / 11.63 |
| `vknn`, `FROM dual` | 12.05 / 6.81 | 5.87 / 3.74 | | |
| `vknn` on a query row | 14.41 / 9.05 | 8.66 / 6.24 | | |

The p99 on the cluster is noisy (up to 5 to 10 times the median on some
shapes): the nodes are small and shared.

Levers repeated at M2:

| Lever | Result |
|---|---|
| 3. No gather across nodes | On the cluster the empty delta cost 15 to 17 ms more than `_snap`: EXPLAIN shows the journal scanned on every node and sent to the initiator (Send/Recv) even when no row qualifies. A replicated (UNSEGMENTED ALL NODES) projection of the journal, ordered by the version column, filled with `REFRESH('table')`, with statistics on the version column: the planner then reads it on the initiator only ("Execute on: Query Initiator"). Without statistics the planner kept the segmented projection; with statistics that were stale (the table doubled after them) it still chose the replica. Since the M2 follow-up this is the default (`journal_replica = auto`, see "Journal replica" below) |
| 7. Lightest function shape | `vknn` (exploder, no OVER(), no view) is the fastest single search: 1.43 ms against 1.88 ms for vsearch `_snap` on the VM (mixed), 5.9 against 7.6 ms on the cluster. Fenced there is little gain (the fenced call dominates). It has no stale check and no journal, and in batches it is 3 to 4 times slower than vsearch (row by row against one batch on all cores). Kept as the lightest single-search shape; vsearch stays the exact and the batch path |

### Journal replica (lever 3 as the default)

`vvector.apply_replica` (sql/procedures.sql) runs in `register_index`, in every `refresh_index` and
in `set_journal_replica`. Mode `auto` keeps the projection `<schema>.<index>_journal_rep` (id, vector,
delete and version columns, `ORDER BY` version, `UNSEGMENTED ALL NODES`) while the database has
more than one node and one copy of the journal (the largest segmented projection of the table,
summed over the nodes, from `v_monitor.projection_storage`) is at most 2048 MB and at most 10% of the
smallest `disk_space_free_mb` of a DATA or DEPOT location. `on` keeps it always, `off` never. It makes
it with `CREATE PROJECTION` and `REFRESH(table)`, takes `ANALYZE_STATISTICS(table.version)` at every
check, drops it when the rule no longer holds, shares it between indexes on the same journal
columns, never touches a projection it did not make, and makes none in `auto` when the table has an
unsegmented projection already. A failure (rights, name taken) is caught and written to
`replica_note` with the statements for a DBA. `CREATE PROJECTION` and `REFRESH` wait for every open
transaction that writes to the table (measured: 12 s behind a writer that committed after 15 s);
`ANALYZE_STATISTICS`, `DROP PROJECTION` and `SELECT` do not. So a new replica is postponed while
`v_monitor.locks` shows another writer (the note says so; the next check tries again), and a
refresh never hangs behind a long load.

Cost and gain, measured on the 3-node Eon test cluster (x86_64, 2 cores, 15 GB per node, Vertica
26.2.0-2), journal of random vectors of 128 dimensions:

| Measurement | Without | With the replica |
|---|---:|---:|
| vsearch over `_delta`, empty delta, mixed (100k rows, through `register_index` with `auto`) | 23.4 ms | 10.7 ms |
| the same with 1000 journal rows | 31.3 ms | 14.6 ms |
| `REFRESH` of the new projection, 200k rows | | 7.5 s once |
| `ANALYZE_STATISTICS` of the version column, 240k rows | | 45 ms |
| bulk `INSERT ... SELECT` of 20,000 rows | 250 to 383 ms | 830 to 914 ms |
| 100 single-row INSERT + COMMIT | 3.2 to 5.1 s | 4.7 to 5.2 s |
| `refresh_index` (flat, 440k to 520k rows; the build reads the segmented projection on all nodes, EXPLAIN checked) | 31.2, 31.5 s | 28.4, 24.6 s |
| storage per node (240k rows) | 63 MB | + 191 MB (full copy) |

In Eon the replica is one set of containers in the `replica` shard (11 containers, each listed on
all 3 nodes in `v_monitor.storage_containers`): one more copy in communal storage, and each node's
depot caches all of it. The limit of 2048 MB keeps the depot cost and the one-time `REFRESH` (about
80 s at the limit, extrapolated) small; above it `status` says why there is none and how to force it.

## Refresh cost

### The fixed part on a cluster (milestone M6)

A refresh that changes nothing, SIFT1M HNSW on the 4-node cluster (Vertica 26.2.0-3): 2.46 s
at the client. A trace of its statements (`v_monitor.query_requests` of the session) showed 382
requests: PL/vSQL runs every assignment and every IF condition as a query of its own (2 to 7 ms
each), about 25 of them to read single columns of the manifest row; the verification of the
journal digest 228 ms, vinfo 78 ms, vnode 56 ms, three manifest UPDATEs about 60 ms each, six
catalog lookups of column types about 23 ms each, the journal replica check about 100 ms. Now the
manifest row and the column types are read with one `SELECT ... INTO` each, the `_snap` view is
not made again when the snapshot did not change, statistics on the version column are taken only
when journal rows arrived, and the replica note is written only when it changed: 2.08 to 2.11 s
(-15%). What is left is mostly the digest verification (`verify_every`) and the procedure's own
expression queries.

SIFT1M, fenced, `refresh_index`: 9.4 s in total. The vbuild statement takes
6.4 s, the vload statement 1.6 s (write 496 MB, verify the checksum, flip
ACTIVE). Inside the vbuild statement:

| Step | Fenced | Unfenced |
|---|---:|---:|
| scan of the journal (1M rows) | 0.14 s | 0.14 s |
| consolidation (`ROW_NUMBER() OVER(PARTITION BY id ORDER BY ver DESC)`) | 2.25 s | 2.28 s |
| vbuild over the raw table (1 GB of FLOAT in, 496 MB out) | 2.04 s | 0.87 s |
| vbuild over the consolidated rows | 4.41 s | 2.51 s |
| the engine build itself (`make bench`) | 0.08 s | 0.08 s |

The rest of the statement stores the 62 chunks of 8 MB in `vvector.snapshot`.
Tried and not better: `LIMIT 1 OVER(PARTITION BY id ...)` (2.28 s), a journal
sorted by (id, ts) (2.26 s), a narrow window on (id, ts, del) with a semi-join
back (4.5 s). The consolidation of the whole journal on every refresh is what
the incremental refresh of milestone M3 removes: it reads only the changed rows.
The row transfer into the fenced process is 2.3 times slower than unfenced;
vbuild stays fenced in `FENCED=mixed` because a crash there would take the node.

At M2 (benchmark.sh, fenced): flat `refresh_index` 11.2 s (vbuild statement
8.1 s, vload 1.7 s; the M1 run measured 9.4 s: the VM varies by this much
between runs). HNSW `refresh_index` 45.6 s: the vbuild statement 41.9 s, of
which the graph build is about 34 s (8 threads, `make bench`), and vload 1.9 s
(write 631 MB, verify the checksum and every link of the graph).

### Incremental refresh at milestone M3

Vertica 26.2.0-1, the VM, fenced, `scripts/benchmark.sh --parts=incremental`: a flat
and an HNSW index (m 16, ef_construction 200) on a copy of the first 900,000 SIFT1M
vectors; after each change `refresh_index` (client time, and the two big statements):

| Change | flat total | vbuild | vload | hnsw total | vbuild | vload |
|---|---:|---:|---:|---:|---:|---:|
| first build (full) | 10.1 s | 7.7 s | 1.6 s | 40.7 s | 38.1 s | 1.8 s |
| none: snapshot kept | 0.53 s | 6 ms | - | 0.51 s | 5 ms | - |
| 100 adds, 50 deletes | 4.4 s | 2.2 s | 1.5 s | 6.5 s | 3.9 s | 1.9 s |
| 1,000 adds, 500 deletes | 4.4 s | 2.3 s | 1.5 s | 6.5 s | 3.9 s | 1.9 s |
| 10,000 adds, 5,000 deletes | 4.7 s | 2.3 s | 1.7 s | 7.2 s | 4.4 s | 2.1 s |
| 50,000 adds, 25,000 deletes | 5.1 s | 2.7 s | 1.7 s | 9.2 s | 6.4 s | 2.0 s |
| full build of the 930,550 vectors | 8.5 s | 6.3 s | 1.6 s | 42.4 s | 39.9 s | 1.8 s |

So an incremental refresh is a fixed cost plus about 0.04 ms (HNSW) or 0.01 ms (flat)
per changed row. The fixed cost is the size of the index, not of the change: the new
snapshot is written to `vvector.snapshot` in full (the vbuild statement: 2.2 s for
450 MB flat, 3.9 s for 570 MB HNSW, of which the engine needs 0.25 s and verifying the
base 0.1 to 0.4 s: 0.39 s for 630 MB when its pages are not mapped yet, 0.10 s after) and loaded on every node in full (vload, 1.5 to 2 s). Before the
three changes below, the 100-refresh test measured 7.3 to 7.6 s per HNSW refresh:

- `refresh_index` called `status` first; its NOTICEs never reached anyone (nested
  CALL), 0.5 s: removed.
- The snapshot size was `SUM(OCTET_LENGTH(chunk))`, which reads every chunk: 1.1 s for
  574 MB. Now the offset and length of the last chunk.
- The highest version was `MAX(ver)` over the whole journal (0.17 s at 1M rows, and it
  grows with the journal); now over the rows after the previous boundary only.

What is left and could be shaped later: moving only the changed parts of the
snapshot. With appended positions every section after the vectors moves, so a
chunk-level difference needs a layout with room to grow (a format version 3), and
the level-0 links of the graph change all over the file (every insert adds back
links). Not planned; noted here with the numbers it would be judged by.

The 100-refresh acceptance test (`tests/sql/test_incremental.sh --sift=VVBENCH`,
before the three changes): 900,000 base vectors, 100 refreshes of 1000 adds and 500
deletes, tombstone_ratio 0.03; one full build fired at round 59 (29,000 tombstones in
958,000 positions), 99 incremental; median 7.8 s, full 46.5 s; the index holds exactly
the 950,000 live vectors (manifest, vinfo and the journal agree) and recall@10 at the
default precision against the exact search of 1000 queries passed (>= 0.95). The
disk use of the database stayed within 7 GB of its start: the Tuple Mover purged the
deleted snapshot rows by itself.

### Journal digest (M3 follow-up)

Carrying the digest forward costs a scan of the rows between the two
boundaries: 1 to 2 ms for a quiet journal. The verification reads every
journal row up to the boundary, the vectors included: a cost that grows with
the journal, like writing the snapshot. The refresh labels the statements
`vvector_verify` and `vvector_digest` and prints the verify time in
`refresh_note`. VM, fenced, `scripts/benchmark.sh --parts=incremental` on the
900,000 x 128 copy of SIFT1M (client ms):

| Refresh | flat total | hnsw total | journal statement |
|---|---:|---:|---:|
| no change, `verify_every` 1 (verified) | 1037 | 1034 | verify 433 to 435 |
| no change, `verify_every` 0 (carried forward) | 599 | 601 | digest 1 to 2 |
| 1,000 adds, 500 deletes, verified | 4979 | 6861 | verify 422 to 425 |
| 50,000 adds, 25,000 deletes, verified | 5718 | 9490 | verify 605 to 793 |
| full build (exact values, up to the new boundary) | 8848 | 42090 | digest 563 to 746 |
| first build, journal read from disk the first time | 10199 | 40970 | digest 382 to 1774 |

The verify scan takes about 0.43 s when the journal is in memory and more
right after a large insert (the new rows are read the first time). At M3,
before the digest, a refresh that changed nothing took 0.53 s; with
`verify_every` 0 it takes 0.60 s: the other 70 ms are the two manifest
updates of the one-refresh-at-a-time guard. A count on the version column
alone took 23 to 29 ms at 1M rows: the vectors are what the verification
pays for. A shorter verification would need digests per partition,
re-read only for partitions whose storage containers changed since the last
refresh (`v_monitor.storage_containers`); not planned now. A journal
compaction (M7, optional) would have to write the new digest in the same
transaction.

### Segmented snapshot table (milestone M6)

Until M6 `vvector.snapshot` was `UNSEGMENTED ALL NODES`: every refresh wrote the whole snapshot
once per node, and on a 4-node cluster the part of a refresh that grows with the index was twice
that of one node. Now the table is segmented by hash of (snapshot_id, byte_offset): a refresh
writes one copy (two with K-safety 1), and vload broadcasts the chunks to one probe row per node
(`JOIN /*+DISTRIB(L,B)*/` with `/*+SYNTACTIC_JOIN*/`; section "vload on every node"). Each node
now receives the chunks over the network instead of reading its own copy; the measurements show
that this costs nothing extra.

The 4-node cluster (Enterprise mode, K-safety 1, 10 cores per node), first the chunks of one
snapshot copied into each kind of table and loaded from it (the refresh's two statements alone):

| snapshot | write, unsegmented | write, segmented | vload, unsegmented | vload, segmented | storage per node |
|---|---:|---:|---:|---:|---|
| SIFT1M HNSW, 630 MB (3 rounds) | 12.0 s | 6.3 s | 5.6 to 8.0 s | 5.6 to 6.4 s | 345 MB -> 170 MB |
| BIGANN 10M HNSW, 6.3 GB (2 rounds) | 112 s | 46 s | 54 s | 39 s | 3.5 GB -> 1.7 GB |

Then `refresh_index` end to end (scripts/benchmark.sh --parts=incremental, 900,000 x 128, fenced):

| change | HNSW before | HNSW after | flat before | flat after |
|---|---:|---:|---:|---:|
| 100 adds, 50 deletes | 17.3 s | 13.6 s | 12.8 s | 11.3 s |
| 1000 adds, 500 deletes | 16.9 s (vbuild 8.5, vload 5.4) | 12.7 s (vbuild 5.0, vload 3.9) | 12.8 s | 11.1 s |
| 10000 adds, 5000 deletes | 18.0 s | 12.9 s | 13.8 s | 10.3 s |
| 50000 adds, 25000 deletes | 21.6 s | 17.2 s | 14.7 s | 11.5 s |
| full build | 66.5 s | 63.4 s | 14.6 s | 12.3 s |

A refresh that changes nothing touches the table with an empty INSERT and one COUNT (about 70 ms
together; the COUNT costs 21 ms segmented against 3 ms unsegmented). Measured
end to end, such a refresh took 2.1 to 2.4 s before and 2.5 to 2.9 s in two runs after; a trace of
its statements (v_monitor.query_requests of the session) puts about 40 ms of that on the snapshot
table and shows the rest (manifest updates of 70 to 300 ms, catalog lookups, the journal
verification, the journal replica check) varying from run to run. A second run of the whole
benchmark after the change gave 12.2 to 13.4 s for HNSW and about 10 s for flat at every change
size up to 10,000 adds.

On one node nothing changes (a segmented table on one node is one copy either way). The upgrade
from an unsegmented table copies it once inside `make deploy`: 39.6 GB of snapshots in 319 s on the
4-node cluster, 2.3 GB in a 14 s deploy on the VM.

### Prewarming (milestone M3)

The cost of the first search of a session (`scripts/latency.sh --first_call`, 20 new
sessions, client ms, median; 1M x 128, warm page cache):

| | first call | second call |
|---|---:|---:|
| HNSW `_snap`, fenced, no advice | 13.0 | 7.6 |
| the same with `madvise(MADV_WILLNEED)` on the mapping | 12.9 | 7.5 |
| the same with `MAP_POPULATE` (every page mapped at open) | 22.3 | 7.9 |
| flat `_snap`, fenced, no advice / WILLNEED / POPULATE | 18.1 / 18.1 / 23.5 | 12.7 / 12.7 / 12.7 |
| `vversion()`, fenced (no index at all) | 11.3 | 8.6 |
| tiny index (16 vectors), fenced | 9.8 | 7.1 |
| HNSW `_snap`, mixed (vsearch in the Vertica process) | 2.4 | 2.1 |

Fenced, the first call pays about 2.7 ms for the new fenced process and its first
function call (vversion and the tiny index show it without a large file) and about
2.7 ms for the new mapping of a 1M-vector index (page faults, the visited array);
flat and HNSW pay the same, so the page faults are the smaller part. `MAP_POPULATE`
maps all 150,000 pages up front and is slower. `MADV_WILLNEED` costs nothing
measurable with the file in the page cache and asks the kernel to read the file ahead
when it is not (after a restart; not measured). vvector keeps WILLNEED on every new
query mapping. vload needs no advice: it reads the whole new file to verify its
checksum, which leaves it in the page cache. Unfenced the mapping is kept by the
Vertica process across sessions, so only the first query after a new snapshot pays.

### Repeated at milestone M3 (Vertica 26.2.0-1, the VM)

`scripts/latency.sh`, 200 runs, client ms, median / p99. The query path did not
change; the numbers are those of M2 within the noise.

| Shape | fenced | mixed |
|---|---:|---:|
| `SELECT 1` | 0.86 / 1.10 | 0.99 / 1.23 |
| HNSW `sift_hnsw_snap` (balanced) | 7.06 / 8.08 | 1.94 / 2.79 |
| HNSW `sift_hnsw_delta`, freshness exact, empty delta | 8.14 / 10.21 | 3.04 / 5.08 |
| HNSW `vknn ... FROM dual` | 7.41 / 8.23 | 1.57 / 1.95 |
| flat `sift_snap` | 12.66 / 13.80 | 5.08 / 5.53 |

### Repeated at milestone M4 (Vertica 26.2.0-1, the VM)

`scripts/benchmark.sh` (latency.sh inside it), 200 runs, client ms, median / p99. The float path is
the same as at M3; sq8 adds an index with codes.

| Shape | fenced | mixed |
|---|---:|---:|
| `SELECT 1` | 0.83 / 1.11 | 0.83 / 1.01 |
| HNSW `sift_hnsw_snap` (balanced) | 7.60 / 7.94 | 1.84 / 2.21 |
| HNSW `sift_hnsw_delta`, freshness exact, empty delta | 8.60 / 9.13 | 2.86 / 3.47 |
| HNSW `vknn ... FROM dual` | 7.97 / 8.65 | 1.54 / 2.01 |
| HNSW with sq8 `sift_sq8_snap` (balanced, 2 x k rescored) | 7.53 / 7.97 | 1.81 / 2.12 |
| HNSW with sq8 `vknn` | 7.91 / 8.29 | 1.51 / 1.78 |
| flat `sift_snap` | 10.98 / 11.50 | 5.52 / 5.80 |

One search spends about 0.1 ms in the engine: sq8 saves a few hundredths of a millisecond of it, so
a single statement costs the same. Batches show the engine: 1000 queries at balanced take 20 ms
with sq8 against 28 ms without (mixed), 11 ms against 14 ms at fast.

Engine (bench_hnsw, SIFT1M, ef 100, recall@10 0.983 in every row), queries per second on 1 / all
threads:

| | VM (8 aarch64 cores) | x86 node (10 cores, AVX-512, g++ 8.5) |
|---|---:|---:|
| hnswlib | 7,151 / 43,060 | 3,760 / 34,719 |
| vvector float | 7,839 / 48,326 | 3,970 / 35,387 |
| vvector sq8, 2 x k rescored | 11,242 / 72,396 | 7,007 / 59,583 |
| vvector sq8, no rescoring (recall 0.970) | 11,973 / 73,536 | 7,244 / 62,248 |

The 4-node cluster (Enterprise mode, SQL, client ms median): `SELECT 1` 3.8 fenced / 3.4 mixed; HNSW
`_snap` balanced 13.3 / 6.4, with sq8 14.4 / 6.1; `vknn` 13.5 / 6.0; empty delta 17.3 / 9.2 (with the
journal replica, made by default there). 1000 queries at balanced: 77 / 49 ms, with sq8 66 / 34 ms.
Refresh: full HNSW 80 s (sq8 87 s; the vbuild statement 64 s of it), incremental after 1000 + 500
changes 16.6 s (vbuild 8.4 s, vload 5.3 s: every node stores and loads the whole snapshot).

The float kernels on x86: 16-byte vectors (committed M3 code) against 32-byte vectors, same node,
SIFT1M: HNSW ef 100 3,853 against 3,953 q/s on one thread (+2.6%), 34,912 against 35,878 on ten;
the HNSW search waits for memory, so the wider registers help little there. On this node
vvector and hnswlib (built with -march=native, AVX-512) are equal: the build took 52 s and 51 s.

### Repeated at milestone M6 (Vertica 26.2.0-3, the 4-node cluster; the VM)

The whole breakdown of "Where a single query spends its time", on the 4-node Enterprise cluster
(x86_64, AVX-512, 10 cores and 78 GB per node), SIFT1M indexes `sift` (flat) and `sift_hnsw`
(balanced), journal replica present (the default there), `scripts/latency.sh`, 200 runs. Client
median in ms (server median in brackets):

| Statement shape | Fenced | Unfenced | Mixed | What it adds (mixed) |
|---|---:|---:|---:|---|
| `SELECT 1` | 3.53 (1.69) | 3.19 (1.63) | 3.25 (1.60) | round trip, parse, plan on a 4-node cluster |
| `vversion() OVER()` | 12.93 (7.04) | 4.47 (2.39) | 4.27 (2.31) | a transform function: 1 ms; fenced 9 ms |
| vsearch, 16-vector index | 14.06 (7.29) | 5.96 (3.30) | 6.15 (3.42) | vsearch setup, parameters, cache check: 1.9 ms |
| HNSW `FROM sift_hnsw_snap` | 13.80 (7.73) | 6.67 (3.98) | 6.60 (3.90) | the HNSW search: 0.5 ms |
| HNSW `FROM dual` | 14.12 (8.06) | 6.68 (4.20) | 6.29 (3.50) | the view: within the noise |
| HNSW `_delta`, empty delta | 18.73 (11.54) | 9.63 (6.09) | 9.61 (6.12) | reading the journal replica: 3 ms |
| HNSW `_delta`, 1000 journal rows | 35.12 (25.84) | 16.55 (12.81) | 16.57 (12.72) | 1000 rows of 128 numbers: 7 ms |
| `vknn ... FROM dual` | 14.20 (7.40) | 5.69 (3.14) | 5.99 (3.47) | the lightest shape |
| `vknn` on a query row | 13.59 (7.12) | 6.10 (3.42) | 6.06 (3.24) | |
| flat `FROM sift_snap` | 20.93 (15.29) | 15.76 (12.64) | 15.24 (12.08) | a scan of 1M vectors on 10 threads: 9 ms |
| flat, query as an ARRAY literal | 45.10 (30.46) | 37.61 (25.44) | 37.86 (24.92) | Vertica parsing a 128-number literal: 23 ms |
| flat `_delta`, empty delta | 24.18 (17.73) | 18.82 (14.88) | 18.46 (14.57) | |
| flat `_delta`, 1000 journal rows | 30.55 (23.60) | 22.70 (18.56) | 22.32 (18.26) | |
| flat, `threads=1` | 107.81 (95.83) | 92.17 (86.13) | 92.29 (86.49) | one thread instead of ten |

First search of a new session (HNSW `_snap`, client ms, 20 sessions, first / second statement):
fenced 41.8 / 18.3, unfenced 13.4 / 7.7, mixed 12.7 / 7.7; `vknn` fenced 41.0 / 14.9, mixed 7.2 /
5.8. Fenced, the new session starts its own fenced process and maps the index.

Against milestone M4 on the same cluster (HNSW `_snap` 13.3 fenced / 6.4 mixed, empty delta 17.3 /
9.2, `SELECT 1` 3.8 / 3.4) nothing changed beyond the noise: milestone M6 did not touch the query
path except the owner check when a cache file is first mapped. What the cluster adds against the VM:
2.4 ms for any statement, 1 ms instead of 0.3 ms for a transform function, and 23 ms instead of 7 ms
for an ARRAY literal (use the `query` parameter or a query row from a table). Filtered and range
search on this cluster (mixed, 20 runs, `scripts/benchmark.sh`): allow-list of 100 / 10,000 /
100,000 ids from an unsegmented table 9.9 / 14.8 / 43.1 ms, from a segmented table 33.3 / 46.6 /
68.0 ms; range search (k 16384, the radius of the 10th neighbour) 8.1 ms.

The VM at M6 (Vertica 26.2.0-1, the definition-of-done run of `scripts/benchmark.sh` on 1M
generated vectors of 128 dimensions, server medians): HNSW `_snap` 4.40 ms fenced, 1.64 ms
unfenced and mixed, `vknn` 1.20 ms mixed; 1000 queries at balanced 32 ms fenced / 21 ms mixed; the engine
tests on SIFT1M gave 49,400 queries/s (float) and 72,600 (sq8) at ef 100, as at M4.

### 10 million vectors (milestone M4, the 4-node cluster)

The first 10M vectors of BIGANN (SIFT1B, 128 dimensions, uint8 values converted to float) with its
ground truth for 10M (idx_10M), loaded with scripts/load_dataset.sh into one journal (all rows the
same day), indexes flat, HNSW and HNSW with sq8 (m 16, ef_construction 200), fenced build:

| | flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot | 4,959 MB | 6,308 MB | 7,567 MB |
| full refresh (vbuild / vload) | 123 s (76 / 40) | 885 s (826 / 52) | 939 s (870 / 62) |
| incremental, 1000 adds + 500 deletes (vbuild / vload) | 99 s (49 / 44) | 146 s (87 / 53) | 194 s (125 / 61) |
| recall@10 fast / balanced / best / exact | 1.0000 | 0.8246 / 0.9525 / 0.9941 / 1.0000 | 0.8165 / 0.9525 / 0.9943 |
| one search, client ms median, fenced / mixed | 91.5 / 88.7 | 14.7 / 6.8 | 14.4 / 6.3 |
| 1000 queries, balanced, fenced / mixed | 18.3 s / 17.6 s | 156 / 53 ms | 132 / 40 ms |

- ef_search 200 on the HNSW index: recall@10 0.9825. After the incremental refresh: 0.979 (balanced
  against exact, 200 queries). A refresh with nothing changed: 5.7 s, of it 4.1 s the journal
  verification over 10M rows.
- The empty delta (`delta0`) costs 33.5 ms mixed against 6.8 ms for `_snap`, and 1000 journal rows
  51.5 ms: the journal (2,713 MB) is above the replica limit (2,048 MB), and all its rows are in
  one day's partition, so the delta scan reads the version column of all 10M rows on every node.
  A journal with daily partitions prunes all but the recent days (the README layout).
- Build memory and page cache were no problem with 78 GB per node: the three indexes and their
  previous snapshots took 44 GB of cache files per node; 67 GB stayed available.

Repeated at milestone M6 (the segmented snapshot table, the caches on a second data disk through
the index option cache_dir, the same journal restored to 10M rows; scripts/scale.sh):

| | flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| full refresh (vbuild / vload) | 132 s (65 / 55) | 899 s (828 / 65) | 944 s (843 / 94) |
| incremental, 1000 adds + 500 deletes (vbuild / vload) | 99 s (35 / 57) | 144 s (55 / 82) | 157 s (68 / 82) |
| recall@10 fast / balanced / best | 1.0000 | 0.8258 / 0.9528 / 0.9946 | 0.8181 / 0.9521 / 0.9945 |
| one search, client ms median, fenced / mixed | 92.0 / 87.4 | 14.4 / 7.0 | 15.1 / 6.3 |
| 1000 queries, balanced, fenced / mixed | 17.7 s / 17.6 s | 154 / 57 ms | 138 / 52 ms |

The vbuild statement is faster (one copy of the chunks written instead of four); vload is slower,
because the cache files now go to a disk that writes at 102 MB/s instead of 192 MB/s (dd with direct
I/O on node 1; it reads faster, 722 against 511 MB/s). Searches read from the page cache and do not
change. A refresh that changes nothing: 5.6 s, of it 3.5 s the journal verification over 10M rows
(`verify_every` 1).

### A journal of one billion rows (milestone M6, the 4-node cluster)

The journal grows with every change; the index holds the latest row per id. This test keeps the index
small and the journal large: 10 million ids, each written 100 times (one version per day over 100 days,
a delete of 1% of the ids in every tenth version), 8 elements per vector so the numbers show the
journal's costs, not the vector build. 1,000,000,000 rows, 109 GB of storage on 4 nodes, the table
partitioned by the day of the version as the README recommends; a flat index with margin 0.

| Step | Time |
|---|---:|
| load, 100 INSERT ... SELECT of 10M rows each | 457 s |
| full build: latest row of each id out of 1B rows, 10M vectors | 104 s (digest of the journal 12.4 s) |
| read the delta after one day of changes (100,000 rows written again, 10,000 deleted) | 75 ms |
| exact search through the delta view (110,000 journal rows beside the snapshot) | 274 ms |
| incremental refresh of that day, journal verified (`verify_every` 1) | 31.2 s (verification 14.9 s) |
| incremental refresh of the next day, not verified (`verify_every` 0) | 20.5 s |
| refresh with no change, not verified | 1.6 s |
| refresh with no change, verified | 16.2 s (verification 14.6 s) |
| one search of the snapshot (flat, 10M vectors of 8 elements) | 44 ms |

The delta view reads only the newest partition: its cost depends on the rows since the refresh, not on
the size of the journal. The full build and the verification read every row: about 10 million rows per
second for the consolidation and 67 million per second for the digest on this cluster. So on a large
journal the verification is most of a refresh that changes little; `verify_every` N does it at every
Nth refresh. For a journal that keeps growing, the consolidated journal (one row per id) is the
compaction candidate of milestone M7.

### 768 and 1536 dimensions (milestone M6, the 4-node cluster)

Embedding models give 768 or 1536 numbers per vector. No public set of that size with ground truth
was used (decision of 2026-09-24); the data is generated: 1,000,000 vectors of Gaussian clusters
(`scripts/scale.sh --generate=768` and `--generate=1536`, seeded), 1000 queries from the same
distribution, metric cosine, recall@10 against the exact search of the flat index. HNSW m 16,
ef_construction 200, fenced build, caches on the second data disk through the index option
`cache_dir`, one journal (all rows the same day). Vertica 26.2.0-3, 4 nodes, 10 cores each.

| 768 dimensions | flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 2,937 MB | 3,072 MB | 3,808 MB |
| full refresh (vbuild / vload) | 100 s (49 / 42) | 197 s (163 / 28) | 218 s (171 / 43) |
| incremental, 1000 adds + 500 deletes (vbuild / vload) | 77 s (38 / 36) | 82 s (42 / 36) | 89 s (47 / 38) |
| refresh with nothing changed (journal verified) | 2.9 s | 2.7 s | 2.9 s |
| recall@10 fast / balanced / best | 1.0000 | 0.8075 / 0.9714 / 0.9977 | 0.6943 / 0.9330 / 0.9920 |
| one search, client ms median, fenced / mixed | 82.0 / 75.9 | 21.7 / 13.2 | 22.8 / 12.8 |
| vknn, client ms median, fenced / mixed | | 19.5 / 9.7 | 18.6 / 9.5 |
| 1000 queries, balanced, fenced / mixed | 7.1 s / 7.1 s | 159 / 110 ms | 111 / 61 ms |

| 1536 dimensions | flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 5,867 MB | 6,001 MB | 7,470 MB |
| full refresh (vbuild / vload) | 172 s (111 / 57) | 360 s (303 / 53) | 384 s (321 / 59) |
| incremental, 1000 adds + 500 deletes (vbuild / vload) | 130 s (67 / 58) | 130 s (68 / 57) | 156 s (84 / 67) |
| refresh with nothing changed (journal verified) | 3.6 s | 3.5 s | 3.7 s |
| recall@10 fast / balanced / best | 1.0000 | 0.8021 / 0.9602 / 0.9988 | 0.7056 / 0.9310 / 0.9962 |
| one search, client ms median, fenced / mixed | 117.0 / 111.6 | 30.4 / 19.9 | 30.3 / 19.7 |
| vknn, client ms median, fenced / mixed | | 25.4 / 14.2 | 23.7 / 13.3 |
| 1000 queries, balanced, fenced / mixed | 15.1 s / 15.0 s | 254 / 167 ms | 173 / 103 ms |

What the numbers say:
- The float rows are most of every file (the graph is 135 MB at any dimension); sq8 adds a quarter.
  Build memory and the page cache were no problem: 71 GB per node stayed available.
- A flat search reads the whole file once per query: 76 ms for 2.9 GB, 112 ms for 5.7 GB, which is
  the memory bandwidth of the node. HNSW reads a few thousand rows: 13 and 20 ms mixed, against
  6.4 ms for 1M vectors of 128 dimensions on this cluster (most of that is the statement).
- sq8 halves the time of a batch but loses more recall on these vectors than on SIFT (balanced 0.93
  against 0.97 for the float index; SIFT: 0.979 against 0.980). The likely reason: sq8 uses one
  value range for all elements, and the elements of these unit-length vectors differ little, so a
  code step is large against the differences between near neighbours. `precision='best'`
  (4 x k rescored) gives 0.992 and 0.996 and is no slower than the float index at balanced;
  measure recall on your own vectors before choosing sq8 at these sizes (README, int8 quantisation).
- Rescoring more candidates closes most of the gap (measured after the tables on the same data, the
  HNSW indexes built again: the parallel build gives a slightly different graph, so recall moves by up
  to 0.002; fenced, 1000 queries in one statement, median of 3 statements, recall@10 against the
  exact search):

  | balanced, candidates rescored | 768: recall | 768: batch | 1536: recall | 1536: batch |
  |---|---:|---:|---:|---:|
  | 2 x k (the preset until then) | 0.9330 | 113 ms | 0.9320 | 159 ms |
  | 3 x k | 0.9579 | 118 ms | 0.9522 | 168 ms |
  | 4 x k | 0.9646 | 115 ms | 0.9564 | 180 ms |
  | 6 x k | 0.9681 | 122 ms | 0.9580 | 179 ms |
  | float HNSW index, balanced | 0.9732 | 165 ms | 0.9618 | 263 ms |

  The walk on the codes finds the neighbours; the codes only order them less well, so more of them
  must be rescored from the floats. Rescoring costs little next to the walk. From 512 dimensions on
  the balanced preset therefore rescores 4 x k: within 0.009 and 0.006 of the float index and still
  30% faster than it. Below 512 dimensions it stays 2 x k (SIFT: 0.979 against 0.980 already). The
  limit 512 lies between the dimensions measured (128, 768); tests/sql/test_sq8.sh checks the
  preset at 512 and at 16 dimensions.
- The HNSW build takes 163 s at 768 and 303 s at 1536 dimensions: the distance computations grow
  with the dimension, the number of them does not.
- An incremental refresh stores and loads the whole file: at 1536 dimensions about 60 s of vbuild
  statement and 57 s of vload (the disk writes about 100 MB/s), for 1000 changed vectors. This is
  the fixed cost that the incremental transfer of milestone M7 (only the changed chunks move) removes.
- The empty delta costs 35 to 44 ms more than `_snap`: the journals (3 GB and 6 GB) are above the
  journal replica limit and all rows are in one day's partition, as in the 10M test.
- A cache file that no query has touched since its load is the first thing the kernel drops when
  Vertica writes: after the three 1536 builds, vinfo showed 1.5 to 2.4 GB of the 5.7 GB flat file
  resident; the first flat search then read the rest from disk.

### 100 million vectors (milestone M6, the 4-node cluster)

The first 100M vectors of BIGANN (SIFT1B, 128 dimensions, uint8 values converted to float; 13.2 GB of
the bvecs file) with the ground truth for 100M (idx_100M), loaded with scripts/load_dataset.sh into one
journal (all rows the same day, 100M rows), indexes flat, HNSW and HNSW with sq8 (m 16,
ef_construction 200), fenced build, caches on the second data disk through the index option
`cache_dir`, 1000 queries. Vertica 26.2.0-3, 4 nodes, 10 cores and 78 GB each. One index was built at a
time; `refresh_index` chose the file-backed build for all of them (the estimate is above half of the
free memory), so the build ran as page cache: the fenced process showed up to 62 GB (flat), 65 GB
(HNSW) and 70 GB (sq8) of resident memory, but the node's available memory never fell below 50 GB,
and Vertica's own process stayed at 10 GB. scripts/scale.sh; 12 hours for the three indexes.

| | flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 49,591 MB | 63,088 MB | 75,677 MB |
| build memory estimate (`sizing`) | 49,973 MB | 65,854 MB | 78,443 MB |
| full refresh (vbuild / vload) | 1,628 s (1,162 / 109) | 12,872 s (12,056 / 178) | 13,641 s (12,586 / 260) |
| incremental, 1000 adds + 500 deletes (vbuild / vload) | 1,156 s (642 / 140) | 1,739 s (1,025 / 234) | 2,766 s (1,625 / 242) |
| refresh with nothing changed (journal verified) | 18.5 s (14.4) | 50.9 s (43.1) | 81.8 s (74.2) |
| recall@10 fast / balanced / best / exact | 1.0000 (exact, 175 s) | 0.7476 / 0.9034 / 0.9813 / 1.0000 | 0.7406 / 0.9027 / 0.9812 / 1.0000 |
| one search, client ms median, fenced / mixed | 779 / 786 | 13.6 / 7.6 | 15.5 / 6.6 |
| the same, server ms median, fenced / mixed | 771 / 782 | 7.6 / 4.7 | 9.1 / 3.9 |
| vknn, client ms median, fenced / mixed | 775 / 787 | 14.9 / 7.0 | 14.8 / 6.5 |
| exact search over the empty delta, client ms, fenced / mixed | 801 / 821 | 42.0 / 35.3 | 43.8 / 33.6 |
| 1000 queries, balanced, fenced / mixed | 175 s / 176 s (exact) | 822 / 452 ms | 1,143 / 918 ms |
| 1000 queries, fast and best, mixed | | 403 / 554 ms | 1,067 / 1,163 ms |

What the numbers say:
- The HNSW build takes 3.3 hours of vbuild on 10 cores (8,300 vectors per second; 12,100 at 10M): the
  graph (13.5 GB) is built in memory, the rows are sorted into a second file, and the file is then
  written once. The vload in passes of 2 GB writes 50 to 76 GB per node in 109 to 260 s (about 300 to
  460 MB/s per node from the broadcast join).
- A search on the HNSW index costs the same as at 10M: 4.7 ms of server time unfenced (4.0 at 10M),
  7.6 ms fenced. The walk touches a few thousand rows of the 63 GB file; the statement is most of the
  time. The batch of 1000 queries takes 452 ms unfenced (2,200 queries per second on one node).
- Recall falls with the size at the same `ef_search`: balanced (ef 100) gives 0.903 at 100M against
  0.953 at 10M and 0.980 at 1M; best (ef 400 with the M6 presets) 0.981; fast 0.748. The presets are
  tuned for 1M; a 100M index needs a larger `ef_search` (index default `ef_search_default`, or the
  parameter) for the 1M recall, at a cost that grows about linearly with ef. The README says so.
- sq8 does not pay at 100M x 128: its batch is slower than the float index (918 against 452 ms
  unfenced), because the walk on the codes is followed by the rescoring of 2 x k candidates from the
  float rows of a 76 GB file (random reads that miss the page cache more often than on a 6 GB file),
  and recall is the same. sq8 is for memory, not speed, at this size; at 128 dimensions the codes save
  nothing (the file holds both). A future graph on the codes without float rows (PLAN 18.3) would.
- The flat index is the correctness reference: one query reads 50 GB (780 ms from the page cache,
  about 65 GB/s), 1000 queries take 175 s. `precision='exact'` on the HNSW index costs the same.
- An incremental refresh before milestone M7 rewrites and reloads the whole snapshot: 1,156 s for
  1000 changes on the flat index, 1,739 s on HNSW, 2,766 s with sq8 (vbuild copies 50 to 76 GB through
  a file, vload writes it on every node). This is the cost that the incremental transfer of M7
  removes (only the changed bytes move; measured below).
- A refresh that changes nothing verifies the journal digest over 100M rows: 14 s while the table was
  in the page cache, 43 and 74 s later, when the snapshot files had pushed it out (the digest reads the
  vectors too, 50 GB of storage; `verify_every` N spreads that cost).
- The exact search over the empty delta costs 30 ms more than `_snap` (the 100M-row journal in one
  day's partition, above the replica limit), as at 10M.
- Three active snapshots (189 GB per node) do not fit the 78 GB of a node together: each build evicts
  the others' files. Measured right after the sq8 build, with the HNSW file down to 4 MB resident on
  node 1 (`vinfo`): the first search took 45 s, the next nine (other queries) 10 to 25 s, then five
  repeats of one query 85 ms, a batch of 1000 queries 129 s and the next one 2.4 s; the file was fully
  resident again after that. The cause is the read-ahead advice a new query mapping gives: MADV_WILLNEED
  over the whole file, which the kernel serves before it returns, 63 GB at about 1.4 GB/s, and the disk
  stays busy with it while the next searches read their pages. The flat index behaved the same: 45 s
  for the first two searches, then 1.1 to 1.7 s (from the page cache 0.78 s: with the HNSW file
  resident the 50 GB do not fit next to it), the batch 174 s twice. So at this size an index must fit
  in the node's page cache beside Vertica's working set, or every eviction costs a minute; `load_all`
  re-reads a file that was pushed out. The M6 decision on `hot_dir` (a pinned second file with the
  graph, ids and codes): not built; the walk itself, once resident, costs 4.7 ms at 100M as at 10M, the
  float rows are half of every search's reads, and a tmpfs `cache_dir` pins an index that fits. What
  the numbers ask for is a bound on the read-ahead: a mapping should not stall its first search for a
  file that cannot be read in seconds (milestone M7: the advice is given section by section in the
  order the walk needs them, ids, id index, tombstones, codes, graph, float rows, and stops at a budget;
  the rest is paged in on demand).
