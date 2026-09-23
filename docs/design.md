# vvector design notes

This file explains the decisions and records the measurements behind them.
The user's view is in the README.

## Pieces

    vector table --vbuild--> vvector.snapshot --vload--> node cache file --mmap--> vsearch
    (customer)               (UNSEGMENTED ALL NODES)     (/tmp/vvector/<index>/)

- `src/engine` is plain C++17 without Vertica includes. `src/udx` holds thin
  Vertica adapters only: they read rows and parameters, call the engine and
  write rows.
- The snapshot table is unsegmented on all nodes. It is backed up and
  replicated with the database. The cache file is only a copy of it.
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
- The 16 lanes are four native 16-byte vectors. A single 64-byte vector type
  is simpler to write, but g++ keeps such a variable in memory on aarch64 (the
  hot loop stored and reloaded it through the stack): changing the type made
  a single query 3.2 times faster with the same result bits.
- Flat search (`src/engine/flat.h`): candidates are ordered by (key, id), so
  ties go to the smaller id and the result does not depend on the order of
  the work. With few queries the rows are split over the threads and the
  per-thread top-k lists are merged; with many queries (or a large k) the
  queries are split. A tile of about 256 KB of rows stays in the L2 cache
  while every group of 4 queries is scored against it, so each row is read
  from memory once per tile, not once per query.
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
  Another thread can reach a node through its upper levels before that
  node's insert gets down to a lower level and link itself to it there;
  those links are merged when the insert writes the node's list, not
  overwritten (a first version lost them: 0.5% of the nodes of a 4-thread
  build had no incoming link).
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

## vload on every node

`vload(...) OVER(PARTITION NODES)` runs one function instance on every node
that receives input rows. Measured on Vertica 26.2 (in the earlier graph
project this code comes from):

- With `vvector.snapshot` (unsegmented) as the only input, Vertica reads the
  replicated table on one node only. On a 3-node Eon cluster only one node ran
  the load. On a single node this cannot be seen.
- So the input is driven by a segmented table. `vvector.probe(k)` holds 8192
  rows, segmented by hash, so every node has some. `vvector.vnode(k)
  OVER(PARTITION NODES)` returns the smallest k stored on each node. The chunks
  are cross joined to those probe rows: one probe row per node, so every node
  gets every chunk exactly once. The join is local because the snapshot table
  is replicated.
- The mapping is computed inside the same statement, so it follows node and
  shard changes. vload returns one row per node that loaded.
- vinfo and vconfig read `vvector.probe` for the same reason and answer once
  per node.

## Cache rules

- Layout: `<cache_dir>/<index>/<snapshot_id>.vv`, `<cache_dir>/<index>/ACTIVE`
  and `<cache_dir>/<index>/OPTIONS` (the index defaults of set_index_options).
- cache_dir: function parameter, then session parameter
  (`ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'`), then `/tmp/vvector`.
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
- vsearch maps the snapshot and copies nothing from it. It holds the queries
  and the live journal vectors: `4 x row_stride` bytes each, and a bitset of
  one bit per snapshot vector when the journal has rows. A graph search keeps
  2 bytes per vector per search thread of visited marks in the process
  between calls (2 MB per thread for 1M vectors).

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
  is idempotent, so rows inside the margin (in the snapshot and in the delta)
  are harmless. `freshness='snapshot'` (the default) ignores journal rows.
  `tests/sql/test_search.sh` checks the result against the full scan of the
  live rows after 1000 adds, 1000 deletes and 500 replacements.
- Stale cache: if the node's active snapshot is older than the snapshot id on
  the view's rows, vsearch fails with "snapshot cache stale on <node>: run
  vload". Newer is fine (a refresh is in flight).
- Refresh order, never changed: boundary, build, insert chunks, vload on all
  nodes, index defaults on all nodes, update manifest and views, delete older
  snapshots. An incremental refresh (next section) keeps the order; only the
  build reads less.

## Incremental refresh (milestone M3)

`refresh_index` builds a new snapshot from the active one and the journal
rows after the previous boundary, the same rows the delta view shows,
consolidated per id (latest row, a delete wins a tie; deletes are passed to
vbuild with `del = true`). vbuild gets `base_snapshot` and reads the base from
the cache of its node, verified like a vload (checksum, ids, graph links): a
damaged base would otherwise be copied into every later snapshot with a fresh
checksum.

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
journal whose rows up to the previous boundary changed in number. That count
(`manifest.boundary_rows`) is taken at every refresh for the new boundary: a
row committed later always has a newer version (that is the boundary rule),
so the count only changes when rows are removed (physical DELETE, dropped
partitions) or written with old versions by hand. It is one scan of the
version column.

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
