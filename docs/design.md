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
VM. vvector keeps a copy of the vectors (and from milestone M2 an HNSW graph
over them) as a snapshot, loads it on every node and answers the same query
from it in 5 to 12 milliseconds (below).

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
- The snapshot and every cache file: 256 bytes plus `4 x row_stride + 8` bytes
  per vector (plus graph and codes from M2 and M4).
- vsearch maps the snapshot and copies nothing from it. It holds the queries
  and the live journal vectors: `4 x row_stride` bytes each, and a bitset of
  one bit per snapshot vector when the journal has rows.

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
  snapshots.

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
| 3. No gather across nodes | single node: does not apply; measured on the Eon cluster at M2 |
| 4. No file system work per call | ACTIVE and OPTIONS trusted for 200 ms, the mapping kept: a warm call makes no system call for the cache |
| 5. Function setup | 0.5 ms more than an empty transform function (unfenced); parameters are read once, the query parsed once |
| 6. Output | k rows written in one pass; no string work |
| 7. Lightest function shape | verified: with `isExploder` a transform function can be called without OVER() and beside other columns (VERTICA_NOTES); `vknn` is prototyped at M2 |
| 8. Prepared statements | not measured yet: the VM has the Vertica ODBC driver and JDBC jar but no driver manager and no Java; needs an install (asked) |
| 9. Resource pool | not measured: it needs a new pool (a database change); a statement this short shows no thread setup in the numbers above |
| 10. Fenced against unfenced | fenced adds about 6 ms per statement; see the recommendation below |
| Query as a parameter (decision 4) | saves about 7 ms per statement against an ARRAY literal: the literal is parsed by Vertica |

Recommendation from the numbers: fenced mode costs about 6 ms per statement,
more than the search itself. For a service that runs single searches, deploy
with `FENCED=mixed` (vbuild, vload and vconfig stay fenced; vsearch, vinfo
and vversion run in the Vertica process). For batch searches (1000 queries in
one statement) the difference is below 2% (957 against 944 queries per
second), and fenced is the safer choice.

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
