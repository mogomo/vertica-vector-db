# vvector design notes

This file explains the decisions. The user's view is in the README.

## Pieces

    vector table --vbuild--> vvector.snapshot --vload--> node cache file --mmap--> vsearch
    (customer)               (UNSEGMENTED ALL NODES)     (/tmp/vvector/<index>/)

- `src/engine` is plain C++17 without Vertica includes. `src/udx` holds thin
  Vertica adapters only.
- The snapshot table is unsegmented on all nodes. It is backed up and
  replicated with the database. The cache file is only a copy of it.
- A missing or damaged cache file is never a data loss: run vload again
  (`CALL vvector.load_all('<index>')`).

## Why an index outside SQL

Vertica 26.2 has no vector type and no vector index. Vectors are stored as
`ARRAY[FLOAT]` (or INT, NUMERIC), and the built-in functions
`COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and `VECTOR_MAGNITUDE` compare
two arrays. A nearest-neighbour query is therefore a full scan with a top-k
sort (verified with EXPLAIN: STORAGE ACCESS, then SORT [TOPK]). vvector keeps
a copy of the vectors, and later an HNSW graph over them, as a snapshot, loads
it on every node and answers k-nearest-neighbour queries from it.

## vload on every node

`vload(...) OVER(PARTITION NODES)` runs one function instance on every node
that receives input rows. Measured on Vertica 26.2 (in the earlier graph
project this code comes from):

- With `vvector.snapshot` (unsegmented) as the only input, Vertica reads the
  replicated table on one node only. On a 3-node Eon cluster only one node ran
  the load. On a single node this cannot be seen.
- So the input is driven by a segmented table. `vvector.probe(k)` holds 1024
  rows, segmented by hash, so every node has some. `vvector.vnode(k)
  OVER(PARTITION NODES)` returns the smallest k stored on each node. The chunks
  are cross joined to those probe rows: one probe row per node, so every node
  gets every chunk exactly once. The join is local because the snapshot table
  is replicated.
- The mapping is computed inside the same statement, so it follows node and
  shard changes. vload returns one row per node that loaded.
- vinfo reads `vvector.probe` for the same reason and answers once per node.

`tests/sql/test_snapshot.sh` fails if the number of nodes that report `loaded`
differs from the number of UP nodes. So far tested on the single-node VM only;
the multi-node test comes with the Eon cluster (milestone M2).

## Cache rules

- Layout: `<cache_dir>/<index>/<snapshot_id>.vv` and `<cache_dir>/<index>/ACTIVE`.
- cache_dir: function parameter, then session parameter
  (`ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'`), then `/tmp/vvector`.
- vload writes to a temporary file, verifies size, structure and checksum,
  renames it into place and then replaces ACTIVE by rename. A failed load
  leaves the cache as it was.
- vload keeps the new and the previously active snapshot file. It removes
  other files in the index's directory only if they start with the vvector
  magic. It never touches anything else. Index names are limited to letters,
  digits and underscore, so a name cannot point outside cache_dir.
- Two vload runs for the same index at the same time are not supported.
- The mapping of the active file is kept by the process and shared by later
  calls: a new mapping pays a page fault for every page a search touches (in
  the graph project this cost several times the search itself at one billion
  rows). A kept mapping is dropped at the next query of that process when its
  file is gone or replaced, or when its index has a newer active snapshot.
  Unfenced, the mapping lives in the Vertica process (address space only: the
  pages are file cache of the operating system). Fenced, the process ends with
  the session, so the first call of a session pays the faults.
- Directories are created with mode 0700, files with 0600.

## Memory

vbuild holds the input while rows arrive: 8 + 4 x dims bytes per vector,
then the snapshot of the same size (twice that if the input was not ordered by
id). Example: 1 million vectors of 768 dimensions: about 3.1 GB, 6.2 GB peak.
In fenced mode this memory belongs to the fenced process; the limit is
`FencedUDxMemoryLimitMB` (-1 = no limit). A streaming build (one chunk of
memory) is possible because the input arrives ordered by id; it is not written.

vsearch maps the snapshot and copies nothing.

## Freshness: exact results between refreshes

The vector table is a journal: rows are only inserted, a delete is a row with
the delete flag, and for every id the row with the latest version wins. The
version is a column of the customer's table (insertion time or an increasing
INT). Vertica's `epoch` pseudo-column is deliberately not used: it is not
unique, it can change, and it cannot be part of a projection.

- The rows a query must apply come from the view `<schema>.<index>_delta`:
  `ver_col > boundary` plus one sentinel row, because Vertica does not call a
  transform function on empty input. Every row carries the id of the snapshot
  the view belongs to.
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
- Consolidation for the build: the latest row per id by version, kept if it is
  not a delete. Without a version column every id must appear once.
- How vsearch will apply the delta (M1): ids that appear in the delta are
  masked in the snapshot; their latest delta row, if it is not a delete, is
  searched exactly next to the snapshot; the two top-k lists are merged.
  Applying is idempotent, so rows inside the margin (in the snapshot and in
  the delta) are harmless.
- Stale cache: if the node's active snapshot is older than the snapshot id on
  the delta rows, vsearch fails with "snapshot cache stale on <node>: run
  vload". Newer is fine (a refresh is in flight).
- Refresh order, never changed: boundary, build, insert chunks, vload on all
  nodes, update manifest and view, delete older snapshots.
