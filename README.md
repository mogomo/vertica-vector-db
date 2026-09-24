# vertica-vector-db (vvector)

Nearest-neighbour vector search inside Vertica. vvector is a C++ UDx library
that keeps an index of the vectors stored in a Vertica table and answers "the
k vectors closest to this one" from SQL. The index lives in Vertica, is loaded
on every node, and every query can see the rows written since the last
refresh.

**Status: milestone M5 (filtered search and vector functions).** Two index
types: `hnsw` (a graph index, approximate, the default) and `flat` (exact),
each optionally with int8 codes (`sq8`) that make searches faster while the
returned scores stay exact. k-nearest-neighbour search works for the metrics
l2, cosine, dot and l1, for one query or thousands in one statement, with or
without the rows written since the last refresh, limited to a list of
allowed ids (filtered search) or to a radius (range search). Vector
functions add the arithmetic Vertica lacks (sum, difference, average,
normalisation, l1, Hamming and Jaccard distance). A refresh adds only the changes since the
last one to the index and rebuilds it in full only when that is needed. Exact
results are tested to equal a full scan with Vertica's built-in functions;
recall is measured on SIFT1M, also after 100 incremental refreshes. Tested on
Vertica 26.2 on one node (aarch64, Rocky Linux 9, g++ 11.5), on a 3-node Eon
cluster and on a 4-node Enterprise cluster (x86_64, Red Hat Enterprise Linux
8, g++ 8.5), fenced, unfenced and mixed. Treat this as a preview: try it on
your own systems before you rely on it.

Contents: [Why](#why) · [Quick start](#quick-start) · [Install](#install) ·
[Prepare a table](#prepare-a-table) · [Register, refresh, schedule](#register-refresh-schedule) ·
[Search](#search) · [Index types and tuning](#index-types-and-tuning) ·
[Freshness explained](#freshness-explained) · [Vector functions](#vector-functions) ·
[Operations](#operations) · [Performance and results](#performance-and-results) ·
[Restrictions](#restrictions-and-not-supported) · [Files](#files)

## Why

Vertica 26.2 has no vector type and no vector index. Vectors are stored as
arrays (`ARRAY[FLOAT]`, `ARRAY[INT]` or `ARRAY[NUMERIC]`), and the built-in
functions `COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and
`VECTOR_MAGNITUDE` compare two arrays. A nearest-neighbour query in SQL is a
full scan of the table:

    SELECT id, VECTOR_L2(vec, ARRAY[0.1, 0.2, 0.3]) AS distance
    FROM app.docs ORDER BY distance LIMIT 10;

On 1,000,000 vectors of 128 dimensions that takes 7.7 seconds on the test
machine. vvector answers the same query in 1.9 ms with an HNSW index (7.1 ms
when the search function is fenced), or with exactly the same result in 5 ms
with a flat index (12 ms fenced). The index is a snapshot of the vectors that
is stored in Vertica, cached on every node, and kept exact by applying the
rows written since the last refresh.

## Quick start

On a Vertica node, with the environment variables of `vsql` set (see
[Install](#install)):

    make && make test && make deploy

    vsql -c "CREATE SCHEMA app;
             CREATE TABLE app.docs (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                                    ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP());
             INSERT INTO app.docs (id, vec) VALUES (1, ARRAY[1.0, 0.0, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (2, ARRAY[0.9, 0.1, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (3, ARRAY[0.0, 1.0, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (4, ARRAY[0.0, 0.0, 1.0]);
             INSERT INTO app.docs (id, vec) VALUES (5, ARRAY[0.5, 0.5, 0.0]); COMMIT;"
    vsql -c "CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', 0);"
    vsql -c "CALL vvector.refresh_index('docs');"
    vsql -c "SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                    USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
             FROM app.docs_snap;"

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

`register_index` creates an HNSW index unless you ask for `flat` (see
[Index types and tuning](#index-types-and-tuning)). The last argument, the
margin, is 0 here so that the refresh builds the rows written just before it;
that is safe on one node with `CLOCK_TIMESTAMP()` versions. With the default
(NULL = 60 s) a refresh builds the rows older than 60 seconds and serves the
newer ones through the delta (see [Freshness explained](#freshness-explained)). The same query with the
built-in function returns the same ids and scores (vvector computes in 32-bit
floats, so scores agree to about 7 digits):

    SELECT id, COSINE_SIMILARITY(vec, ARRAY[1, 0.2, 0]) AS score FROM app.docs ORDER BY score DESC LIMIT 3;

     id |       score
    ----+-------------------
      2 | 0.996240588195683
      1 |  0.98058067569092
      5 | 0.832050294337844

## Install

### Prerequisites

- Vertica 26.x (tested: 26.2.0-1 single node, 26.2.0-2 Eon with 3 nodes,
  26.2.0-3 Enterprise mode with 4 nodes) with the C++ SDK in `/opt/vertica/sdk`
  (another place: `make SDK_HOME=...`).
- g++ with C++17 (tested: 11.5 on aarch64, 8.5 on x86_64) and GNU make, on a
  Vertica node: `CREATE LIBRARY` reads the .so from the initiator node's file
  system and copies it to the other nodes. No CPU flags are needed: on x86_64
  the distance code is compiled for SSE2, AVX2 and AVX-512 and the best one is
  picked at load time, with bit-identical results on every level.
- Step by step on x86_64 (Red Hat 8, Eon cluster), with the expected output:
  [docs/build-x86.md](docs/build-x86.md).
- A database user that may create a schema, a library, functions and a role
  (dbadmin, or a user with those rights).
- For the tests and the benchmark only: Vertica's packages `VectorOps`
  (`VECTOR_L2`, `COSINE_SIMILARITY`, `DOT_PRODUCT`: the tests compare with
  them) and `approximate` (`APPROXIMATE_PERCENTILE`: the benchmark's medians).
  A new database usually has them; if not, see
  [docs/build-x86.md](docs/build-x86.md), "Problems seen". vvector itself
  needs neither.

### Build, test, deploy

    make                      # build/libvvector.so
    make test                 # engine unit tests, no database needed
    make deploy               # install into the database, fenced (the default)
    make deploy FENCED=no     # every function inside the Vertica process
    make deploy FENCED=mixed  # vbuild, vload, vconfig, vnode fenced; vsearch, vknn, vinfo, vversion not fenced
    make deploy SEARCH=public # searching for every user (default: the role vvector_search)
    make undeploy             # remove the library and its functions; tables and data stay
    tests/sql/run_all.sh      # integration tests: fenced, unfenced, mixed (creates test schemas)

What the modes mean:

| Mode | Where the functions run | For | Risk |
|---|---|---|---|
| `yes` (default) | a separate fenced process per session | safety first | none for the node; about 6 ms more per statement |
| `mixed` | build and load fenced, search in the Vertica process | single searches with low latency | a fault in vsearch, vknn, vinfo or vversion would stop the node |
| `no` | everything in the Vertica process | the lowest latency and fastest refresh | a fault in any vvector function would stop the node |

The measurements behind this are in [Performance and results](#performance-and-results).

Every script reads the connection from the environment: `VSQL_HOST`,
`VSQL_PORT`, `VSQL_USER`, `VSQL_PASSWORD`, `VSQL_DATABASE`. Every script
accepts `--help`, and every script that changes the database accepts
`--echo_only`: it prints the commands and changes nothing.

### What the install creates

Everything is in schema `vvector`, except the functions that build and load
snapshots, which are in schema `vvector_admin`:

| Object | What it is |
|---|---|
| `vvector.snapshot` | table (segmented by snapshot_id and byte_offset): the index snapshots in chunks of 8 MB |
| `vvector.manifest` | table: one row per index with its source, options and state |
| `vvector.probe` | table (8192 rows, segmented): makes node-wise functions run once on every node |
| `vvector.snapshot_seq` | sequence of snapshot ids (never reused) |
| role `vvector_admin` | may build, load and manage indexes; holds `vvector_search` |
| role `vvector_search` | may search (the functions in `vvector`) |
| functions in `vvector` | `vsearch`, `vknn`, `vinfo`, `vversion` (search and information) |
| functions in `vvector_admin` | `vbuild`, `vload`, `vconfig`, `vnode` (build and load; `refresh_index` and `load_all` call them) |
| procedures | `register_index`, `set_index_options`, `set_journal_replica`, `refresh_index`, `load_all`, `status`, `sizing`, `schedule_refresh`, `unregister_index` |

Rights:

- **Searching needs the role `vvector_search`**: `vsearch`, `vknn`, `vinfo`,
  `vversion` and the vector functions (`GRANT vvector_search TO someone;
  ALTER USER someone DEFAULT ROLE vvector_search;`, or `SET ROLE
  vvector_search` in the session). `make deploy SEARCH=public` opens
  searching to every user, as it was before milestone M6 (a role granted to
  PUBLIC is not enabled for anyone in Vertica, so `GRANT vvector_search TO
  PUBLIC` does not do that). The procedure
  `sizing` is open to everyone. The role covers every index: a search needs
  no view (`vknn` and `vsearch ... FROM dual` with the `query` parameter
  search any index by its name), so SELECT on the views of an index does not
  decide who may search it. The views protect the journal rows only: the
  `_delta` view shows the rows of the source table, and it is not granted to
  anyone. Grant SELECT on the views to the users who should run exact
  searches (with the changes since the refresh). Whoever has the role can
  find the k nearest ids of any index.
- **The manifest** (`vvector.manifest`: source tables, boundaries, who runs a
  refresh) is readable by `vvector_admin` only; queries never read it.
- **Building and loading needs the role `vvector_admin`**
  (`GRANT vvector_admin TO someone; ALTER USER someone DEFAULT ROLE vvector_admin;`,
  or `SET ROLE vvector_admin` in the session): the functions in schema
  `vvector_admin` and every procedure except `sizing`. `vbuild` is there
  because an incremental build reads a whole snapshot from the node cache:
  open to everyone, it would hand out the vectors of every index.
  `vvector_admin` holds `vvector_search`, so an index administrator can also
  search.
  `schedule_refresh` also needs a superuser: Vertica lets only a superuser
  create a trigger.
- Rights are given per schema because Vertica 26.2 cannot grant a single
  function that has an ARRAY argument.

A user who registers and refreshes an index needs, besides `vvector_admin`,
USAGE and CREATE on the schema of the source table (the views go there) and
SELECT on the table. `tests/sql/test_rights.sh` checks this with a user that
has no other rights: register, full and incremental refresh, status, search
and unregister work; with the role switched off (`SET ROLE NONE`) the same
user can still search, but not build, load, refresh or read
`vvector.snapshot`.

## Prepare a table

vvector indexes a table with an id and a vector column. Changes are written
as new rows: the table is a journal.

    id | vec                | del   | ts
    ---+--------------------+-------+---------------------------
     7 | [0.1, 0.2, ...]    | false | 2026-09-20 10:00:00+00     vector 7 added
     8 | [0.4, 0.1, ...]    | false | 2026-09-20 10:05:00+00     vector 8 added
     7 | [0.3, 0.3, ...]    | false | 2026-09-21 09:00:00+00     vector 7 changed: the newer row wins
     8 |                    | true  | 2026-09-22 11:00:00+00     vector 8 deleted

- **id**: INT (64-bit), unique per vector. Business keys and payload stay in
  your own tables; join them to the results by id. Rows with a NULL id are
  ignored (by the refresh and by the delta view).
- **vec**: `ARRAY[FLOAT]` (recommended), `ARRAY[INT]` or `ARRAY[NUMERIC]`.
  Every vector of one index has the same number of elements. vvector stores
  and computes in 32-bit floats.
- **del** (optional): BOOLEAN, true = deleted; or INT, +1 added and -1 deleted.
- **ts** (optional): the version. For every id the row with the latest
  version wins; with equal versions a delete wins. Use `TIMESTAMPTZ NOT NULL
  DEFAULT CLOCK_TIMESTAMP()`: the database sets the time of the write, which
  is what makes the results exact (see [Freshness](#freshness-explained)).
  TIMESTAMP and increasing INT versions are accepted too. The column must be
  NOT NULL: `register_index` refuses one that is not (a row without a version
  would be in no delta and no refresh).

Recommended table, partitioned by the date of the version so new rows stay in
their own storage and are found quickly:

    CREATE TABLE app.docs (
        id   INT NOT NULL,
        vec  ARRAY[FLOAT],
        del  BOOLEAN NOT NULL DEFAULT FALSE,
        ts   TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP()
    )
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE
    GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);

    INSERT INTO app.docs (id, vec) VALUES (7, ARRAY[0.1, 0.2, 0.3]);   -- add or change
    INSERT INTO app.docs (id, del) VALUES (8, TRUE);                    -- delete

To keep the journal append-only, grant the users that write to it INSERT
only, no UPDATE or DELETE on the table.

A physical `DELETE` or `UPDATE` on the table (or a dropped partition) is
allowed, but queries see it only after the next refresh. Any physical change
to rows up to the boundary of the last refresh is noticed by the next refresh
(with the default `verify_every` 1), which then rebuilds the index in full
(see [refresh_index](#refresh_index)). Rows after the boundary are read by
every refresh anyway. A table without a version column is a **static index**: queries see the
snapshot only, every id must appear once, changes show up at the next refresh,
and every refresh is a full build.

## Register, refresh, schedule

All procedures need the role `vvector_admin`, except `sizing` (everyone);
`schedule_refresh` also needs a superuser.

### register_index

    CALL vvector.register_index(index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin [, index_type]);
    CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', 0);

| Argument | Meaning |
|---|---|
| index_name | 1 to 64 letters, digits or underscores |
| source_table | `schema.table` |
| id_col, vec_col | the id (INT) and vector column |
| op_col | delete flag (BOOLEAN or INT), or NULL when rows are only added; needs ver_col |
| ver_col | version (TIMESTAMPTZ, TIMESTAMP or INT), or NULL for a static index |
| metric | `l2` (VECTOR_L2), `cosine` (COSINE_SIMILARITY), `dot` (DOT_PRODUCT) or `l1` (Manhattan distance); fixed for the life of the index |
| margin | how far the delta boundary stays behind the refresh: seconds for a timestamp version (NULL = 60; 0 is safe on one node with `CLOCK_TIMESTAMP()` versions, a cluster needs a few seconds for clock differences), units of the column for an INT version (required). A refresh builds the rows up to the boundary; newer rows stay in the delta until the next refresh (see [Freshness explained](#freshness-explained)) |
| index_type | `hnsw` (default: a graph index, fast and approximate) or `flat` (exact, reads every vector); see [Index types and tuning](#index-types-and-tuning) |

It creates the views `<schema>.<index>_snap` and, with a version column,
`<schema>.<index>_delta` in the schema of the source table:

    NOTICE 2005:  vvector: index docs registered. Next: CALL vvector.refresh_index('docs'). Queries read docs_snap (snapshot only) or docs_delta (with the changes since the refresh) in schema app; grant SELECT on them to the users who may search the index.
    NOTICE 2005:  vvector: index docs: journal replica: none: a single node reads the delta locally already

The second line is about the journal replica (see [Operations](#operations)):
on a cluster it names the projection that was made.

### refresh_index

    CALL vvector.refresh_index(index_name [, mode]);
    CALL vvector.refresh_index('docs');                  -- the index's refresh_mode (default auto)
    CALL vvector.refresh_index('docs', 'full');          -- rebuild from every row of the table

The first refresh of the quick start, then refreshes after an added and a
deleted vector (the output of each call):

    NOTICE 2005:  vvector: index docs refreshed: snapshot 959, full build (first build), 5 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.282 seconds; journal digest taken in 0.014 seconds
    NOTICE 2005:  vvector: index docs: journal replica: none: a single node reads the delta locally already

    NOTICE 2005:  vvector: index docs refreshed: snapshot 960, incremental from snapshot 959 (1 vectors appended, 1 tombstoned), 5 vectors of 3 dimensions, 1 tombstones, 0 MB, 0.342 seconds; journal verified in 0.021 seconds
    NOTICE 2005:  vvector: index docs refreshed: snapshot 960 kept, no vector changed since it was built; the delta starts at the new boundary; 0.178 seconds; journal verified in 0.022 seconds
    NOTICE 2005:  vvector: index docs refreshed: snapshot 962, full build (mode full), 5 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.287 seconds; journal digest taken in 0.015 seconds

It takes the delta boundary, builds a new snapshot, stores it in
`vvector.snapshot`, loads it on every node, writes the index defaults to every
node, updates the manifest and the views, deletes snapshots older than the
previous one, and makes, keeps or drops the journal replica. Queries keep
working during a refresh. The first line says what was built and why.

One refresh of an index runs at a time. A second `refresh_index` of the same
index (by hand, or by the schedule) while one runs stops at once with an
error that says since when and by whom the index is being refreshed:

    ERROR 2005:  vvector.refresh_index: index docs is being refreshed since 2026-09-23 16:56:19 UTC (by dbadmin, session v_vdb_node0001-1391:0x241bd). Two refreshes of one index cannot run at the same time: wait until it ends. If it no longer runs (its session was killed or its node went down), the mark is ignored as soon as its session is gone (seen by a superuser, or by the same user), else 6 hours after its start (6, or 4 times the last build time), or a vvector_admin removes it: UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'docs'; COMMIT;

The refresh marks the manifest row (`refresh_started_at`,
`refresh_started_by`) and removes the mark when it ends, also when it fails.
If its session is killed or its node goes down, the mark stays. The next
refresh ignores it at once when the session is gone from
`v_monitor.sessions` and the caller can see that: a superuser sees every
session, another user only its own. A mark of the caller's own session is
ignored too (a refresh of that session failed on a cluster in a way that left
the mark). A refresh started by
[schedule_refresh](#schedule_refresh) runs in a session that
`v_monitor.sessions` does not show; its mark reads "scheduled, internal
session ..." and counts by its age only. A mark is always ignored after 6
hours, or after 4 times the index's last build time when that is longer (a
3-hour build keeps its mark for 12 hours). A `vvector_admin` can remove a
mark with the statement the message gives
(`UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'docs'; COMMIT;`).
`status` shows a running refresh.

There are two ways to build:

- **incremental**: starts from the active snapshot in the cache of the node
  that runs the refresh and reads only the journal rows written after the
  previous boundary (the latest row of each id). A new or changed vector is
  added to the snapshot (and inserted into the graph of an HNSW index); the
  old vector of a changed or deleted id stays in the snapshot as a
  **tombstone**: searches skip it. A row that repeats the vector the index
  already has, or deletes an id that is not there, changes nothing. When
  nothing changed, the snapshot is kept and only the boundary moves ("kept, no
  vector changed").
- **full**: reads every row of the table (the latest row of each id, deletes
  left out) and builds a new snapshot without tombstones.

`mode` (or the index's `refresh_mode`, see `set_index_options`):

| mode | Builds |
|---|---|
| `auto` (default) | incremental; full when the tombstones exceed `tombstone_ratio` (default 0.2) of the snapshot, or after `rebuild_every` incremental refreshes (default: never by count) |
| `incremental` | incremental; the ratio and the count are ignored (`status` warns when the tombstones pass the ratio): schedule `refresh_index(name, 'full')` yourself, for example at night |
| `full` | full, always |

In every mode the build is full when an incremental one cannot give the right
answer: the first build; a static index (no version column); changed build
options (`index_type`, `m`, `ef_construction`, `quantization`) or a new
snapshot format; a node whose cache does not hold the active snapshot; and
journal rows up to the previous boundary that are not the rows the last
refresh saw (a physical `DELETE` or `UPDATE`, a dropped partition, versions
set by hand). The manifest keeps the number of those rows and a digest of
them (the sum of `HASH(id, vector, delete flag, version)`). Every refresh
carries both forward from the journal rows between the previous and the new
boundary, which it reads anyway. To verify them, a refresh computes both again
over every row up to the previous boundary, the vectors included, and compares;
any difference is a full build. The index option `verify_every` says when:
`1` at every refresh (the default), `N` every N refreshes, `0` never. The
verification is the part of a refresh that grows with the whole journal (table
below); with `0` a refresh reads only the new rows, and a physical change
goes unnoticed: then run `CALL vvector.refresh_index('docs', 'full')` after
every physical UPDATE, DELETE or dropped partition yourself. A full build
takes exact values. The first line says what happened, for example
`full build (the journal rows up to the previous boundary changed since the last refresh: same count (20000), other digest (a physical UPDATE, or rows replaced by hand)), ...; journal verified in 0.43 seconds` (900,000 x 128),
or, with `verify_every` 10, `...; journal not verified (verify_every 10, last verified 1 refreshes ago)`.
An index made by a version without the digest gets it at its next refresh by
one scan, without a rebuild ("journal digest taken for the first time").

Measured on the test machine (`scripts/benchmark.sh --parts=incremental`,
fenced; 900,000 SIFT1M vectors of 128 dimensions, then changes of growing
size, each followed by `refresh_index`):

| Change since the last refresh | flat | hnsw |
|---|---:|---:|
| none (the snapshot is kept), journal verified (`verify_every` 1) | 1.0 s | 1.0 s |
| none, not verified (`verify_every` 0) | 0.6 s | 0.6 s |
| 100 adds, 50 deletes | 5.3 s | 6.9 s |
| 1,000 adds, 500 deletes | 5.0 s | 6.9 s |
| 10,000 adds, 5,000 deletes | 5.0 s | 7.3 s |
| 50,000 adds, 25,000 deletes | 5.7 s | 9.5 s |
| full build of the same 930,550 vectors | 8.8 s | 42.1 s |

An incremental refresh costs a fixed part plus a small part per change. The
fixed part is the whole snapshot being written to `vvector.snapshot` and
loaded on every node again (about 4 s for 450 to 600 MB), and the
verification of the journal (0.4 s here, 0.6 to 0.8 s right after a large
insert; none with `verify_every` 0); the build itself
takes 0.25 s for 1000 adds and 500 deletes on 1M vectors. A full build of an
HNSW index spends most of its time on the graph; the graph after 100
incremental refreshes finds as much as a new one (recall@10 0.9845 against
0.9837 at `ef_search` 100 on SIFT1M, `make test DATA_DIR=...`).

### schedule_refresh

    CALL vvector.schedule_refresh('docs', '0 * * * *');      -- every hour, at minute 0
    CALL vvector.schedule_refresh('docs', '*/15 * * * *');   -- every 15 minutes

Creates `vvector.docs_refresh_schedule` and `vvector.docs_refresh_trigger`
(Vertica's CRON schedule; the trigger runs as the definer). Calling it again
replaces the schedule.

### set_index_options

    CALL vvector.set_index_options(index_name, index_type, m, ef_construction, quantization, refresh_mode,
                                   tombstone_ratio, rebuild_every, memory_mode, precision_default,
                                   freshness_default, ef_search_default, threads_default [, verify_every [, cache_dir]]);

    -- keep the node caches of index docs on a data disk (loaded there at once):
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '/data/vvector');

    -- verify the journal every 10 refreshes instead of at every one:
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 10);

    -- queries of index docs apply the journal by default:
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'exact', NULL, NULL);

    -- a denser graph from the next refresh on, and precision best for every query from now on:
    CALL vvector.set_index_options('docs', NULL, 32, 400, NULL, NULL, NULL, NULL, NULL, 'best', NULL, NULL, NULL);

    NOTICE 2005:  vvector: index docs options changed. Build options apply at the next refresh; query defaults apply now.

NULL keeps a value; the 13-argument form leaves `verify_every` and `cache_dir`
as they are, the 14-argument form `cache_dir`.
Query defaults (precision, freshness, ef_search, threads) apply at once on every
node (within 200 ms); `'default'` (text) or `0` (numbers) sets one back to the
built-in default. Build options apply at the next refresh. Values that belong
to later milestones are refused with a message that names the milestone.

| Option | Values | Default | Now |
|---|---|---|---|
| index_type | flat, hnsw | hnsw (as registered) | in use |
| m, ef_construction | 2 to 256, 1 to 100000 | 16, 200 | in use (HNSW) |
| quantization | none, sq8 | none | in use: sq8 adds one byte per element; searches rank by those bytes and rescore the best candidates (see [int8 quantisation](#int8-quantisation-sq8)); a change means a full build at the next refresh |
| refresh_mode | auto, incremental, full | auto | in use (see [refresh_index](#refresh_index)) |
| tombstone_ratio | above 0 to 1 | 0.2 | in use: `auto` builds in full once the tombstones exceed this share of the snapshot |
| rebuild_every | 0 (never) or more | never | in use: `auto` builds in full after this many incremental refreshes |
| verify_every | 0 (never), 1 (every refresh) or more | 1 | in use: how often a refresh verifies the journal rows up to the boundary (see [refresh_index](#refresh_index)) |
| memory_mode | ram, compact | ram | in use: `compact` (needs sq8) reads ahead only the bytes, ids and graph of a snapshot, not its floats; takes effect with the next snapshot a node maps (the next refresh) |
| precision_default | fast, balanced, best, exact | balanced | in use (HNSW); a flat index is always exact |
| freshness_default | snapshot, exact | snapshot | in use |
| ef_search_default | 0 to 100000 | 0 (preset) | in use (HNSW) |
| threads_default | 0 (one per core) to 64 | 0 | in use |
| cache_dir | an absolute path (letters, digits, `/ . _ -`), or `default` | the default directory (`/tmp/vvector`, or the session parameter) | in use: where every node keeps the cache files of this index; see [Operations](#operations) |

`cache_dir` takes effect at once: the active snapshot is loaded into the new
directory on every node (as `load_all` does), and the change counts only when
every node has it; if a node cannot load it, the option stays as it was and
the error (vload's) says why. The files in the old
directory stay; remove them by hand. From then on the refresh, `load_all` and
`status` use the option, also in a session that sets the `cache_dir` session
parameter. Queries without a `cache_dir` parameter find the index through a
small `OPTIONS` file that names the directory, which vvector writes into the
default directory (and into the one of the calling session) on every node.

### status and sizing

    CALL vvector.status('docs');

    NOTICE 2005:  vvector: index docs: hnsw index, 5 live vectors, 1 tombstones (tombstone_ratio 0.2), refresh_mode auto, 1 incremental refreshes since the last full build (rebuild_every never)
    NOTICE 2005:  vvector: index docs: journal digest verified at every refresh (verify_every 1); 0 refreshes since the last verification or full build
    NOTICE 2005:  vvector: index docs: last refresh: refreshed: snapshot 963, incremental from snapshot 962 (1 vectors appended, 1 tombstoned), 5 vectors of 3 dimensions, 1 tombstones, 0 MB, 0.347 seconds; journal verified in 0.021 seconds
    NOTICE 2005:  vvector: index docs: 3 journal rows in the delta, read in 7 ms
    NOTICE 2005:  vvector: index docs: journal replica auto: none: a single node reads the delta locally already
    NOTICE 2005:  vvector: index docs: sizing: index 0 MB, all indexes 1126 MB, build about 0 MB, smallest node 35155 MB of memory (28580 MB free or cache), 8 cores

`status` reports the live vectors and the tombstones, the refresh mode, what
the last refresh did, a refresh that is running now, the rows in the delta and how long they take to read,
the journal replica, open transactions that write to the table, versions in
the future (a sign that an application sets the version column itself), and a
sizing check: index size against node memory, the memory a refresh needs
against `FencedUDxMemoryLimitMB` and free memory, `threads_default` against
cores, and all indexes together against the page cache. Each problem is a
WARNING with the recommended fix. It never changes anything.

`sizing` estimates the memory of an index before you load the table; anyone
may call it:

    CALL vvector.sizing(10000000, 768, 'hnsw', 'none');

    NOTICE 2005:  vvector.sizing: 10000000 vectors of 768 dimensions (768 floats per row): vectors 29296.9 MB, ids 76.3 MB, graph 1349.8 MB, sq8 codes 0.0 MB
    NOTICE 2005:  vvector.sizing: snapshot and cache file 30722.9 MB per node; build memory about 30961.4 MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)
    NOTICE 2005:  vvector.sizing: queries read the cache file through the page cache: keep it in memory. Smallest node here: 34.3 GB of memory
    WARNING 2005:  vvector.sizing: the index needs more than half of the memory of the smallest node. Use quantization sq8 with memory_mode compact, or larger nodes.

### load_all, unregister_index

    CALL vvector.load_all('docs');          -- load the active snapshot and the defaults again on every node
    CALL vvector.unregister_index('docs');  -- remove schedule, views, snapshots and manifest row

`load_all` repairs node caches (a node that was down during a refresh, a
deleted cache directory). `unregister_index` leaves the cache files on the
nodes: remove `<cache_dir>/<index_name>` by hand.

### Scripts

The same from the shell:

    scripts/register.sh --index=docs --table=app.docs --id=id --vec=vec --op=del --ver=ts --metric=cosine
    scripts/refresh.sh --index=docs                        # refresh_index
    scripts/refresh.sh --index=docs --mode=full            # refresh_index('docs', 'full')
    scripts/refresh.sh --index=docs --schedule='0 * * * *' # schedule_refresh
    scripts/refresh.sh --index=docs --status               # status
    scripts/refresh.sh --index=docs --load_only            # load_all

`scripts/demo.sh` walks through everything on a table of its own (schema
VVDEMO, removed at the end unless `--keep`): it loads generated vectors
(100,000 x 128 by default) or SIFT1M (`--dir=<directory with sift_base.fvecs>`),
registers and builds an HNSW index, searches one query with vvector and with
the built-in full scan, measures the recall of the three precision levels,
adds and deletes a vector without a refresh and shows the difference between
`freshness='snapshot'` and `'exact'`, refreshes incrementally and shows every
node's cache. On the test VM with SIFT1M: one query 17 ms against 6.2 s for
the full scan, recall@10 0.93 / 0.99 / 0.999 (fast / balanced / best).

### The manifest

`SELECT * FROM vvector.manifest;` shows one row per index: the source
(`source_table`, `id_col`, `vec_col`, `op_col`, `ver_col`, `ver_margin`,
`metric`), the options of `set_index_options`, and the state of the active
snapshot (`active_snapshot`, `active_max_ver`, `delta_from` = the boundary of
the delta view, `vector_count` (live vectors), `tombstones`, `base_snapshot`
(the snapshot an incremental build started from, 0 after a full build),
`dims`, `graph_bytes` (the HNSW graph), `index_bytes` (the whole snapshot),
`built_at`, `build_seconds`, `format_version`, `active_options` (the build
options of the active snapshot), `incremental_count` (incremental refreshes
since the last full build), `boundary_rows` and `boundary_digest` (the number
and the digest of the journal rows up to the boundary, carried forward and
verified), `verify_every`, `refreshes_since_verify` (refreshes since the last
verification or full build), `refresh_note` (what the last refresh did and why),
`refresh_started_at` and `refresh_started_by` (set while a refresh runs)), and
the journal replica (`journal_replica` = auto, on or off;
`replica_projection`, the projection vvector made; `replica_note`, what the
last check did and why).

    SELECT index_name, source_table, metric, index_type, active_snapshot, vector_count, dims, graph_bytes, index_bytes, build_seconds
    FROM vvector.manifest WHERE index_name = 'docs';

     index_name | source_table | metric | index_type | active_snapshot | vector_count | dims | graph_bytes | index_bytes | build_seconds
    ------------+--------------+--------+------------+-----------------+--------------+------+-------------+-------------+---------------
     docs       | app.docs     | cosine | hnsw       |             959 |            5 |    3 |         896 |        1536 |         0.282

## Search

### vsearch

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs' [, name=value ...]) OVER()
    FROM <input>;

vsearch is a transform function: it reads all rows of its input, then
returns the k nearest neighbours of every query. The input always has these
seven columns; the role of a row is given by which columns are NULL:

| qid | qvec | id | vec | del | Role |
|---|---|---|---|---|---|
| set | set | NULL | NULL | NULL | a query |
| NULL | NULL | set | set | false | a journal row: add or change (from the delta view) |
| NULL | NULL | set | NULL | true | a journal row: delete (from the delta view) |
| NULL | NULL | set | NULL | NULL | an allow-list member: only allowed ids are returned (see [Filtered search](#filtered-search)) |
| NULL | NULL | NULL | NULL | NULL | the sentinel row of a view: carries `snapshot_id` only |

`ver` orders journal rows; `snapshot_id` tells vsearch which snapshot the view
belongs to, so a node with an older cache fails instead of answering from it.

Output: `(qid, id, score, rank)`. `score` is what the built-in function of the
metric returns: `VECTOR_L2` distance (smaller is closer), `COSINE_SIMILARITY`
and `DOT_PRODUCT` (larger is closer), the Manhattan distance for l1 (smaller
is closer). `rank` 1 is the closest; equal scores are ranked by id. Results do
not depend on the number of threads, the batch size, the CPU or the order of
the input rows.

Parameters:

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| index_name | (required) | | the index |
| k | 10 | 1 to 16384 | neighbours per query |
| query | | `'[x1, x2, ...]'` | one query vector as text, with qid 0; beside query rows or alone |
| freshness | snapshot | snapshot, exact | `exact` applies the journal rows of the input; `snapshot` ignores them |
| radius | off | a number | only neighbours within it, at most k: l2 and l1 `score <= radius`; cosine and dot `score >= radius` (on HNSW: see [Range search](#range-search)) |
| threads | 0 | 0 (one per core) to 64 | threads for one statement |
| precision | balanced | fast, balanced, best, exact | the speed and recall trade-off, a preset of ef_search (HNSW; fast: 2 x k, at least 32, and 32 with a radius; balanced: 100; best: 400) and, with sq8, of rescore and oversampling (fast: no rescoring; balanced: 2 x k candidates rescored; best: 4 x k). exact reads every float vector. A flat index without sq8 is always exact |
| ef_search | 0 (preset) | 0 to 100000 | HNSW: the length of the candidate list; overrides the preset of `precision`; below k it is raised to k (with a radius it grows from there, see [Range search](#range-search)). No effect on a flat index |
| exact | false | true, false | `true` reads every vector of an HNSW index (the same as `precision='exact'`) |
| rescore, oversampling | preset of precision | true or false; 1 to 100 | sq8 only: rank by the bytes, then compute the exact scores of the best k x oversampling candidates from the floats (`rescore=true`), or return the k best with approximate scores (`rescore=false`). No effect on an index without sq8 |
| filtered | false | true, false | `true` returns only allow-listed ids even when the input has no allow-list row (a filter that matched nothing returns nothing); without it, allow-list rows alone switch the filter on |
| cache_dir | the index option `cache_dir`, else `/tmp/vvector` | absolute path | where the node cache is; without it (and without the session parameter) a query follows the index option |

Every tuning value except `index_name`, `query`, `radius` and `filtered` can also be set
for a session, and `precision`, `freshness`, `ef_search` and `threads` per
index (`set_index_options`). The first that is set wins: function parameter,
then session parameter, then index default, then built-in default.

    ALTER SESSION SET UDPARAMETER FOR vvector freshness = 'exact';
    ALTER SESSION SET UDPARAMETER FOR vvector k = '20';

### One query, snapshot only (the fastest statement)

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

`app.docs_snap` is one row that carries the active snapshot id. The search
itself is the index's: on an HNSW index (the default) the graph with
`precision='balanced'`, on a flat index every vector. Pass the
query vector as the `query` parameter, not as an `ARRAY[...]` literal:
Vertica needs about 7 ms to parse a literal of 128 numbers, the parameter
costs nothing measurable. To build the text from a stored vector:
`SELECT TO_JSON(qvec) FROM ...` (it prints enough digits to read back the
same value).

### One query as a row

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', k=2) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT 7, ARRAY[0.0, 1.0, 0.1], NULL, NULL, NULL, NULL, NULL) q;

     qid | id |       score       | rank
    -----+----+-------------------+------
       7 |  3 | 0.995037198066711 |    1
       7 |  5 | 0.703597545623779 |    2

### Many queries from a table

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', k=2) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM app.questions) q
    ORDER BY qid, rank;

     qid | id |       score       | rank
    -----+----+-------------------+------
     100 |  2 | 0.996240615844727 |    1
     100 |  1 | 0.980580687522888 |    2
     200 |  4 | 0.990147531032562 |    1
     200 |  5 | 0.140028014779091 |    2

One statement with many queries is much faster than one statement per query:
on the test machine 1000 queries on 1M vectors take 25 to 35 ms with HNSW at
the default precision (29,000 to 40,000 queries per second; 12 to 22 ms with
`precision='fast'`) and about 1.1 s with a flat index.

### Precision, ef_search and exact search on an HNSW index

HNSW finds most, not always all, of the true nearest neighbours. `precision`
chooses how hard it looks; `ef_search` sets the same thing as a number.
`precision='exact'` or `exact=true` reads every vector, as a flat index does:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, precision='exact') OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, ef_search=200) OVER()
    FROM app.docs_snap;

(same result on this small index). What each level gives on 1M vectors is in
[Index types and tuning](#index-types-and-tuning).

### vknn: one vector per row, without OVER()

`vknn` searches one vector per input row and returns its k neighbours as rows
`(id, score, rank)`. It needs no `OVER()` and no view, and columns selected
beside it are repeated on every output row:

    SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='docs', k=2) FROM app.questions q ORDER BY 1, 4;

     qid | id |       score       | rank
    -----+----+-------------------+------
     100 |  2 | 0.996240615844727 |    1
     100 |  1 | 0.980580687522888 |    2
     200 |  4 | 0.990147531032562 |    1
     200 |  5 | 0.140028014779091 |    2

With the `query` parameter (used for every row whose vector is NULL):

    SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) FROM dual;

     id |       score       | rank
    ----+-------------------+------
      2 | 0.996240615844727 |    1
      1 | 0.980580687522888 |    2
      5 | 0.832050263881683 |    3

Parameters: those of vsearch without `freshness`. `vknn` searches the
snapshot only: it does not apply the journal, and it does not check that the
node's cache matches the active snapshot (it has no view input that carries
the snapshot id). Right after a refresh a node may answer from the previous
snapshot for up to 200 ms; a node that missed the refresh answers from its old
snapshot until `load_all` runs. A row with a NULL vector and no `query`
parameter gives no rows. Use vsearch over the views when the stale check or the
journal matter, and for many queries in one statement (vsearch spreads one
batch over all cores; vknn searches row by row).

### With the changes since the refresh

After the refresh above, vector 6 is added and vector 2 is deleted:

    INSERT INTO app.docs (id, vec) VALUES (6, ARRAY[1.0, 0.25, 0.0]);
    INSERT INTO app.docs (id, del) VALUES (2, TRUE);
    COMMIT;

The delta view now holds these two rows and the sentinel (the row that
carries the snapshot id). The five rows of the quick start are in the
snapshot: the refresh built every row up to its boundary, and with margin 0
that is every row written before it (see [Freshness](#freshness-explained)).

    SELECT id, del, ver IS NOT NULL AS has_ver, snapshot_id FROM app.docs_delta ORDER BY id;

     id | del | has_ver | snapshot_id
    ----+-----+---------+-------------
        |     | f       |         959
      2 | t   | t       |         959
      6 | f   | t       |         959

With `freshness='exact'`, id 6 is found and id 2 is gone; with the default
`snapshot` the result is the one of the last refresh:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, freshness='exact') OVER()
    FROM app.docs_delta;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

### Range search

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=10, radius=0.9, freshness='exact') OVER()
    FROM app.docs_delta;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2

At most k rows are returned. To get every vector within the radius, ask for
a large k; on an HNSW index that costs only as much as the vectors within the
radius, not k:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=16384, radius=0.9) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2

    SELECT vvector.vknn(ARRAY[0, 0.1, 1] USING PARAMETERS index_name='docs', k=16384, radius=0.95);

     id |       score       | rank
    ----+-------------------+------
      4 | 0.995037198066711 |    1
      7 | 0.970358312129974 |    2

How it works on HNSW: the graph search starts with the ef_search of the
precision level (fast: 32) and makes its candidate list four times longer,
up to k, as long as more than a quarter of the list is within the radius. A
narrow radius therefore stops after the first walk; a wide one grows to k.
On SIFT1M (1M vectors, k 16384) the search with a radius that holds about 10
vectors per query answers 2,674 queries per second against 485 when the list
is k long from the start, at recall 0.9999 (docs/design.md). Like every
graph search it is approximate: add `exact=true` for a complete answer.

### Filtered search

Only some vectors may be results: the documents of one customer, one
language, one category. Send their ids as allow-list rows (id set, vec and
del NULL) beside the query; vsearch returns only those ids. The examples
group the colours of the Quick start into families:

    CREATE TABLE app.families (id INT, family VARCHAR(20));
    INSERT INTO app.families VALUES (1, 'red'); INSERT INTO app.families VALUES (6, 'red');
    INSERT INTO app.families VALUES (5, 'warm'); INSERT INTO app.families VALUES (3, 'green');
    INSERT INTO app.families VALUES (4, 'blue'); INSERT INTO app.families VALUES (7, 'blue');
    INSERT INTO app.families VALUES (8, 'blue'); COMMIT;

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM app.families WHERE family = 'blue') x;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  8 |  0.32893693447113 |    1
       0 |  7 | 0.249459221959114 |    2
       0 |  4 |                 0 |    3

(Without the filter the same query returns 6, 1 and 5.) The filter applies to the journal rows too:
with `freshness='exact'` and the `_delta` view, a vector added since the
refresh is returned only when its id is allowed. Ids that are not in the
index are ignored.

A filter that matches nothing sends no allow-list row, and without any
allow-list row vsearch does not filter. When the filter may be empty, add
`filtered=true`: it returns nothing instead of the unfiltered answer.

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, filtered=true) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM app.families WHERE family = 'purple') x;

     qid | id | score | rank
    -----+----+-------+------
    (0 rows)

How it is searched: when fewer than max(10,000, sqrt(64 x ef_search x
vectors in the index)) allowed vectors remain (80,000 for 1M vectors at the
default precision), vsearch reads exactly those vectors: the answer is exact
and fast. With more, it walks the graph (or scans a flat index) and skips
the vectors outside the list. On SIFT1M a filter of 1% of the vectors
answers 75,700 queries per second in a batch (0.25 ms for a single query);
50% of the vectors 24,400 per second with recall 0.99 (docs/design.md).
The allow-list is part of the statement's input, so a list of millions of
ids costs the time Vertica needs to send them.

On a cluster, vsearch runs on the node that received the statement, and
allow-list rows from a segmented table must first be gathered from every
node. On the 4-node test cluster that costs about 24 ms whatever the list
size (one search, mixed, client median):

| allowed ids | from an UNSEGMENTED ALL NODES table | from a segmented table |
|---|---:|---:|
| no filter | 6.8 ms | 6.8 ms |
| 100 | 8.7 ms | 32.9 ms |
| 10,000 | 14.2 ms | 43.3 ms |
| 100,000 | 41.2 ms | 68.9 ms |

So keep the filter columns a search uses (id plus the category, tenant or
language) in a small table that is `UNSEGMENTED ALL NODES`, or give such a
table an unsegmented projection. Every node then has a full copy, and the
rows are read where vsearch runs. On one node it makes no difference.

For a filter that keeps most rows, searching without it and filtering the
results can be simpler; see [Recipes](#recipes).

### Join the results to your data

    SELECT r.rank, r.id, t.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
          FROM app.docs_snap) r
    JOIN app.titles t ON t.id = r.id
    ORDER BY r.rank;

     rank | id |  title   |       score
    ------+----+----------+-------------------
        1 |  2 | dark red | 0.996240615844727
        2 |  1 | red      | 0.980580687522888
        3 |  5 | yellow   | 0.832050263881683

### Recipes

The examples use the live rows of the journal as a view:

    CREATE VIEW app.docs_live AS
    SELECT id, vec FROM (SELECT id, vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, del DESC) AS rn
                         FROM app.docs) j
    WHERE rn = 1 AND NOT del;

**Filter after the search** (post-filter). Ask for more neighbours than you
need, join, filter and keep the first rows. Simple, and fine when the filter
removes few rows; with a selective filter use [Filtered search](#filtered-search).

    SELECT r.id, f.family, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=30) OVER()
          FROM app.docs_snap) r
    JOIN app.families f ON f.id = r.id
    WHERE f.family <> 'red'
    ORDER BY r.rank LIMIT 3;

     id | family |       score
    ----+--------+-------------------
      5 | warm   | 0.832050263881683
      8 | blue   |  0.32893693447113
      3 | green  | 0.303203642368317

**Recommendation by centroids.** "More like these, less like that": the
query is the average of the liked vectors minus the average of the disliked
ones ([Vector functions](#vector-functions)).

    SELECT r.rank, r.id, t.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='docs', k=3) OVER()
          FROM (SELECT * FROM app.docs_snap
                UNION ALL
                SELECT 1, vvector.vector_sub(liked.vector_avg, disliked.vector_avg), NULL, NULL, NULL, NULL, NULL
                FROM (SELECT vvector.vector_avg(vec) OVER() FROM app.docs_live WHERE id IN (1, 5)) liked
                CROSS JOIN (SELECT vvector.vector_avg(vec) OVER() FROM app.docs_live WHERE id = 4) disliked) x) r
    LEFT JOIN app.titles t ON t.id = r.id ORDER BY r.rank;

     rank | id | title  |       score
    ------+----+--------+-------------------
        1 |  6 | orange | 0.618346929550171
        2 |  1 | red    | 0.588348388671875
        3 |  5 | yellow | 0.554700195789337

Add the liked ids as allow-list rows the other way round (or filter them
out afterwards) to leave them out of the answer.

**The best match per group.** Search once with a k large enough to reach
every group, then keep the first row of each group:

    SELECT family, id, score FROM (
      SELECT f.family, r.id, r.score, ROW_NUMBER() OVER(PARTITION BY f.family ORDER BY r.rank) AS n
      FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                   USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=100) OVER()
            FROM app.docs_snap) r
      JOIN app.families f ON f.id = r.id) g
    WHERE n = 1 ORDER BY score DESC;

     family | id |       score
    --------+----+-------------------
     red    |  6 |  0.99886816740036
     warm   |  5 | 0.832050263881683
     blue   |  8 |  0.32893693447113
     green  |  3 | 0.303203642368317

**Near-duplicates.** Every vector as a query, k 2 (the vector itself and its
nearest other one), pairs above a threshold:

    SELECT r.qid AS id, r.id AS near_id, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', k=2) OVER()
          FROM (SELECT * FROM app.docs_snap
                UNION ALL SELECT id, vec, NULL, NULL, NULL, NULL, NULL FROM app.docs_live) x) r
    WHERE r.id <> r.qid AND r.score >= 0.97 AND r.qid < r.id
    ORDER BY r.score DESC;

     id | near_id |       score
    ----+---------+-------------------
      7 |       8 |  0.98894989490509
      1 |       6 | 0.970142483711243

For a large table run it in slices of query ids (`WHERE id % 10 = 0`, ...):
each statement searches all its query rows in one call.

## Index types and tuning

Two index types, chosen at registration (`register_index`, last argument) or
later with `set_index_options` (applies at the next refresh):

- **hnsw** (default): a graph over the vectors (HNSW, Malkov and Yashunin
  2018, built the way hnswlib builds it). A search walks the graph from an
  entry point towards the query and reads a few thousand vectors instead of
  all of them. It is approximate: it finds most of the true neighbours, and
  `precision` says how many. On 1M vectors of 128 dimensions one search
  takes 0.05 ms in the engine instead of 2.3 ms; a single-search statement is
  about 2.6 times faster (the rest is the cost of the statement), a batch of
  1000 queries 50 to 90 times.
- **flat**: every search reads every vector (with SIMD instructions and all
  cores). Always exact. Choose it for small indexes (up to about 100,000
  vectors a flat search takes well under a millisecond of engine time), when
  every answer must be exact, or when refreshes must be as fast as possible
  (no graph to build).

`precision` levels on SIFT1M (1M vectors of 128 dimensions, k = 10, 1000
queries, recall@10 = the share of the true 10 nearest neighbours found):

| precision | ef_search | recall@10 | 1000 queries in one statement | one search in the engine |
|---|---:|---:|---:|---:|
| fast | 2 x k, at least 32 | 0.892 | 14 ms | 0.05 ms |
| balanced (default) | 100 | 0.980 | 28 ms | 0.14 ms |
| best | 400 | 0.999 | not measured | 0.46 ms |
| exact | (every vector) | 0.999 (the rest are ties) | 1.1 s (measured on the flat index) | 9 ms (1 thread) |

A single statement costs about 1.5 ms more than the engine time (see
[Performance and results](#performance-and-results)), so for single searches
`balanced` costs little more than `fast` (0.1 ms), and it is the default. Use
`fast` for large batches where throughput counts more than the last 9% of
recall. Recall depends on the data: measure
it on your own vectors with `precision='exact'` as the reference. For 100
queries of a query table (here SIFT1M in schema VVBENCH):

    WITH q AS (SELECT * FROM VVBENCH.sift_hnsw_snap
               UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM VVBENCH.sift_query WHERE qid < 100),
         a AS (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10) OVER() FROM q),
         e AS (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10, precision='exact') OVER() FROM q)
    SELECT (COUNT(a.id) / COUNT(*))::NUMERIC(5,3) AS recall
    FROM e LEFT JOIN a ON a.qid = e.qid AND a.id = e.id;

     recall
    --------
      0.990

Add the settings you want to try to the parameters of `a`, for example
`precision='fast'` or `ef_search=150`.

Build options of an HNSW index (`set_index_options`, then `refresh_index`):

| Option | Default | Effect |
|---|---:|---|
| m | 16 | links per vector (2 x m on the lowest level). Higher: better recall for high-dimensional data, more memory ((2m + 1) x 4 bytes per vector and more), slower build |
| ef_construction | 200 | the candidate list while building. Higher: a better graph and recall, slower build |

`ef_search` sets the candidate list of a search directly and overrides
`precision`; per query, per session (`ALTER SESSION SET UDPARAMETER FOR
vvector ef_search = '150'`) or per index (`set_index_options`).

| If you want | Set |
|---|---|
| the lowest latency for single queries | the `query` parameter, `FROM <index>_snap`, deploy with `FENCED=mixed`; `vknn` is a little faster still but has no stale check |
| higher recall | `precision='best'`, or a larger `ef_search`; for all queries of an index: `set_index_options` |
| more throughput in large batches | `quantization='sq8'` (same recall), or `precision='fast'` (recall 0.89 instead of 0.98 on SIFT1M) |
| exact answers on an HNSW index | `precision='exact'` or `exact=true` for that query |
| results that include every committed change | `freshness='exact'` and `FROM <index>_delta` (per query, session or index) |
| many queries at once | one vsearch statement with all query rows (a table), not one statement per query |
| fewer cores for one statement | `threads=N` |
| the same default for every user of an index | `set_index_options` |
| faster searches, same exact scores | `quantization='sq8'` (see below) |

### int8 quantisation (sq8)

With `quantization='sq8'` the snapshot also stores every element of every
vector as one byte: a code from 0 to 255 in one range for the whole index,
trained on a sample of the vectors at a full build. A search first ranks the
candidates by these bytes (a quarter of the memory to read, integer
arithmetic), then computes the exact scores of the best k x oversampling
candidates from the float vectors (rescoring) and returns the k best of them.
The scores you get are exact; only the choice of candidates is approximate.

    CALL vvector.set_index_options('docs', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    CALL vvector.refresh_index('docs');

    NOTICE 2005:  vvector: index docs refreshed: snapshot 2016, full build (build options changed from hnsw cosine none m=16 ef_construction=200 to hnsw cosine sq8 m=16 ef_construction=200), 7 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.307 seconds; journal digest taken in 0.015 seconds

The same search as before; the scores are exact (rescoring), and with
`precision='fast'` they come from the codes alone:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

    -- precision='fast':
       0 |  6 | 0.997308850288391 |    1
       0 |  1 | 0.980392277240753 |    2
       0 |  5 | 0.830449938774109 |    3

`vinfo` shows the codes on every node (column `quantization`: `sq8`), and
`memory_mode` can then be set:

    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'compact', NULL, NULL, NULL, NULL);
    -- and back without codes needs memory_mode ram in the same call:
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, 'none', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    ERROR 2005:  vvector.set_index_options: memory_mode compact needs quantization sq8

Works with both index types, every metric, the journal (its rows are always
searched exactly) and incremental refresh (which keeps the range). The
`precision` levels on an index with sq8:

| precision | codes | rescoring |
|---|---|---|
| fast | graph walk or scan on the codes | none: the scores are approximate |
| balanced (default) | the same | 2 x k candidates rescored |
| best | the same, ef_search 400 | 4 x k candidates rescored |
| exact | not used: every float vector is read | |

`rescore` and `oversampling` override the preset per query, per session or
per index. SIFT1M (1M vectors of 128 dimensions, k = 10), engine alone:

| Search | recall@10 float | recall@10 sq8 | queries/s float | queries/s sq8 |
|---|---:|---:|---:|---:|
| HNSW, ef_search 100, rescoring 2 x k (balanced) | 0.983 | 0.983 | 48,900 / 35,500 | 72,600 / 60,300 |
| HNSW, ef_search 100, no rescoring | 0.983 | 0.970 | 48,900 / 35,500 | 73,500 / 61,500 |
| flat, rescoring 2 x k | 0.999 | 0.999 | 832 / 576 | 1,190 / 708 |

(queries/s: the test VM with 8 aarch64 cores / a 10-core x86_64 node, all
threads, 10,000 queries; `make test DATA_DIR=...`.) Through SQL, recall@10 of
1000 queries: fast 0.884, balanced 0.979, best 0.999, against 0.892, 0.980
and 0.999 without sq8.

Use it when the index is large and searches read much memory (flat indexes,
large batches, high dimensions). The codes do not replace the float vectors,
they come in addition: 1M x 128 grows from 631 MB to 757 MB as an HNSW
index. With `memory_mode='compact'` a node keeps the codes, ids and graph in
memory and reads float vectors only for rescoring (see the option table).

Memory:

| What | How much |
|---|---|
| snapshot and cache file per node | 256 bytes + (4 x row_stride + 8) bytes per vector; row_stride = dims rounded up to a multiple of 16 |
| HNSW graph (in the snapshot) | about (2m + 1) x 4 + 5 + (m + 1) x 4 / (m - 1) bytes per vector: 141 bytes with m = 16 |
| sq8 codes (in the snapshot) | row_stride + 4 bytes per vector: 132 bytes with 128 dimensions |
| a refresh (on one node) | full build: about the snapshot size + 4 bytes per vector; HNSW adds 5 + 2 x cores bytes per vector. Incremental: the new snapshot + 4 x row_stride bytes per changed row + 2 x cores bytes per vector (HNSW); the base is read from the cache file. Fenced it counts against `FencedUDxMemoryLimitMB` (-1 = no limit) |
| a query | 4 x row_stride bytes per query and per journal row, plus 1 bit per vector when the journal has rows; HNSW: 2 bytes per vector per search thread for the visited marks, kept by the process between queries |

1,000,000 vectors of 128 dimensions take 496 MB as a flat index and 631 MB as
an HNSW index; of 768 dimensions, 2.9 GB and 3.0 GB. Queries read the cache
file through the operating system's page cache: keep all indexes of a node in
memory (`status` warns when they do not fit). `CALL vvector.sizing(...)`
estimates an index before you load it.

## Freshness explained

- A query with `freshness='snapshot'` (the default) sees the vectors of the
  last refresh.
- A query with `freshness='exact'` over the `_delta` view sees every
  committed change: the view returns the journal rows with a version after
  the boundary of the last refresh, and vsearch applies them on top of the
  snapshot (the latest row per id wins, a delete removes the id).
- The boundary is taken before the refresh reads the table: the earlier of
  "now" and the start of the oldest open transaction that writes to the
  table, minus the margin (default 60 s). The refresh builds the rows up to
  the boundary; the delta view returns the rows after it; the two never
  overlap. So the rows of the last margin seconds before a refresh stay in
  the delta (`status` counts them) and a query with `freshness='snapshot'`
  sees them after the next refresh. A refresh that finds no live vector up
  to the boundary, such as the first refresh right after a load, stops with
  an error that says so: refresh again when the margin has passed.
- An open transaction that writes to the table holds the boundary back:
  everything written after its start stays in the delta until it ends. When
  it is older than ten times the margin (at least 60 s), the refresh note
  says so ("the boundary lags N minutes behind ...").
- Why the snapshot takes nothing after the boundary: the journal digest
  (see [refresh_index](#refresh_index)) covers the rows up to the boundary, so a physical DELETE or UPDATE
  of any row in the snapshot is found at the next refresh.
- Every query over the delta pays for its rows: about 1.7 ms per 1000 rows
  of 128 numbers on one node. `status` warns above 100,000 rows or when
  reading the delta is slow; refresh more often then. On a cluster the delta
  read also costs a transfer between nodes (see [Operations](#operations)).
- The journal works the same with both index types: journal rows are searched
  exactly and merged with the result of the graph or flat search; ids changed
  or deleted in the journal are never returned from the snapshot. Tested on
  flat and HNSW indexes: with `precision='exact'` the result equals the full
  scan of the live rows, before and after the next refresh.
- An incremental refresh folds the delta into the snapshot: it reads the same
  rows the delta view shows (the journal after the previous boundary), so the
  refresh costs the reading of the delta plus the writing of the snapshot, not
  a pass over the whole table.
- If a node answers with `snapshot cache stale on <node>: run vload`, its
  cache is older than the view: `CALL vvector.load_all('<index>')`.
- Grant SELECT on `<index>_snap` and `<index>_delta` to the users who search.
  The delta view shows the journal rows themselves.

## Vector functions

Vertica 26.2 has no arithmetic on arrays (`ARRAY[1, 2] + ARRAY[3, 4]` is an
error). vvector adds what is missing, in schema `vvector`, for the role
`vvector_search` (for everyone with `make deploy SEARCH=public`). They compute in FLOAT64; ARRAY[INT] and ARRAY[NUMERIC] arguments
are cast to ARRAY[FLOAT] (except Hamming and Jaccard, which take ARRAY[INT]).
A NULL argument gives NULL. Vectors of different lengths and NULL elements
are errors. They run fenced unless deployed with `FENCED=no` or `mixed`.

| Function | Returns | Meaning |
|---|---|---|
| `vector_add(a, b)` | ARRAY[FLOAT] | a[i] + b[i] |
| `vector_sub(a, b)` | ARRAY[FLOAT] | a[i] - b[i] |
| `vector_mul(a, b)` | ARRAY[FLOAT] | a[i] x b[i] (element by element) |
| `scalar_vector_mul(s, a)` | ARRAY[FLOAT] | s x a[i] |
| `vector_normalize(a)` | ARRAY[FLOAT] | a divided by its length (unit length); a zero vector stays zero |
| `vector_l1(a, b)` | FLOAT | sum of ABS(a[i] - b[i]) (Manhattan distance; the score of an l1 index) |
| `vector_l2sq(a, b)` | FLOAT | sum of (a[i] - b[i])^2 (`VECTOR_L2` squared, without the square root) |
| `vector_hamming(a, b)` | INT | the number of bits that differ, on ARRAY[INT]: elements 0 and 1, or 64 bits packed per element |
| `vector_jaccard(a, b)` | FLOAT | bits set in both / bits set in either (Tanimoto similarity), on ARRAY[INT] like Hamming; 1 when neither has a bit set |
| `vector_sum(a) OVER(...)` | ARRAY[FLOAT] | element sum of the vectors of a partition |
| `vector_avg(a) OVER(...)` | ARRAY[FLOAT] | element average of the vectors of a partition (a centroid) |

    SELECT vvector.vector_add(ARRAY[1, 2, 3], ARRAY[0.5, 0.5, 0.5]) AS sum,
           vvector.vector_sub(ARRAY[1, 2, 3], ARRAY[0.5, 0.5, 0.5]) AS difference,
           vvector.scalar_vector_mul(2, ARRAY[1, 2, 3]) AS twice;

          sum      |  difference   |     twice
    ---------------+---------------+---------------
     [1.5,2.5,3.5] | [0.5,1.5,2.5] | [2.0,4.0,6.0]

    SELECT vvector.vector_normalize(ARRAY[3, 4]) AS unit, vvector.vector_l1(ARRAY[0, 0], ARRAY[3, 4]) AS l1,
           vvector.vector_l2sq(ARRAY[0, 0], ARRAY[3, 4]) AS l2sq, VECTOR_L2(ARRAY[0, 0], ARRAY[3, 4]) AS l2;

       unit    | l1 | l2sq | l2
    -----------+----+------+----
     [0.6,0.8] |  7 |   25 |  5

    SELECT vvector.vector_hamming(ARRAY[1, 0, 1, 1], ARRAY[1, 1, 0, 1]) AS hamming,
           vvector.vector_jaccard(ARRAY[1, 0, 1, 1], ARRAY[1, 1, 0, 1]) AS jaccard,
           vvector.vector_hamming(ARRAY[255], ARRAY[15]) AS packed;

     hamming | jaccard | packed
    ---------+---------+--------
           2 |     0.5 |      4

`vector_sum` and `vector_avg` are transform functions, not aggregates
(Vertica 26.2 aggregates cannot take an array): write them with `OVER()` for
the whole input or `OVER(PARTITION BY ...)` per group, with only the
partition columns beside them, and put anything else in an outer query.
NULL vectors are skipped; a partition of only NULL vectors gives NULL.

    SELECT family, vector_avg FROM (SELECT family, vvector.vector_avg(vec) OVER(PARTITION BY family)
                                    FROM app.docs_live d JOIN app.families f USING (id)) c ORDER BY family;

     family |                          vector_avg
    --------+--------------------------------------------------------------
     blue   | [0.16666666666666667,0.10000000000000002,0.9333333333333332]
     green  | [0.1,0.9,0.0]
     red    | [1.0,0.125,0.0]
     warm   | [0.5,0.5,0.0]

Use Vertica's own functions where they exist: `VECTOR_L2`,
`COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_MAGNITUDE` (package VectorOps),
`APPLY_SUM(a)` for the sum of the elements of one vector (also `APPLY_AVG`,
`APPLY_MAX`, `APPLY_MIN`), `'[1.5, 2]'::ARRAY[FLOAT]` from text and
`TO_JSON(a)` to text.

## Operations

- **Multi-node**: every refresh loads the snapshot on every node (`vload`
  through `vvector.probe`); every node answers from its own cache file.
  Tested on one node and on a 3-node Eon cluster. A search runs on the node
  that receives the statement; the index is not split over nodes.
- **Journal replica on a cluster**: a statement over the `_delta` view
  reads the journal. With the journal segmented over the nodes, every node
  scans its part and sends the rows to the node that runs the search, even
  when no row qualifies: 15 to 17 ms per statement on the 3-node test
  cluster. vvector therefore keeps a replica of the journal: a projection
  `<schema>.<index>_journal_rep` of the id, vector, delete and version
  columns, sorted by the version column, `UNSEGMENTED ALL NODES`, with
  statistics on the version column. The planner then reads the delta on the
  node that runs the search.

  | Measured on the 3-node test cluster (100,000 x 128, mixed mode) | Without | With the replica |
  |---|---:|---:|
  | search over the `_delta` view, empty delta | 23.4 ms | 10.7 ms |
  | the same with 1000 journal rows | 31.3 ms | 14.6 ms |
  | storage per node | the node's share of the journal | a full copy of the journal (in Eon: in the depot of every node, one more copy in communal storage) |
  | bulk insert into the journal (20,000 rows) | 250 to 340 ms | 830 ms |
  | single-row insert and commit | about the same | about the same |
  | `refresh_index` | no difference | no difference |

  Because the cost grows with the journal, the default mode `auto` keeps the
  replica only where it is cheap: on a database with more than one node, when
  one copy of the journal takes at most 2048 MB and at most 10% of the
  smallest free disk space of a node. `register_index` and every
  `refresh_index` apply the rule: they make the replica (and refresh its
  statistics), or drop it when the journal has grown beyond the limit. On a
  single node there is nothing to gain and nothing is made. Indexes on the
  same journal columns share one replica; `unregister_index` drops it with
  the last of them. Change the mode per index:

      CALL vvector.set_journal_replica('docs', 'on');     -- always, whatever the size
      CALL vvector.set_journal_replica('docs', 'off');    -- never (drops it)
      CALL vvector.set_journal_replica('docs', 'auto');   -- the default rule

  `register_index`, `refresh_index`, `set_journal_replica` and `status` print
  what was done and why, for example
  `journal replica: created app.docs_journal_rep (journal 79 MB, one copy on each of 3 nodes)` or
  `journal replica: none: the journal takes 5120 MB, more than the limit of 2048 MB ...`.
  A new replica is not made while another transaction writes to the journal
  (Vertica's `CREATE PROJECTION` would wait for it to end): the message says
  "postponed" and the next refresh tries again, so a refresh never hangs
  behind a long load. Making the projection needs the right to create a
  projection on the journal table. Without it nothing fails: the message says "not created",
  gives the reason, and lists the three statements a DBA can run instead.
  vvector never drops a projection it did not make, and in `auto` mode it
  makes none when the table has an unsegmented projection already.
- **What each node has**:

      SELECT node_name, index_name, snapshot_id, vector_count, dims, metric, index_type, freshness_default, loaded
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='docs') OVER(PARTITION NODES) FROM vvector.probe) i;

         node_name    | index_name | snapshot_id | vector_count | dims | metric | index_type | freshness_default | loaded
      ----------------+------------+-------------+--------------+------+--------+------------+-------------------+--------
       v_vdb_node0001 | docs       |         959 |            5 |    3 | cosine | hnsw       | exact             | t

  Without `index_name` it lists every index in the cache directory.
  Other columns: max_ver, quantization, graph_bytes, tombstones,
  base_snapshot, precision_default, ef_search_default, threads_default,
  cache_file (or why the cache cannot be read), resident_mb (how much of the
  cache file is in the node's memory now: what a query reads without going to
  disk; vinfo itself reads nothing ahead).
- **Cache directory**: `/tmp/vvector` by default. Per index (recommended):
  the option `cache_dir` of [set_index_options](#set_index_options); the
  procedures and every query then use it without further settings. Per
  session: `ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/data/vvector';`
  (the refresh and every query must use the same one), or `cache_dir=` per
  call. The order: the function parameter, the session parameter, the index
  option, the default; inside the procedures the index option comes first.
  `/tmp` may be cleaned at reboot: that is safe (`load_all` restores it),
  but a node answers "run vload" until then. Use a directory on a data disk
  that belongs to the database's operating system user (vload creates it
  with mode 0700 if it is missing): a search reads only cache files and
  index directories of that user, and treats any other as "no cache" (any
  user who may search can set cache_dir, so a file someone else wrote is
  never read).
- **Memory**: a search reads the cache file through the page cache, and
  `vinfo` shows how much of it is in memory (`resident_mb`). Vertica's own
  scans go through the same cache, so a large scan can push index pages out;
  the next search then reads them from disk (on the test VM: one search of
  0.3 s for a 632 MB index, then normal speed; at 100M vectors a minute or
  more). Two ways to keep an index in memory, both with `cache_dir` on a RAM
  file system: `/dev/shm` (no setup; the memory is taken for good, and
  after a reboot the cache is empty until `load_all`), or a tmpfs mounted
  with huge pages, which is also faster (a root step on every node, for
  example `mount -t tmpfs -o size=8G,huge=always,mode=0755 tmpfs
  /data/vvhot` and `chown` to the database user): on the test VM batches of
  1000 searches took 22% less time and single searches 4 instead of 5 ms.
  Size it for the index files of every index placed there, plus one more
  snapshot during a refresh.
  A refresh whose build needs more than half of the smallest node's free
  memory (with the page cache) builds in a file in the index's cache
  directory instead, about 10% slower, and says so in its note; the kernel
  can then write the build out instead of running out of memory. That does
  not get around `FencedUDxMemoryLimitMB`: Vertica applies it as the address
  space limit of the fenced process, which counts a file mapping too.
- **Backup**: the snapshots are rows of `vvector.snapshot` and the options
  are rows of `vvector.manifest`: a backup of the database contains them.
- **Disk space**: a refresh keeps the active and the previous snapshot in the
  table and in every cache. The table is segmented: a snapshot is stored once
  across the cluster (with K-safety 1, twice), compressed; every node's cache
  holds the whole snapshot. Every refresh that changes the index writes a
  whole new snapshot, also an incremental one (631 MB for 1M vectors of 128
  dimensions with HNSW; Vertica compresses it to about half). The older one is
  deleted, and its rows keep using space until Vertica's Tuple Mover purges
  them; on the test VM that happened by itself within minutes (100 refreshes
  in 15 minutes never used more than 7 GB above the start). To free it at once:
  `SELECT PURGE_TABLE('vvector.snapshot');`.
- **Library version**: `SELECT vvector.vversion() OVER();`

       library_version | format_version |                           build_flags
      -----------------+----------------+------------------------------------------------------------------
       0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 aarch64 g++-11.5.0 kernels=neon

  A new library that reads another snapshot format refuses the old cache
  files with "format version 1, this library reads version 2: refresh the
  index": run `refresh_index` for every index after such an upgrade.
  Upgrading from a version that had `vbuild`, `vload`, `vconfig` and `vnode`
  in schema `vvector`: `make deploy` drops the library with its functions and
  creates them again (Vertica cannot drop a single function with an ARRAY
  argument); searches running at that moment fail. The first refresh of every
  index after that upgrade is a full build (the digest is new).
  Upgrading from a version before milestone M6, where `vvector.snapshot` was
  `UNSEGMENTED ALL NODES`: `make deploy` copies its rows once into a segmented
  table of the same name (a table cannot be resegmented in place). Deploy when
  no refresh runs; the copy takes about as long as writing the stored
  snapshots once (2.3 GB: a few seconds on the test VM). The copy does not
  touch the caches on the nodes, which stay valid.
  Upgrading from a version before milestone M6, where searching was open to
  PUBLIC: `make deploy` moves the search functions to the role
  `vvector_search` and closes the manifest to PUBLIC. Grant the role to the
  users who search, or deploy with `make deploy SEARCH=public` to keep the
  old behaviour.
- **Monitoring**: the refresh labels its statements `vvector_verify` (count
  and digest of the journal recomputed), `vvector_digest` (carried forward from
  the new rows, or taken for a full build), `vvector_build` and `vvector_load`:
  `SELECT request_label, request_duration_ms FROM v_monitor.query_requests WHERE request_label LIKE 'vvector%' ORDER BY start_timestamp DESC;`
  Put your own `/*+LABEL(name)*/` in searches to find them the same way.

### Low-level functions

`refresh_index` and `load_all` call these; you need them only to build or load
by hand.

| Function | Rights | What it does |
|---|---|---|
| `vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name, metric, index_type, max_ver, m, ef_construction, threads, quantization, base_snapshot, cache_dir, build_in) OVER()` | vvector_admin | turns (id, vector) rows into a snapshot; `build_in='file'` builds in an unlinked file in the index's cache directory instead of memory, so a build larger than the free memory can finish (slower); returns (byte_offset, chunk, vector_count, dims, max_ver, format_version), chunks of 8 MB; vector_count counts the live vectors. Rows with `del = true` are left out. No ORDER BY: it sorts by id itself. `metric` l2 (default), cosine, dot, l1; `index_type` flat (default of the function; the procedures pass the index's type) or hnsw with `m` (16), `ef_construction` (200) and `threads` (0 = one per core) for the graph build. With `base_snapshot` it builds incrementally from that snapshot in the cache of the node that runs it: the rows are the changes (one per id; `del = true` deletes), and it returns no rows when they change nothing |
| `vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name, snapshot_id, cache_dir) OVER(PARTITION NODES)` | vvector_admin | writes the snapshot to the cache of the node, verifies it, makes it active; returns (node_name, snapshot_id, bytes, status). Run again at any time |
| `vvector_admin.vconfig(k USING PARAMETERS index_name, options, cache_dir, index_cache_dir) OVER(PARTITION NODES) FROM vvector.probe` | vvector_admin | writes the index defaults (`options='precision=best,threads=4'`) to every node; with `index_cache_dir` (the index option; '' = none) into that directory, plus an OPTIONS file that names it in the default directory and in the session's; returns (node_name, status) |
| `vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe` | vvector_admin | one row per node: (node_name, k); used to send every chunk to every node exactly once |
| `vvector.vinfo([USING PARAMETERS index_name, cache_dir]) OVER(PARTITION NODES) FROM vvector.probe` | vvector_search | what every node has cached (see above) |
| `vvector.vversion() OVER()` | vvector_search | (library_version, format_version, build_flags) |

A snapshot built and loaded by hand (the procedures do the same, plus the
views, the manifest and the checks):

    INSERT INTO vvector.snapshot
    SELECT 'docs', 900, byte_offset, chunk FROM (
      SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='docs', metric='cosine') OVER()
      FROM app.docs) b;
    COMMIT;
    SELECT vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='docs', snapshot_id=900) OVER(PARTITION NODES)
    FROM (SELECT /*+SYNTACTIC_JOIN*/ s.byte_offset, s.chunk FROM vvector.probe p JOIN /*+DISTRIB(L,B)*/ vvector.snapshot s ON TRUE
          WHERE s.index_name = 'docs' AND s.snapshot_id = 900
            AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c;

The join sends every chunk to one probe row per node. The two hints make
Vertica broadcast the chunks (the table is segmented); without them a node
may get only the chunks it stores, and its vload refuses the incomplete
file. On a single node Vertica warns that the hint is not feasible and runs
the statement as written.

Take snapshot ids from `vvector.snapshot_seq` if the index is also refreshed
by the procedures: a node refuses a snapshot id lower than the one of the
views ("stale").

### Troubleshooting

Every message starts with the function or procedure that raised it. `vknn`
raises the messages of `vsearch` (those about queries, parameters and the
cache), starting with `vknn:`.

| Message (shortened) | Cause | Fix |
|---|---|---|
| `vsearch: no snapshot cache for index 'x' in DIR: run vload` | the index was never refreshed, or this node's cache is missing, or the session uses another cache_dir | `CALL vvector.load_all('x')`; check cache_dir |
| `vsearch: snapshot cache stale on NODE: run vload` | this node missed the last vload (down during a refresh) | `CALL vvector.load_all('x')` |
| `vsearch: snapshot cache of index 'x' is missing or damaged (...): run vload` | the cache file was deleted or changed | `CALL vvector.load_all('x')` |
| `... format version 1, this library reads version 2: refresh the index ...` | cache written by an older library | `CALL vvector.refresh_index('x')` |
| `vsearch: index 'x' has N dimensions, the query parameter has M` (or `query Q has M`, `the journal vector of id I has M`) | vectors of another length | use vectors of the index's length |
| `vsearch: no query: give the query parameter, or query rows (qid, qvec)` | the input has no query | add `query=` or query rows |
| `vsearch: query vector text: expected ',' or ']' at character N` (and similar) | malformed `query` text | write `'[1.5, 2, -3e-2]'` |
| `vsearch: query Q has no vector (qvec is NULL)`, `a query row has a vector (qvec) but no qid` | incomplete query row | set both qid and qvec |
| `vsearch: journal row with id I has no vector and is not a delete` | a journal row with a NULL vector and del false | fix the row, or set del |
| `vsearch: k must be 1 to 16384`, `precision must be ...`, `freshness must be ...`, `ef_search must be ...`, `oversampling must be 1 to 100`, `threads must be 0 (one per core) to 64`, `radius must be a finite number` | a parameter out of range | use a value from the parameter table |
| `vsearch: session parameter NAME = 'v' is not an integer` (or `index default ...`) | a bad session value or index default | `ALTER SESSION SET UDPARAMETER FOR vvector NAME = ...`, or `set_index_options` |
| `vsearch: the snapshot of index 'x' changed while the query ran: run it again` | a refresh changed the dimensions during the query | run the query again |
| `vbuild: id I appears twice` | a static index (no version column) with a repeated id (or the same id twice in the changes of a hand-made incremental build) | make the ids unique, or register a version column |
| `vbuild: index 'x': base snapshot S is not usable in the cache of NODE (...): refresh with mode full` | an incremental build found no valid active snapshot on the node that runs it (a refresh checks this first and builds in full; a hand-made build does not) | `CALL vvector.refresh_index('x', 'full')` |
| `vbuild: ... the base snapshot has metric M, not N`, `... is an HNSW index, the build is flat`, `... is a flat index, the build is hnsw`, `m is N, the base snapshot was built with m M`: `... a full build is needed` | a hand-made incremental build with other options than the base | build in full, or use the base's options |
| `vbuild: vector of id I has N elements, the index has M` | a change of another length than the index | one length per index |
| `vbuild: base_snapshot must be a snapshot id, or 0 for a full build` | a negative base_snapshot | a snapshot id from the manifest |
| `vbuild: the vector of id I is NULL (a delete needs del = true)`, `... has a NULL element`, `... element N is not a finite float32 value` | a missing vector, a NULL, NaN, Infinity or a value beyond +-3.4e38 | fix the row |
| `vbuild: vector of id I has N elements, the ones before have M` | vectors of different lengths | one length per index |
| `vbuild: vectors have N elements, at most 32768 are supported` | too many dimensions | reduce the dimensions |
| `vbuild: metric must be l2, cosine, dot or l1`, `index_type must be flat or hnsw`, `quantization ...`, `m must be 2 to 256`, `ef_construction must be 1 to 100000` | bad build parameter | see the parameter tables |
| `vbuild: ... too many vectors for an HNSW graph with m = N` | the upper levels of the graph would need more than 4,294,967,294 blocks | a larger `m`, or split the index |
| `vload: on NODE: bad snapshot graph: ... in FILE`, `vsearch: bad snapshot graph: ...` | the graph section of the snapshot or cache file is damaged (vload checks every link) | `CALL vvector.refresh_index('x')` |
| `vload: on NODE: bad snapshot sq8 section: ... in FILE`, `vsearch: bad snapshot sq8 section: ...` | the int8 codes of the snapshot or cache file are damaged (vload checks every row) | `CALL vvector.refresh_index('x', 'full')` |
| `vbuild: index 'x': the base snapshot has quantization sq8, not none: refresh with mode full` (or `none, not sq8`) | a hand-made incremental build with another quantization than the base (a refresh builds in full by itself when the option changed) | `CALL vvector.refresh_index('x', 'full')` |
| `... is not implemented yet (milestone Mn)`, `... not supported yet (milestone Mn)` | a feature of a later milestone | use what the message says is available |
| `vbuild: out of memory: cannot map N MB for the snapshot` | the node has too little memory for the build | `CALL vvector.sizing(...)`; a larger node or FencedUDxMemoryLimitMB |
| `vload: on NODE: pieces are missing or duplicated`, `bad snapshot: checksum mismatch`, `cannot write ...` | a damaged transfer or a full disk | check disk space of cache_dir; `load_all` |
| `vload`, `vinfo`, `vsearch`: `cache_dir '...' must be an absolute path`, `index name '...' is not valid` | bad cache_dir or index name | use `/path` and letters, digits, underscore |
| `vsearch: index 'x': OPTIONS file in the cache DIR: ...: run vvector.load_all` | the index defaults file of the node was changed by hand | `CALL vvector.load_all('x')` |
| `vvector.register_index: ...` (table, column, type, metric, margin, op_col needs ver_col, already registered) | a bad argument; the message names it | fix the argument |
| `vvector.refresh_index: index x: table T has no vectors, nothing to build` | the table has no live rows | insert rows first |
| `vvector.refresh_index: index x: no live vector up to the delta boundary B ...; N rows of T are newer` | every live row was written after the boundary (within the margin before the refresh, or after the start of an open writer), for example the first refresh right after a load | refresh again when the margin has passed; the rows are found meanwhile with `freshness='exact'`. On one node with `CLOCK_TIMESTAMP()` versions a margin of 0 is safe |
| `vvector.refresh_index: mode must be auto, incremental or full` | a bad second argument | `'auto'`, `'incremental'` or `'full'` |
| `vvector.refresh_index: index x is being refreshed since T UTC (by USER, session S). Two refreshes of one index cannot run at the same time ...` | another refresh of the index runs (by hand or by the schedule), or one was killed and left its mark (a superuser, or the same user, gets past a killed one at once) | wait until it ends; if none runs, `UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'x'; COMMIT;` (or wait for the hours the message names) |
| `Function vvector.vsearch(...) does not exist, or permission is denied for vvector.vsearch(...)` (or vknn, vinfo, a vector function) | the user lacks the role `vvector_search`, or has it but not enabled | `GRANT vvector_search TO someone; ALTER USER someone DEFAULT ROLE vvector_search;` (or `make deploy SEARCH=public`) |
| `Permission denied for schema vvector_admin` | a build or load function called without the role `vvector_admin` | `GRANT vvector_admin TO someone;` and enable it (default role or `SET ROLE`) |
| `Function vvector.refresh_index(unknown) does not exist, or permission is denied ...` (any procedure) | the caller lacks the role `vvector_admin`, or has it but not enabled in the session | grant it and enable it (default role or `SET ROLE vvector_admin`) |
| `Only a Super User can drop triggers` (from `schedule_refresh`) | `schedule_refresh` was called by a user who is not a superuser | a superuser runs `schedule_refresh` |
| `vvector.load_on_nodes: index x, snapshot S: loaded on N of M nodes` | a node could not load (disk, rights) | vinfo shows the cause per node; `load_all` |
| `vvector.push_options: index x: options written on N of M nodes` | a node could not write its cache directory | check the cache directory; `load_all` |
| `vvector.<procedure>: index x is not registered` | a wrong index name | `SELECT index_name FROM vvector.manifest` |
| `vvector.set_index_options: memory_mode compact needs quantization sq8` | `compact` on an index without sq8, or `quantization none` while `memory_mode` is `compact` | set sq8 first, or `memory_mode ram` in the same call |
| `vvector.set_index_options: ...` | a value out of range or of a later milestone | see the option table |
| `vvector.schedule_refresh: cron_expr may hold digits, spaces and * / , - only` | a bad cron expression | e.g. `'0 * * * *'` |
| `vvector.set_journal_replica: mode must be auto, on or off` | a bad mode | `'auto'`, `'on'` or `'off'` |
| `vector_add: the vectors have different lengths: N and M elements` (any vector function, `vector_avg`, `vector_sum`) | two vectors of different lengths | vectors of one length |
| `vector_l1: element I of the second vector is NULL` (any vector function; `vector_avg: element I of a vector is NULL`) | a NULL inside an array | replace the NULL, or filter the row |
| `Function vvector.vector_hamming(array[numeric], array[numeric]) does not exist` (or `array[float]`) | Hamming and Jaccard take ARRAY[INT] (bits) | cast to `ARRAY[INT]` |
| `ERROR 2521: Cannot specify anything other than user defined transforms and partitioning expressions in the SELECT list` | `vector_sum` and `vector_avg` are transform functions: only PARTITION BY columns may stand beside them | put other expressions in an outer query (see [Vector functions](#vector-functions)) |
| `journal replica: not created: ... A DBA can run: CREATE PROJECTION ...` (a NOTICE) | the caller may not create a projection on the journal table, or the name is taken | run the three statements as the table owner, or ignore it (searches work, only the cluster delta read stays slower) |

Warnings of `status` and `sizing` are explained in their text.

## Performance and results

Measured on the test VM: Vertica 26.2.0-1, one node, aarch64, 8 cores, 34 GB,
g++ 11.5 (milestone M4, 2026-09-24). Data: SIFT1M (1,000,000 vectors of 128
dimensions, 10,000 queries with ground truth, TEXMEX corpus), metric l2,
k = 10. HNSW with m = 16, ef_construction = 200. Three numbers, reported
separately.

**Engine alone** (`make bench DATA_DIR=...`, no Vertica, all 10,000 queries):

| Measurement | Flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| build of the snapshot (8 threads) | 0.09 s | 35 s | 35 s (the codes: 0.08 s) |
| snapshot size | 496 MB | 631 MB | 757 MB |
| 1 query, 1 thread | 9.2 ms | 0.14 ms (balanced) | 0.09 ms (balanced) |
| 1 query, 8 threads | 2.3 ms (the machine's full memory bandwidth) | (one thread per query) | (one thread per query) |
| queries per second, 8 threads, balanced | 950 (batch of 1000; exact) | 48,000 | 72,000 |
| recall@10 fast / balanced / best | 0.999 (exact; the rest are ties) | 0.903 / 0.983 / 0.999 | 0.895 (no rescoring) / 0.983 / 0.999 |

HNSW against hnswlib (v0.10.0-rc.2, the reference implementation), same
machine, same data and parameters (`make bench HNSWLIB_DIR=...`):

| ef_search | recall@10 vvector | recall@10 hnswlib | queries/s, 1 thread, vvector / with sq8 | hnswlib | queries/s, 8 threads, vvector / with sq8 | hnswlib |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.903 | 0.904 | 19,188 / 28,973 | 18,019 | 125,119 / 179,532 | 106,828 |
| 100 | 0.983 | 0.983 | 7,839 / 11,242 | 7,151 | 48,326 / 72,396 | 43,060 |
| 400 | 0.999 | 0.999 | 2,281 / 3,384 | 2,176 | 14,137 / 20,842 | 13,221 |

Equal recall; the float index 5 to 17% more queries per second than
hnswlib, with sq8 (2 x k rescored) 55 to 68% more; build 35 s against 40 s.

**One statement** (`scripts/latency.sh`, median at the client, 200 runs, `query`
parameter):

| Statement | Fenced | Mixed (or unfenced) |
|---|---:|---:|
| `SELECT 1` (the floor of any statement) | 0.8 ms | 0.8 ms |
| HNSW (precision balanced): vsearch `FROM sift_hnsw_snap` | 7.6 ms | 1.8 ms |
| HNSW: the same over `sift_hnsw_delta`, `freshness='exact'`, empty delta | 8.6 ms | 2.9 ms |
| HNSW: `vknn ... FROM dual` | 8.0 ms | 1.5 ms |
| HNSW with sq8 (balanced): vsearch `FROM sift_sq8_snap` / `vknn` | 7.5 ms / 7.9 ms | 1.8 ms / 1.5 ms |
| flat: vsearch `FROM sift_snap` | 11.0 ms | 5.5 ms |
| flat: the same over `sift_delta`, empty delta | 12.2 ms | 6.7 ms |
| SQL full scan (`ORDER BY VECTOR_L2(vec, q) LIMIT 10`) | 7797 ms | |

A single search spends about 0.1 ms in the engine; the rest is the
statement, so sq8 changes little for single searches and much for batches.

**Throughput, recall and refresh** (one statement with 1000 queries):

| Measurement | Fenced | Mixed |
|---|---:|---:|
| vsearch HNSW, precision fast | 24 ms (42,000 queries/s) | 14 ms (71,000 queries/s) |
| vsearch HNSW, precision balanced | 38 ms (26,000 queries/s) | 28 ms (36,000 queries/s) |
| vsearch HNSW with sq8, precision fast (codes only) | 20 ms (50,000 queries/s) | 11 ms (91,000 queries/s) |
| vsearch HNSW with sq8, precision balanced | 32 ms (31,000 queries/s) | 20 ms (50,000 queries/s) |
| vknn HNSW, precision balanced, 1000 rows | 158 ms (6,300 queries/s) | 149 ms (6,700 queries/s) |
| vsearch flat | 1108 ms (900 queries/s) | 1108 ms (900 queries/s) |
| recall@10 against the ground truth: HNSW fast / balanced / best / exact; flat | 0.892 / 0.980 / 0.999 / 0.999; 0.999 | |
| the same with sq8: fast / balanced / best | 0.884 / 0.979 / 0.999 | |
| `refresh_index`, 1M vectors, full build: HNSW / HNSW with sq8 / flat | 47 s / 49 s / 12 s | |
| `refresh_index` after 1000 adds and 500 deletes, 900,000 vectors, incremental: HNSW / flat | 6.8 s / 4.9 s | |
| `refresh_index` with nothing changed (verify_every 1 / 0) | 1.0 s / 0.6 s | |

On the 4-node Enterprise test cluster (x86_64 with AVX-512, 10 cores and 78 GB
per node, Vertica 26.2.0-3, g++ 8.5; SIFT1M as above, loaded on all 4 nodes) a
statement costs more, the engine is slower per core and hnswlib and vvector
are equal there: `SELECT 1` 3.4 to 3.8 ms; HNSW `_snap` 13.3 ms fenced and 6.4
ms mixed, with sq8 14.4 and 6.1 ms; `vknn` 13.5 and 6.0 ms. One statement with
1000 queries at precision balanced: 49 ms mixed, 34 ms with sq8. Engine, 10
threads, ef_search 100: 35,400 queries/s (hnswlib 34,700), with sq8 59,600.
Recall through SQL as on the VM. A full refresh of 1M x 128 HNSW takes 80 s
(87 s with sq8), an incremental one of 900,000 vectors after 1000 adds and 500
deletes 12.7 s: every node loads the whole snapshot (see Restrictions; 16.9 s
before the snapshot table was segmented, docs/design.md). A range
search (k 16384, the radius of the query's 10th neighbour) costs 15.6 ms
fenced and 7.1 ms mixed; filtered searches: see
[Filtered search](#filtered-search).

**10 million vectors** (the first 10M of BIGANN / SIFT1B, 128 dimensions, with
its ground truth for 10M; the 4-node cluster, 1000 queries; measured at
milestone M4, before the snapshot table was segmented, which makes refreshes
shorter):

| Measurement | Flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 4.8 GB | 6.2 GB | 7.4 GB |
| full build (`refresh_index`) | 123 s | 885 s | 939 s |
| incremental refresh, 1000 adds and 500 deletes | 99 s | 146 s | 194 s |
| recall@10 fast / balanced / best | 1.000 (exact) | 0.825 / 0.953 / 0.994 | 0.817 / 0.953 / 0.994 |
| one search, mixed (client ms) | 89 ms | 6.8 ms | 6.3 ms |
| 1000 queries in one statement, balanced, mixed | 17.6 s | 53 ms | 40 ms |

At 10M a larger `ef_search` keeps the recall of 1M: 200 gives 0.983 (1000
queries in 0.4 s). The incremental refresh is dominated by storing and loading
the whole snapshot on every node. An exact search over the delta view read the
10M-row journal on every node (33 ms instead of 7 ms): its rows were all loaded
the same day, so partitioning by date could not skip any, and at 2.7 GB it is
above the size the journal replica is made for; a journal partitioned by day
reads only the recent partitions.

On the 3-node Eon test cluster (x86_64, 2 cores and 15 GB per node, Vertica
26.2.0-2; HNSW index of 100,000 random vectors of 128 dimensions) a statement
costs more: `SELECT 1` 2.7 ms, vsearch `_snap` 12.8 ms fenced and 7.6 ms mixed,
`vknn` 12.1 and 5.9 ms (precision fast). An exact search over the empty delta
takes 10.7 ms mixed with the journal replica, which is made there by default,
and 23.4 ms without it (see [Operations](#operations)).

**Filtered and range search** (the VM, SIFT1M, one query, median at the
client): with an allow-list of 100 ids 8.6 ms fenced / 3.0 ms mixed, 10,000
ids 12.5 / 5.3 ms, 100,000 ids 29.8 / 13.9 ms (against 7.4 / 1.9 ms without a
filter); most of the added time is Vertica passing the allow-list rows.
Engine alone, 1000 queries: a filter of 1% of the vectors 75,700 queries/s
(exact), 50% 24,400 queries/s (recall 0.99). A range search with k 16384 and
a radius that holds about 10 vectors per query costs what a plain search
costs (1.9 ms mixed); in the engine it answers 2,674 queries/s against 485
with a candidate list of k. Details: docs/design.md.

The first search of a session costs more when vsearch is fenced, because the
session starts its own fenced process and maps the index: 13.3 ms instead of
7.6 ms (HNSW, 1M vectors); in mixed mode 2.4 ms instead of 2.1 ms. Keep
sessions open (a connection pool) for single searches.

Fenced mode adds about 6 ms per statement, more than the search itself; for
single searches `FENCED=mixed` gives the unfenced latency while the memory-
heavy build stays fenced. For batches the difference is smaller. Where each
millisecond goes is in [docs/design.md](docs/design.md).

To reproduce (loads SIFT1M into schema VVBENCH; about 40 minutes with the
hnswlib comparison):

    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
    make && make tools && make deploy
    scripts/benchmark.sh --data_dir=$PWD/sift [--hnswlib=<a clone of github.com/nmslib/hnswlib>]

**Scale tests.** `scripts/scale.sh` measures one data set end to end: loading,
full builds of a flat, an HNSW and an HNSW index with sq8 (`vvector.sizing`
first), vinfo on every node, recall@10 at every precision level, single
searches and batches fenced and mixed, and an incremental refresh. It is not
part of `make test`; `--help` lists the options. Examples:

    # the first 10M vectors of BIGANN (bigann_base.bvecs, bigann_query.bvecs and gnd/ of the TEXMEX corpus)
    scripts/scale.sh --dataset=bigann --dir=<dir> --rows=10000000 --gt=<dir>/gnd/idx_10M.ivecs
    # generated vectors (Gaussian clusters) of 768 numbers: for memory and speed; recall against exact search
    scripts/scale.sh --dataset=g768 --generate=768 --rows=1000000

Loading alone: `scripts/load_dataset.sh` takes the same `--dir`, `--rows`,
`--gt`, `--generate` and `--streams` options (parallel COPY streams, one per
two cores by default).

## Restrictions and not supported

Vertica and SDK:
- Vertica 26.2 has no VECTOR type and no vector index. vvector does not
  change `ORDER BY ... LIMIT`: searches are written with `vvector.vsearch`.
- The default array bound is 65000 bytes (8125 FLOAT elements); longer
  vectors need an explicit bound on the column (`ARRAY[FLOAT, 16384]`) and
  are untested.
- A transform function is not called on empty input: the views carry a
  sentinel row for that reason.
- Rights are granted per schema (Vertica cannot grant a single function with
  an ARRAY argument): the search functions in `vvector` need the role
  `vvector_search`, the build and load functions in `vvector_admin` the role
  `vvector_admin`.
- Searching cannot be limited per index: whoever has `vvector_search` can
  search every index by name (`vknn`, `vsearch ... FROM dual`), without a
  view. The views protect the journal rows, not the index.
- `schedule_refresh` needs a superuser (Vertica: only a superuser may create
  a trigger).
- A UNION ALL of `ARRAY[INT]` and `ARRAY[NUMERIC]` columns fails inside
  Vertica 26.2 (INTERNAL 5445); cast to `ARRAY[FLOAT]` first.
- Fenced mode adds about 6 ms per statement, and about 6 ms more to the first
  search of every session; unfenced functions run inside the Vertica process,
  where a fault stops the node.
- On a multi-node cluster a statement over the `_delta` view without the
  journal replica reads the journal on every node and sends the rows to the
  node that runs the search: 15 to 17 ms more per statement on the 3-node test
  cluster. The replica (automatic up to 2048 MB of journal, see
  [Operations](#operations)) costs a full copy of the journal on every node
  and makes bulk loads into the journal about 2.5 times slower.
  Snapshot-only statements (`_snap` view, `FROM dual`, `vknn`) do not read
  the journal at all.
- An `ARRAY[...]` literal of many numbers is slow to parse (about 7 ms for
  128): use the `query` parameter.

Data model:
- Ids are INT and unique per index; every vector of an index has the same
  number of elements, at most 32768; NULL elements, NaN, Infinity and values
  beyond +-3.4e38 are refused.
- The table is a journal: adds and deletes are INSERTs. A physical DELETE or
  UPDATE (or a dropped partition) becomes visible at the next refresh, which
  notices any physical change to rows up to its boundary (count and digest of
  those rows) and rebuilds in full because of it. With `verify_every` N it
  notices it within N refreshes; with 0 not at all: run
  `refresh_index(name, 'full')` after such a change. The verification reads
  every journal row up to the boundary.
- Without a version column the index is static and every id must appear once.
- The version column must be filled by the database (`DEFAULT
  CLOCK_TIMESTAMP()`); versions set by an application can make results wrong
  (`status` warns about versions in the future). Plain TIMESTAMP versions need
  the same time zone in every session; use TIMESTAMPTZ.
- Two rows of one id with the same version: a delete wins; between two adds
  either may win.
- The metric is fixed at registration; changing it means unregister and
  register again.

Index and search:
- HNSW is approximate: recall depends on `precision` / `ef_search`, `m`,
  `ef_construction` and the data (0.89 / 0.98 / 0.999 for fast / balanced /
  best on SIFT1M). `precision='exact'` or a flat index gives exact answers at
  the cost of reading every vector: about 2 to 3 ms per million vectors of 128
  dimensions on 8 cores.
- Range search on an HNSW index is approximate like every graph search: a
  vector inside the radius can be missed (use `exact=true` for a complete
  answer). At most k rows (k up to 16384) are returned.
- Filtered search: the allow-list is sent as rows with every statement (one
  row per allowed id); vvector cannot read a filter column itself. Above the
  exact-search limit the graph walk passes through the vectors outside the
  list; with a very selective filter just above that limit the walk is slow
  (it visits about ef_search x vectors / allowed nodes). `vknn` has no
  allow-list. On a multi-node cluster, allow-list rows from a segmented
  table are gathered from every node: about 24 ms more per statement on the
  4-node test cluster; read them from an `UNSEGMENTED ALL NODES` table
  ([Filtered search](#filtered-search)).
- `vector_sum` and `vector_avg` are transform functions, not aggregates
  (Vertica 26.2 aggregates cannot take an ARRAY argument): use them with
  `OVER()` or `OVER(PARTITION BY ...)` and only partition columns beside
  them. Hamming and Jaccard work on bits of ARRAY[INT] (0/1 elements or
  packed 64-bit words), not on sets of values.
- `vknn` searches the snapshot only: no journal and no stale check (a node
  that missed a refresh answers from its old snapshot until `load_all`).
- `register_index` creates an HNSW index by default since milestone M2 (it
  was flat before). Existing indexes keep their type.
- At most 4,294,967,295 vectors per index; one snapshot per index, cached
  whole on every node: an index larger than one node's memory works from disk,
  slowly.
- k is at most 16384; a radius search returns at most k rows.
- Journal rows since the last refresh are searched at every exact query; a
  large delta (100,000 rows and more) slows every such query: refresh more
  often.
- Scores are 32-bit floats: they agree with the FLOAT built-ins to about
  1e-6 relative; nearly equal scores can be ranked differently than the
  built-ins. Cosine with a zero vector gives score 0, as the built-in.
- int8 quantisation (sq8) is lossy. With rescoring (the default of balanced
  and best) the returned scores are exact, but a true neighbour whose codes
  rank it outside the k x oversampling candidates is missed; with
  `rescore=false` (and `precision='fast'`) the scores are approximate. One
  code range holds for the whole index; it is trained at a full build and
  kept by incremental refreshes, so vectors added later with values outside
  it get clipped codes until the next full build. The codes come in
  addition to the float vectors (about a quarter more bytes for float
  data); `memory_mode compact` only changes what is read ahead, and it takes
  effect at the next refresh.
- Not supported: product quantisation, IVF, DiskANN, sparse vectors, several
  vectors per id, GPU, hybrid text and vector search, sharding one index over
  nodes, big-endian hosts, Windows.

Operations:
- A new index default is seen by queries within 200 ms; a new snapshot at once.
- Two refreshes of one index at the same time are refused: the second one
  gets an error. A refresh whose session was killed leaves its mark in the
  manifest; the next refresh ignores it at once when it can see that the
  session is gone (a superuser, or the same user), else after 6 hours or 4
  times the last build time; the mark of a scheduled refresh counts by age
  only (or remove it by hand, see [refresh_index](#refresh_index)).
- Cache files stay on the nodes after `unregister_index`.
- A refresh builds on one node; its memory is the build memory above.
- A node that missed a refresh answers "snapshot cache stale ... run vload"
  until `load_all` runs.
- A new snapshot format needs a refresh of every index; the error says so.
- An incremental refresh reads only the changes, but writes and loads the
  whole snapshot again: its cost has a part that grows with the index size
  (about 6.9 s for 900,000 x 128 HNSW, 5.0 s flat, on the test VM, of which
  0.4 s is the verification of the journal) besides the part that grows
  with the changes. A full build of 1M x 128 takes 11 s as a flat index and
  46 s as an HNSW index. The part that grows with the index also grows with
  the number of nodes, because every node loads the whole snapshot: on a
  4-node cluster an incremental refresh of 900,000 x 128 HNSW takes 12.7 s
  (16.9 s before milestone M6, when every node also stored a copy of it).
- Tombstones (the old positions of changed and deleted vectors) stay in the
  snapshot until the next full build: they take memory, and an HNSW search
  passes through them. `refresh_mode auto` rebuilds in full at
  `tombstone_ratio`; with `refresh_mode incremental` schedule full refreshes
  yourself.
- A static index (no version column) is rebuilt in full at every refresh.
- The graph of a parallel build depends on the order in which threads insert:
  two builds of the same data give slightly different graphs (and recall);
  the results of a search on a given snapshot are always the same.

## Files

| File | What it does |
|---|---|
| `Makefile` | `make`, `make test`, `make bench`, `make tools`, `make deploy [FENCED=yes\|no\|mixed]`, `make undeploy` |
| `src/engine/` | pure C++17, no Vertica includes: snapshot format, incremental build (`delta.cpp`), node cache, distance kernels, flat search, HNSW (`hnsw.cpp`), int8 codes (`sq8.cpp`), the search with rescoring and filters (`search.cpp`), the arithmetic of the vector functions (`vecmath.cpp`), threads, query text |
| `src/udx/` | the Vertica adapters: one small file per SQL function (`vsearch.cpp`, `vknn.cpp`, `vbuild.cpp`, ...), and one per family of vector functions (`vector_functions.cpp`, `vector_aggregates.cpp`) |
| `sql/` | `install.sql`, `procedures.sql`, `uninstall.sql` |
| `scripts/` | `deploy.sh`, `register.sh`, `refresh.sh`, `load_dataset.sh`, `latency.sh`, `benchmark.sh`, `scale.sh` |
| `tools/fvecs.cpp` | converts `.fvecs`, `.ivecs`, `.bvecs` files (SIFT1M, BIGANN) to text for COPY, and generates random clustered vectors |
| `tests/engine/` | unit tests (`make test`, among them `test_hnsw.cpp`, `test_delta.cpp`, `test_sq8.cpp`, `test_filter.cpp` and `test_vecmath.cpp`) and the engine benchmarks (`make bench`: `bench_flat.cpp`, `bench_hnsw.cpp` (float and sq8), `bench_filter.cpp` (filtered and range search), and `bench_hnswlib.cpp` with `HNSWLIB_DIR=`) |
| `tests/sql/` | integration tests: `test_snapshot.sh`, `test_freshness.sh`, `test_search.sh`, `test_hnsw.sh`, `test_incremental.sh` (`--sift=SCHEMA` adds the 100-refresh test on SIFT1M), `test_rights.sh` (a user with only the documented rights; needs a superuser connection), `test_sq8.sh` (int8 quantisation; `--sift=SCHEMA` adds recall on SIFT1M), `test_filter.sh` (filtered and range search), `test_vector_functions.sh`; `run_all.sh` runs them in every mode |
| `docs/` | `design.md` (decisions, measurements), `format.md` (snapshot format), `build-x86.md` (step by step on x86_64 and Eon), `VERTICA_NOTES.md` (verified Vertica behaviour) |

## License

MIT. Author: Mo (github.com/mogomo).

The snapshot, cache, load and freshness code comes from
[vertica-graph-udx](https://github.com/mogomo/vertica-graph-udx).

## Disclaimer

This repository is a demo. It is not a product of, and is not endorsed or
supported by, Rocket Software, Vertica or any other company. The software is
provided "as is", without warranty of any kind, as the MIT license says.
Please try it on your own systems and data before you rely on it, and take
special care with unfenced mode, where the code runs inside the Vertica process.

Vertica, Rocket Software and all other product and company names are
trademarks or registered trademarks of their respective owners.
