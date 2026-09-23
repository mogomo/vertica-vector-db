# vertica-vector-db (vvector)

Nearest-neighbour vector search inside Vertica. vvector is a C++ UDx library
that keeps an index of the vectors stored in a Vertica table and answers "the
k vectors closest to this one" from SQL. The index lives in Vertica, is loaded
on every node, and every query can see the rows written since the last
refresh.

**Status: milestone M2 (HNSW).** Two index types: `hnsw` (a graph index,
approximate, the default) and `flat` (exact). k-nearest-neighbour search works
for the metrics l2, cosine, dot and l1, for one query or thousands in one
statement, with or without the rows written since the last refresh. Exact
results are tested to equal a full scan with Vertica's built-in functions;
HNSW recall is measured on SIFT1M. Tested on Vertica 26.2 on one node
(aarch64, Rocky Linux 9, g++ 11.5) and on a 3-node Eon cluster (x86_64, Red
Hat Enterprise Linux 8, g++ 8.5), fenced, unfenced and mixed. Not yet
available: incremental refresh (M3; every refresh rebuilds the index), int8
quantisation (M4), filtered search, range search on the graph and vector
functions (M5). Treat this as a preview: try it on your own systems before you
rely on it.

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
machine. vvector answers the same query in 1.9 ms with an HNSW index (7.8 ms
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
    vsql -c "CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', NULL);"
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
[Index types and tuning](#index-types-and-tuning)). The same query with the
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

- Vertica 26.x (tested: 26.2.0-1 single node, 26.2.0-2 Eon with 3 nodes) with
  the C++ SDK in `/opt/vertica/sdk` (another place: `make SDK_HOME=...`).
- g++ with C++17 (tested: 11.5 on aarch64, 8.5 on x86_64) and GNU make, on a
  Vertica node: `CREATE LIBRARY` reads the .so from the initiator node's file
  system and copies it to the other nodes. No CPU flags are needed: on x86_64
  the distance code is compiled for SSE2, AVX2 and AVX-512 and the best one is
  picked at load time, with bit-identical results on every level.
- Step by step on x86_64 (Red Hat 8, Eon cluster), with the expected output:
  [docs/build-x86.md](docs/build-x86.md).
- A database user that may create a schema, a library, functions and a role
  (dbadmin, or a user with those rights).

### Build, test, deploy

    make                      # build/libvvector.so
    make test                 # engine unit tests, no database needed
    make deploy               # install into the database, fenced (the default)
    make deploy FENCED=no     # every function inside the Vertica process
    make deploy FENCED=mixed  # vbuild, vload, vconfig, vnode fenced; vsearch, vknn, vinfo, vversion not fenced
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

Everything is in schema `vvector`:

| Object | What it is |
|---|---|
| `vvector.snapshot` | table (UNSEGMENTED ALL NODES): the index snapshots in chunks of 8 MB |
| `vvector.manifest` | table: one row per index with its source, options and state |
| `vvector.probe` | table (8192 rows, segmented): makes node-wise functions run once on every node |
| `vvector.snapshot_seq` | sequence of snapshot ids (never reused) |
| role `vvector_admin` | may build, load and manage indexes |
| functions | `vsearch`, `vknn`, `vinfo`, `vversion` (search and information), `vbuild`, `vload`, `vconfig`, `vnode` (build and load) |
| procedures | `register_index`, `set_index_options`, `refresh_index`, `load_all`, `status`, `sizing`, `schedule_refresh`, `unregister_index` |

Rights: `vsearch`, `vknn`, `vinfo`, `vversion`, `vbuild`, `vnode` and the procedure
`sizing` are open to everyone (PUBLIC); `vload`, `vconfig` and all other
procedures need `vvector_admin` (`GRANT vvector_admin TO someone;`). vbuild is
open because Vertica 26.2 cannot grant a single function that has an ARRAY
argument; it writes nothing, and storing its output needs rights on
`vvector.snapshot`. The views of an index are not granted to anyone: grant
SELECT on them to the users who may search (the `_delta` view shows the rows
of the journal).

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
  your own tables; join them to the results by id.
- **vec**: `ARRAY[FLOAT]` (recommended), `ARRAY[INT]` or `ARRAY[NUMERIC]`.
  Every vector of one index has the same number of elements. vvector stores
  and computes in 32-bit floats.
- **del** (optional): BOOLEAN, true = deleted; or INT, +1 added and -1 deleted.
- **ts** (optional): the version. For every id the row with the latest
  version wins; with equal versions a delete wins. Use `TIMESTAMPTZ NOT NULL
  DEFAULT CLOCK_TIMESTAMP()`: the database sets the time of the write, which
  is what makes the results exact (see [Freshness](#freshness-explained)).
  TIMESTAMP and increasing INT versions are accepted too.

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

A physical `UPDATE` or `DELETE` on the table is allowed, but queries see it
only after the next refresh. A table without a version column is a **static
index**: queries see the snapshot only, every id must appear once, and
changes show up at the next refresh.

## Register, refresh, schedule

All procedures need the role `vvector_admin`.

### register_index

    CALL vvector.register_index(index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin [, index_type]);
    CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', NULL);

| Argument | Meaning |
|---|---|
| index_name | 1 to 64 letters, digits or underscores |
| source_table | `schema.table` |
| id_col, vec_col | the id (INT) and vector column |
| op_col | delete flag (BOOLEAN or INT), or NULL when rows are only added; needs ver_col |
| ver_col | version (TIMESTAMPTZ, TIMESTAMP or INT), or NULL for a static index |
| metric | `l2` (VECTOR_L2), `cosine` (COSINE_SIMILARITY), `dot` (DOT_PRODUCT) or `l1` (Manhattan distance); fixed for the life of the index |
| margin | overlap of the delta: seconds for a timestamp version (NULL = 60); units of the column for an INT version (required) |
| index_type | `hnsw` (default: a graph index, fast and approximate) or `flat` (exact, reads every vector); see [Index types and tuning](#index-types-and-tuning) |

It creates the views `<schema>.<index>_snap` and, with a version column,
`<schema>.<index>_delta` in the schema of the source table:

    NOTICE 2005:  vvector: index docs registered. Next: CALL vvector.refresh_index('docs'). Queries read docs_snap (snapshot only) or docs_delta (with the changes since the refresh) in schema app; grant SELECT on them to the users who may search the index.

### refresh_index

    CALL vvector.refresh_index('docs');

    NOTICE 2005:  vvector: index docs refreshed: snapshot 240, 5 vectors of 3 dimensions, 0 MB, 0.237 seconds

It takes the delta boundary, builds a new snapshot from the latest row of
every id (deletes left out), stores it in `vvector.snapshot`, loads it on
every node, writes the index defaults to every node, updates the manifest and
the views, and deletes snapshots older than the previous one. Queries keep
working during a refresh. Every refresh is a full rebuild until milestone M3.
On the test machine 1,000,000 vectors of 128 dimensions take 11 s as a flat
index and 46 s as an HNSW index (the graph build uses every core of the node
that runs the refresh).

### schedule_refresh

    CALL vvector.schedule_refresh('docs', '0 * * * *');      -- every hour, at minute 0
    CALL vvector.schedule_refresh('docs', '*/15 * * * *');   -- every 15 minutes

Creates `vvector.docs_refresh_schedule` and `vvector.docs_refresh_trigger`
(Vertica's CRON schedule; the trigger runs as the definer). Calling it again
replaces the schedule.

### set_index_options

    CALL vvector.set_index_options(index_name, index_type, m, ef_construction, quantization, refresh_mode,
                                   tombstone_ratio, rebuild_every, memory_mode, precision_default,
                                   freshness_default, ef_search_default, threads_default);

    -- queries of index docs apply the journal by default:
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'exact', NULL, NULL);

    -- a denser graph from the next refresh on, and precision balanced for every query from now on:
    CALL vvector.set_index_options('docs', NULL, 32, 400, NULL, NULL, NULL, NULL, NULL, 'balanced', NULL, NULL, NULL);

    NOTICE 2005:  vvector: index docs options changed. Build options apply at the next refresh; query defaults apply now.

NULL keeps a value. Query defaults (the last four) apply at once on every
node (within 200 ms); `'default'` (text) or `0` (numbers) sets one back to the
built-in default. Build options apply at the next refresh. Values that belong
to later milestones are refused with a message that names the milestone.

| Option | Values | Default | Now |
|---|---|---|---|
| index_type | flat, hnsw | hnsw (as registered) | in use |
| m, ef_construction | 2 to 256, 1 to 100000 | 16, 200 | in use (HNSW) |
| quantization | none, sq8 | none | none only (sq8: M4) |
| refresh_mode | auto, incremental, full | auto | every refresh is full (incremental: M3) |
| tombstone_ratio, rebuild_every | above 0 to 1; 0 = never | 0.2; never | stored for M3 |
| memory_mode | ram, compact | ram | ram only (compact: M4) |
| precision_default | fast, balanced, best, exact | fast | in use (HNSW); a flat index is always exact |
| freshness_default | snapshot, exact | snapshot | in use |
| ef_search_default | 0 to 100000 | 0 (preset) | in use (HNSW) |
| threads_default | 0 (one per core) to 64 | 0 | in use |

### status and sizing

    CALL vvector.status('docs');

    NOTICE 2005:  vvector: index docs: hnsw index, refresh_mode auto, 0 tombstones
    NOTICE 2005:  vvector: index docs: 7 journal rows in the delta, read in 8 ms
    NOTICE 2005:  vvector: index docs: sizing: index 0 MB, all indexes 1126 MB, build about 0 MB, smallest node 35155 MB of memory (30443 MB free or cache), 8 cores

`status` reports the rows in the delta and how long they take to read, open
transactions that write to the table, versions in the future (a sign that an
application sets the version column itself), and a sizing check: index size
against node memory, the memory a refresh needs against
`FencedUDxMemoryLimitMB` and free memory, `threads_default` against cores, and
all indexes together against the page cache. Each problem is a WARNING with
the recommended fix. It never changes anything. `refresh_index` runs it first.

`sizing` estimates the memory of an index before you load the table; anyone
may call it:

    CALL vvector.sizing(10000000, 768, 'hnsw', 'none');

    NOTICE 2005:  vvector.sizing: 10000000 vectors of 768 dimensions (768 floats per row): vectors 29296.9 MB, ids 76.3 MB, graph 1349.8 MB, sq8 codes 0.0 MB
    NOTICE 2005:  vvector.sizing: snapshot and cache file 30722.9 MB per node; build memory about 30961.4 MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)
    NOTICE 2005:  vvector.sizing: queries read the cache file through the page cache: keep it in memory. Smallest node here: 34.3 GB of memory
    WARNING 2005:  vvector.sizing: the index needs more than half of the memory of the smallest node. Use quantization sq8 with memory_mode compact (milestone M4) or larger nodes.

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
    scripts/refresh.sh --index=docs --schedule='0 * * * *' # schedule_refresh
    scripts/refresh.sh --index=docs --status               # status
    scripts/refresh.sh --index=docs --load_only            # load_all

### The manifest

`SELECT * FROM vvector.manifest;` shows one row per index: the source
(`source_table`, `id_col`, `vec_col`, `op_col`, `ver_col`, `ver_margin`,
`metric`), the options of `set_index_options`, and the state of the active
snapshot (`active_snapshot`, `active_max_ver`, `delta_from` = the boundary of
the delta view, `vector_count`, `dims`, `graph_bytes` (the HNSW graph), `index_bytes` (the whole snapshot), `built_at`,
`build_seconds`, `format_version`).

    SELECT index_name, source_table, metric, index_type, active_snapshot, vector_count, dims, graph_bytes, index_bytes, build_seconds
    FROM vvector.manifest WHERE index_name = 'docs';

     index_name | source_table | metric | index_type | active_snapshot | vector_count | dims | graph_bytes | index_bytes | build_seconds
    ------------+--------------+--------+------------+-----------------+--------------+------+-------------+-------------+---------------
     docs       | app.docs     | cosine | hnsw       |             240 |            5 |    3 |         896 |        1536 |         0.237

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
| NULL | NULL | set | NULL | NULL | allow-list member for filtered search (milestone M5; refused now) |
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
| precision | fast | fast, balanced, best, exact | HNSW: the speed and recall trade-off, a preset of ef_search (fast: 2 x k, at least 32; balanced: 100; best: 400; exact: read every vector). A flat index is always exact |
| ef_search | 0 (preset) | 0 to 100000 | HNSW: the length of the candidate list; overrides the preset of `precision`; below k it is raised to k. No effect on a flat index |
| exact | false | true, false | `true` reads every vector of an HNSW index (the same as `precision='exact'`) |
| rescore, oversampling | true, 1 | true or false; 1 to 100 | int8 quantisation (M4); no effect now |
| cache_dir | `/tmp/vvector` | absolute path | where the node cache is |

Every tuning value except `index_name`, `query` and `radius` can also be set
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
`precision='fast'`, on a flat index every vector. Pass the
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
on the test machine 1000 queries on 1M vectors take 12 to 22 ms with HNSW
(45,000 to 83,000 queries per second) and about 1.1 s with a flat index.

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

The delta view now holds these rows, the sentinel, and here also the five
rows of the quick start: they were written less than a minute before the
refresh, inside the margin (see [Freshness](#freshness-explained)). They are in
the snapshot too; applying them again changes nothing.

    SELECT id, del, ver IS NOT NULL AS has_ver, snapshot_id FROM app.docs_delta ORDER BY id;

     id | del | has_ver | snapshot_id
    ----+-----+---------+-------------
        |     | f       |         240
      1 | f   | t       |         240
      2 | f   | t       |         240
      2 | t   | t       |         240
      3 | f   | t       |         240
      4 | f   | t       |         240
      5 | f   | t       |         240
      6 | f   | t       |         240

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

At most k rows are returned; raise k for a wide radius. On an HNSW index the
radius filters the candidates the graph search finds (ef_search of them), so
it can miss vectors inside the radius; add `exact=true` for a complete answer
(true range search on the graph comes with milestone M5).

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

A filter on your own columns can be applied after the search: ask for more
neighbours than you need (for example `k=100`), join, filter, and keep the
first rows. Filtered search inside the index comes with milestone M5.

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
| fast (default) | 2 x k, at least 32 | 0.893 | 12 ms | 0.05 ms |
| balanced | 100 | 0.980 | 25 ms | 0.14 ms |
| best | 400 | 0.999 | not measured | 0.44 ms |
| exact | (every vector) | 0.999 (the rest are ties) | 1.1 s (measured on the flat index) | 9 ms (1 thread) |

A single statement costs about 1.5 ms more than the engine time (see
[Performance and results](#performance-and-results)), so for single searches
`balanced` costs little more than `fast`. Recall depends on the data: measure
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
      0.930

Add the settings you want to try to the parameters of `a`, for example
`precision='balanced'` or `ef_search=150`.

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
| higher recall | `precision='balanced'` or `'best'`, or a larger `ef_search`; for all queries of an index: `set_index_options` |
| exact answers on an HNSW index | `precision='exact'` or `exact=true` for that query |
| results that include every committed change | `freshness='exact'` and `FROM <index>_delta` (per query, session or index) |
| many queries at once | one vsearch statement with all query rows (a table), not one statement per query |
| fewer cores for one statement | `threads=N` |
| the same default for every user of an index | `set_index_options` |

Memory:

| What | How much |
|---|---|
| snapshot and cache file per node | 256 bytes + (4 x row_stride + 8) bytes per vector; row_stride = dims rounded up to a multiple of 16 |
| HNSW graph (in the snapshot) | about (2m + 1) x 4 + 5 + (m + 1) x 4 / (m - 1) bytes per vector: 141 bytes with m = 16 |
| a refresh (on one node) | about the snapshot size + 4 bytes per vector; HNSW adds 5 + 2 x cores bytes per vector; fenced it counts against `FencedUDxMemoryLimitMB` (-1 = no limit) |
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
  table, minus the margin (default 60 s). Rows just before the boundary are
  in the snapshot and in the delta; applying them twice gives the same
  result. That is why `status` may count rows in the delta right after a
  refresh.
- Every query over the delta pays for its rows: about 1.7 ms per 1000 rows
  of 128 numbers on one node. `status` warns above 100,000 rows or when
  reading the delta is slow; refresh more often then. On a cluster the delta
  read also costs a transfer between nodes (see [Operations](#operations)).
- The journal works the same with both index types: journal rows are searched
  exactly and merged with the result of the graph or flat search; ids changed
  or deleted in the journal are never returned from the snapshot.
- If a node answers with `snapshot cache stale on <node>: run vload`, its
  cache is older than the view: `CALL vvector.load_all('<index>')`.
- Grant SELECT on `<index>_snap` and `<index>_delta` to the users who search.
  The delta view shows the journal rows themselves.

## Vector functions

Milestone M5 adds vector functions that Vertica does not have (sum,
difference, normalisation, Hamming and Jaccard distance, average). Until then
use Vertica's own `COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and
`VECTOR_MAGNITUDE`, and for the l1 distance an expression such as
`ABS(a[0] - b[0]) + ABS(a[1] - b[1]) + ...`.

## Operations

- **Multi-node**: every refresh loads the snapshot on every node (`vload`
  through `vvector.probe`); every node answers from its own cache file.
  Tested on one node and on a 3-node Eon cluster. A search runs on the node
  that receives the statement; the index is not split over nodes.
- **Journal on a cluster**: a statement over the `_delta` view reads the
  journal on every node and sends the rows to the node that runs the search,
  even when no row qualifies: 15 to 17 ms per statement on the 3-node test
  cluster. A replicated projection of the journal, sorted by the version
  column, makes that read local:

      CREATE PROJECTION app.docs_rep AS SELECT id, vec, del, ts FROM app.docs ORDER BY ts UNSEGMENTED ALL NODES;
      SELECT REFRESH('app.docs');
      SELECT ANALYZE_STATISTICS('app.docs');

  Check that `EXPLAIN SELECT ... FROM app.docs_delta` shows the scan of
  `docs_rep` with "Execute on: Query Initiator". Without the statistics the
  planner kept the segmented projection. Measured: the empty delta 4 ms
  instead of 17 ms, 1000 journal rows 7 ms instead of 25 ms (mixed mode). The
  price: every node stores the whole journal (81 MB per node for 100,000
  vectors of 128 dimensions, against 27 MB segmented over 3 nodes), and every
  insert writes to every node. Worth it when single searches with
  `freshness='exact'` matter and the journal fits on every node.
- **What each node has**:

      SELECT node_name, index_name, snapshot_id, vector_count, dims, metric, index_type, freshness_default, loaded
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='docs') OVER(PARTITION NODES) FROM vvector.probe) i;

         node_name    | index_name | snapshot_id | vector_count | dims | metric | index_type | freshness_default | loaded
      ----------------+------------+-------------+--------------+------+--------+------------+-------------------+--------
       v_vdb_node0001 | docs       |         240 |            5 |    3 | cosine | hnsw       | exact             | t

  Without `index_name` it lists every index in the cache directory.
  Other columns: max_ver, quantization, graph_bytes, tombstones,
  base_snapshot, precision_default, ef_search_default, threads_default,
  cache_file (or why the cache cannot be read).
- **Cache directory**: `/tmp/vvector` by default. Another one per session:
  `ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/data/vvector';`
  (the refresh and every query must use the same one), or `cache_dir=` per
  call. `/tmp` may be cleaned at reboot: that is safe (`load_all` restores
  it), but a node answers "run vload" until then.
- **Backup**: the snapshots are rows of `vvector.snapshot` and the options
  are rows of `vvector.manifest`: a backup of the database contains them.
- **Disk space**: a refresh keeps the active and the previous snapshot in the
  table and in every cache. Deleted rows of `vvector.snapshot` keep using
  space until Vertica purges them: `SELECT PURGE_TABLE('vvector.snapshot');`
  after many refreshes.
- **Library version**: `SELECT vvector.vversion() OVER();`

       library_version | format_version |                           build_flags
      -----------------+----------------+------------------------------------------------------------------
       0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 aarch64 g++-11.5.0 kernels=neon

  A new library that reads another snapshot format refuses the old cache
  files with "format version 1, this library reads version 2: refresh the
  index": run `refresh_index` for every index after such an upgrade.
- **Monitoring**: the refresh labels its statements `vvector_build` and
  `vvector_load`:
  `SELECT request_label, request_duration_ms FROM v_monitor.query_requests WHERE request_label LIKE 'vvector%' ORDER BY start_timestamp DESC;`
  Put your own `/*+LABEL(name)*/` in searches to find them the same way.

### Low-level functions

`refresh_index` and `load_all` call these; you need them only to build or load
by hand.

| Function | Rights | What it does |
|---|---|---|
| `vbuild(id, vec, del USING PARAMETERS index_name, metric, index_type, max_ver, m, ef_construction, threads, quantization, base_snapshot, cache_dir) OVER()` | PUBLIC | turns (id, vector) rows into a snapshot; returns (byte_offset, chunk, vector_count, dims, max_ver, format_version), chunks of 8 MB. Rows with `del = true` are left out. No ORDER BY: it sorts by id itself. `metric` l2 (default), cosine, dot, l1; `index_type` flat (default of the function; the procedures pass the index's type) or hnsw with `m` (16), `ef_construction` (200) and `threads` (0 = one per core) for the graph build |
| `vload(byte_offset, chunk USING PARAMETERS index_name, snapshot_id, cache_dir) OVER(PARTITION NODES)` | vvector_admin | writes the snapshot to the cache of the node, verifies it, makes it active; returns (node_name, snapshot_id, bytes, status). Run again at any time |
| `vconfig(k USING PARAMETERS index_name, options, cache_dir) OVER(PARTITION NODES) FROM vvector.probe` | vvector_admin | writes the index defaults (`options='precision=best,threads=4'`) to every node; returns (node_name, status) |
| `vnode(k) OVER(PARTITION NODES) FROM vvector.probe` | PUBLIC | one row per node: (node_name, k); used to send every chunk to every node exactly once |
| `vinfo([USING PARAMETERS index_name, cache_dir]) OVER(PARTITION NODES) FROM vvector.probe` | PUBLIC | what every node has cached (see above) |
| `vversion() OVER()` | PUBLIC | (library_version, format_version, build_flags) |

A snapshot built and loaded by hand (the procedures do the same, plus the
views, the manifest and the checks):

    INSERT INTO vvector.snapshot
    SELECT 'docs', 900, byte_offset, chunk FROM (
      SELECT vvector.vbuild(id, vec, FALSE USING PARAMETERS index_name='docs', metric='cosine') OVER()
      FROM app.docs) b;
    COMMIT;
    SELECT vvector.vload(byte_offset, chunk USING PARAMETERS index_name='docs', snapshot_id=900) OVER(PARTITION NODES)
    FROM (SELECT s.byte_offset, s.chunk FROM vvector.snapshot s CROSS JOIN vvector.probe p
          WHERE s.index_name = 'docs' AND s.snapshot_id = 900
            AND p.k IN (SELECT k FROM (SELECT vvector.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c;

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
| `vsearch: row with id I and neither vec nor del: allow-list rows ... (milestone M5)` | filtered search is not available yet | filter after the search (see Search) |
| `vsearch: k must be 1 to 16384`, `precision must be ...`, `freshness must be ...`, `ef_search must be ...`, `oversampling must be 1 to 100`, `threads must be 0 (one per core) to 64`, `radius must be a finite number` | a parameter out of range | use a value from the parameter table |
| `vsearch: session parameter NAME = 'v' is not an integer` (or `index default ...`) | a bad session value or index default | `ALTER SESSION SET UDPARAMETER FOR vvector NAME = ...`, or `set_index_options` |
| `vsearch: the snapshot of index 'x' changed while the query ran: run it again` | a refresh changed the dimensions during the query | run the query again |
| `vbuild: id I appears twice` | a static index (no version column) with a repeated id | make the ids unique, or register a version column |
| `vbuild: the vector of id I is NULL (a delete needs del = true)`, `... has a NULL element`, `... element N is not a finite float32 value` | a missing vector, a NULL, NaN, Infinity or a value beyond +-3.4e38 | fix the row |
| `vbuild: vector of id I has N elements, the ones before have M` | vectors of different lengths | one length per index |
| `vbuild: vectors have N elements, at most 32768 are supported` | too many dimensions | reduce the dimensions |
| `vbuild: metric must be l2, cosine, dot or l1`, `index_type must be flat or hnsw`, `quantization ...`, `m must be 2 to 256`, `ef_construction must be 1 to 100000` | bad build parameter | see the parameter tables |
| `vbuild: ... too many vectors for an HNSW graph with m = N` | the upper levels of the graph would need more than 4,294,967,294 blocks | a larger `m`, or split the index |
| `vload: on NODE: bad snapshot graph: ... in FILE`, `vsearch: bad snapshot graph: ...` | the graph section of the snapshot or cache file is damaged (vload checks every link) | `CALL vvector.refresh_index('x')` |
| `... is not implemented yet (milestone Mn)` | a feature of a later milestone | use what the message says is available |
| `vbuild: out of memory: cannot map N MB for the snapshot` | the node has too little memory for the build | `CALL vvector.sizing(...)`; a larger node or FencedUDxMemoryLimitMB |
| `vload: on NODE: pieces are missing or duplicated`, `bad snapshot: checksum mismatch`, `cannot write ...` | a damaged transfer or a full disk | check disk space of cache_dir; `load_all` |
| `vload`, `vinfo`, `vsearch`: `cache_dir '...' must be an absolute path`, `index name '...' is not valid` | bad cache_dir or index name | use `/path` and letters, digits, underscore |
| `vsearch: index 'x': OPTIONS file in the cache: ...: run vvector.load_all` | the index defaults file of the node was changed by hand | `CALL vvector.load_all('x')` |
| `vvector.register_index: ...` (table, column, type, metric, margin, op_col needs ver_col, already registered) | a bad argument; the message names it | fix the argument |
| `vvector.refresh_index: index x: table T has no vectors, nothing to build` | the table has no live rows | insert rows first |
| `vvector.load_on_nodes: index x, snapshot S: loaded on N of M nodes` | a node could not load (disk, rights) | vinfo shows the cause per node; `load_all` |
| `vvector.push_options: index x: options written on N of M nodes` | a node could not write its cache directory | check the cache directory; `load_all` |
| `vvector.<procedure>: index x is not registered` | a wrong index name | `SELECT index_name FROM vvector.manifest` |
| `vvector.set_index_options: ...` | a value out of range or of a later milestone | see the option table |
| `vvector.schedule_refresh: cron_expr may hold digits, spaces and * / , - only` | a bad cron expression | e.g. `'0 * * * *'` |

Warnings of `status` and `sizing` are explained in their text.

## Performance and results

Measured on the test VM: Vertica 26.2.0-1, one node, aarch64, 8 cores, 34 GB,
g++ 11.5. Data: SIFT1M (1,000,000 vectors of 128 dimensions, 10,000 queries
with ground truth, TEXMEX corpus), metric l2, k = 10. HNSW with m = 16,
ef_construction = 200. Three numbers, reported separately.

**Engine alone** (`make bench DATA_DIR=...`, no Vertica, all 10,000 queries):

| Measurement | Flat | HNSW |
|---|---:|---:|
| build of the snapshot (8 threads) | 0.08 s | 34 s |
| snapshot size | 496 MB | 631 MB |
| 1 query, 1 thread | 9.3 ms | 0.05 ms (fast) / 0.14 ms (balanced) |
| 1 query, 8 threads | 2.3 ms (the machine's full memory bandwidth) | (one thread per query) |
| queries per second, 8 threads | 990 (batch of 1000) | 125,000 (fast) / 49,000 (balanced) |
| recall@10 | 0.999 (exact; the rest are ties in distance) | 0.903 (fast) / 0.983 (balanced) / 0.999 (best) |

HNSW against hnswlib (v0.10.0-rc.2, the reference implementation), same
machine, same data and parameters (`make bench HNSWLIB_DIR=...`):

| ef_search | recall@10 vvector | recall@10 hnswlib | queries/s, 1 thread, vvector | hnswlib | queries/s, 8 threads, vvector | hnswlib |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.903 | 0.904 | 19,858 | 17,996 | 124,908 | 108,145 |
| 100 | 0.983 | 0.983 | 8,080 | 7,239 | 49,352 | 43,029 |
| 400 | 0.999 | 0.999 | 2,338 | 2,206 | 14,356 | 13,401 |

Equal recall, 6 to 16% more queries per second; build 34 s against 39 s.

**One statement** (`scripts/latency.sh`, median at the client, 200 runs, `query`
parameter):

| Statement | Fenced | Mixed (or unfenced) |
|---|---:|---:|
| `SELECT 1` (the floor of any statement) | 0.8 ms | 0.8 ms |
| HNSW: vsearch `FROM sift_hnsw_snap` | 7.8 ms | 1.9 ms |
| HNSW: the same over `sift_hnsw_delta`, `freshness='exact'`, empty delta | 8.9 ms | 2.7 ms |
| HNSW: `vknn ... FROM dual` | 7.8 ms | 1.4 ms |
| flat: vsearch `FROM sift_snap` | 12.4 ms | 5.1 ms |
| flat: the same over `sift_delta`, empty delta | 13.6 ms | 6.2 ms |
| SQL full scan (`ORDER BY VECTOR_L2(vec, q) LIMIT 10`) | 7673 ms | |

**Throughput, recall and refresh** (one statement with 1000 queries):

| Measurement | Fenced | Mixed |
|---|---:|---:|
| vsearch HNSW, precision fast | 22 ms (45,000 queries/s) | 12 ms (83,000 queries/s) |
| vsearch HNSW, precision balanced | 35 ms (29,000 queries/s) | 25 ms (40,000 queries/s) |
| vknn HNSW, precision fast, 1000 rows | 71 ms (14,000 queries/s) | 55 ms (18,000 queries/s) |
| vsearch flat | 1076 ms (930 queries/s) | 1098 ms (910 queries/s) |
| recall@10 against the ground truth: HNSW fast / balanced / best / exact; flat | 0.893 / 0.980 / 0.999 / 0.999; 0.999 | |
| `refresh_index`, 1M vectors: HNSW / flat | 46 s / 11 s | |

On the 3-node Eon test cluster (x86_64, 2 cores and 15 GB per node, Vertica
26.2.0-2; HNSW index of 100,000 random vectors of 128 dimensions) a statement
costs more: `SELECT 1` 2.7 ms, vsearch `_snap` 12.8 ms fenced and 7.6 ms mixed,
`vknn` 12.1 and 5.9 ms, the empty delta 27.9 and 24.2 ms (12 ms mixed with a
replicated journal, see [Operations](#operations)).

Fenced mode adds about 6 ms per statement, more than the search itself; for
single searches `FENCED=mixed` gives the unfenced latency while the memory-
heavy build stays fenced. For batches the difference is smaller. Where each
millisecond goes is in [docs/design.md](docs/design.md).

To reproduce (loads SIFT1M into schema VVBENCH; about 40 minutes with the
hnswlib comparison):

    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
    make && make tools && make deploy
    scripts/benchmark.sh --data_dir=$PWD/sift [--hnswlib=<a clone of github.com/nmslib/hnswlib>]

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
  an ARRAY argument); vbuild is PUBLIC.
- A UNION ALL of `ARRAY[INT]` and `ARRAY[NUMERIC]` columns fails inside
  Vertica 26.2 (INTERNAL 5445); cast to `ARRAY[FLOAT]` first.
- Fenced mode adds about 6 ms per statement; unfenced functions run inside
  the Vertica process, where a fault stops the node.
- On a multi-node cluster a statement over the `_delta` view reads the journal
  on every node and sends the rows to the node that runs the search, even when
  no row qualifies: 15 to 17 ms more per statement on the 3-node test cluster
  (1 ms on one node). Snapshot-only statements (`_snap` view, `FROM dual`,
  `vknn`) do not pay it; a replicated journal projection removes most of it
  (see [Operations](#operations)).
- An `ARRAY[...]` literal of many numbers is slow to parse (about 7 ms for
  128): use the `query` parameter.

Data model:
- Ids are INT and unique per index; every vector of an index has the same
  number of elements, at most 32768; NULL elements, NaN, Infinity and values
  beyond +-3.4e38 are refused.
- The table is a journal: adds and deletes are INSERTs; a physical UPDATE or
  DELETE becomes visible at the next refresh.
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
- On an HNSW index `radius` filters the candidates of the graph search: a
  vector inside the radius can be missed (use `exact=true`). Range search on
  the graph: M5.
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
- Not supported: product quantisation, IVF, DiskANN, sparse vectors, several
  vectors per id, GPU, hybrid text and vector search, sharding one index over
  nodes, big-endian hosts, Windows.

Operations:
- A new index default is seen by queries within 200 ms; a new snapshot at once.
- Two refreshes of one index at the same time are not supported.
- Cache files stay on the nodes after `unregister_index`.
- A refresh builds on one node; its memory is the build memory above.
- A node that missed a refresh answers "snapshot cache stale ... run vload"
  until `load_all` runs.
- A new snapshot format needs a refresh of every index; the error says so.
- Every refresh rebuilds the whole index (incremental refresh: M3); on the
  test VM 1M x 128 takes 11 s as a flat index and 46 s as an HNSW index. The
  graph of a parallel build depends on the order in which threads insert:
  two builds of the same data give slightly different graphs (and recall);
  the results of a search on a given snapshot are always the same.

## Files

| File | What it does |
|---|---|
| `Makefile` | `make`, `make test`, `make bench`, `make tools`, `make deploy [FENCED=yes\|no\|mixed]`, `make undeploy` |
| `src/engine/` | pure C++17, no Vertica includes: snapshot format, node cache, distance kernels, flat search, HNSW (`hnsw.cpp`), threads, query text |
| `src/udx/` | the Vertica adapters: one small file per SQL function (`vsearch.cpp`, `vknn.cpp`, `vbuild.cpp`, ...) |
| `sql/` | `install.sql`, `procedures.sql`, `uninstall.sql` |
| `scripts/` | `deploy.sh`, `register.sh`, `refresh.sh`, `load_dataset.sh`, `latency.sh`, `benchmark.sh` |
| `tools/fvecs.cpp` | converts `.fvecs`, `.ivecs`, `.bvecs` files (SIFT1M) to text for COPY |
| `tests/engine/` | unit tests (`make test`, among them `test_hnsw.cpp`) and the engine benchmarks (`make bench`: `bench_flat.cpp`, `bench_hnsw.cpp`, and `bench_hnswlib.cpp` with `HNSWLIB_DIR=`) |
| `tests/sql/` | integration tests: `test_snapshot.sh`, `test_freshness.sh`, `test_search.sh`, `test_hnsw.sh`; `run_all.sh` runs them in every mode |
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
