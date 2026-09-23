# vertica-vector-db (vvector)

Nearest-neighbour vector search inside Vertica. vvector is a C++ UDx library
that keeps an index of the vectors stored in a Vertica table and answers "the
k vectors closest to this one" from SQL. The index lives in Vertica, is loaded
on every node, and every query can see the rows written since the last
refresh.

**Status: milestone M1 (exact search).** Exact k-nearest-neighbour search
works for the metrics l2, cosine, dot and l1, for one query or thousands in
one statement, with or without the rows written since the last refresh.
Results are tested to equal a full scan with Vertica's built-in functions.
Tested on Vertica 26.2 on one node (aarch64, Rocky Linux 9, g++ 11.5),
fenced, unfenced and mixed. Not yet available: the HNSW index (milestone M2),
tests on a multi-node Eon cluster and x86_64 (M2), incremental refresh (M3),
int8 quantisation (M4), filtered search and vector functions (M5). Treat this
as a preview: try it on your own systems before you rely on it.

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

On 1,000,000 vectors of 128 dimensions that takes 7.3 seconds on the test
machine. vvector answers the same query with the same result in 5 ms
(unfenced) or 12 ms (fenced), from a snapshot of the vectors that is stored
in Vertica, cached on every node, and kept exact by applying the rows written
since the last refresh.

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

The same query with the built-in function returns the same ids and scores
(vvector computes in 32-bit floats, so scores agree to about 7 digits):

    SELECT id, COSINE_SIMILARITY(vec, ARRAY[1, 0.2, 0]) AS score FROM app.docs ORDER BY score DESC LIMIT 3;

     id |       score
    ----+-------------------
      2 | 0.996240588195683
      1 |  0.98058067569092
      5 | 0.832050294337844

## Install

### Prerequisites

- Vertica 26.x (tested: 26.2.0-1) with the C++ SDK in `/opt/vertica/sdk`
  (another place: `make SDK_HOME=...`).
- g++ with C++17 (tested: 11.5 on aarch64) and GNU make, on a Vertica node:
  `CREATE LIBRARY` reads the .so from the initiator node's file system and
  copies it to the other nodes.
- A database user that may create a schema, a library, functions and a role
  (dbadmin, or a user with those rights).

### Build, test, deploy

    make                      # build/libvvector.so
    make test                 # engine unit tests, no database needed
    make deploy               # install into the database, fenced (the default)
    make deploy FENCED=no     # every function inside the Vertica process
    make deploy FENCED=mixed  # vbuild, vload, vconfig, vnode fenced; vsearch, vinfo, vversion not fenced
    make undeploy             # remove the library and its functions; tables and data stay
    tests/sql/run_all.sh      # integration tests: fenced, unfenced, mixed (creates test schemas)

What the modes mean:

| Mode | Where the functions run | For | Risk |
|---|---|---|---|
| `yes` (default) | a separate fenced process per session | safety first | none for the node; about 6 ms more per statement |
| `mixed` | build and load fenced, search in the Vertica process | single searches with low latency | a fault in vsearch or vinfo would stop the node |
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
| functions | `vsearch`, `vinfo`, `vversion` (search and information), `vbuild`, `vload`, `vconfig`, `vnode` (build and load) |
| procedures | `register_index`, `set_index_options`, `refresh_index`, `load_all`, `status`, `sizing`, `schedule_refresh`, `unregister_index` |

Rights: `vsearch`, `vinfo`, `vversion`, `vbuild`, `vnode` and the procedure
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
| index_type | `flat` (default). `hnsw` comes with milestone M2 |

It creates the views `<schema>.<index>_snap` and, with a version column,
`<schema>.<index>_delta` in the schema of the source table:

    NOTICE 2005:  vvector: index docs registered. Next: CALL vvector.refresh_index('docs'). Queries read docs_snap (snapshot only) or docs_delta (with the changes since the refresh) in schema app; grant SELECT on them to the users who may search the index.

### refresh_index

    CALL vvector.refresh_index('docs');

    NOTICE 2005:  vvector: index docs refreshed: snapshot 101, 5 vectors of 3 dimensions, 0 MB, 0.223 seconds

It takes the delta boundary, builds a new snapshot from the latest row of
every id (deletes left out), stores it in `vvector.snapshot`, loads it on
every node, writes the index defaults to every node, updates the manifest and
the views, and deletes snapshots older than the previous one. Queries keep
working during a refresh. Every refresh is a full rebuild until milestone M3.

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

NULL keeps a value. Query defaults (the last four) apply at once on every
node (within 200 ms); `'default'` (text) or `0` (numbers) sets one back to the
built-in default. Build options apply at the next refresh. Values that belong
to later milestones are refused with a message that names the milestone.

| Option | Values | Default | Now |
|---|---|---|---|
| index_type | flat, hnsw | flat | flat only (hnsw: M2) |
| m, ef_construction | 2 to 256, 1 to 100000 | 16, 200 | stored for HNSW (M2) |
| quantization | none, sq8 | none | none only (sq8: M4) |
| refresh_mode | auto, incremental, full | auto | every refresh is full (incremental: M3) |
| tombstone_ratio, rebuild_every | above 0 to 1; 0 = never | 0.2; never | stored for M3 |
| memory_mode | ram, compact | ram | ram only (compact: M4) |
| precision_default | fast, balanced, best, exact | fast | accepted; a flat index is always exact |
| freshness_default | snapshot, exact | snapshot | in use |
| ef_search_default | 0 to 100000 | 0 (preset) | stored for HNSW (M2) |
| threads_default | 0 (one per core) to 64 | 0 | in use |

### status and sizing

    CALL vvector.status('docs');

    NOTICE 2005:  vvector: index docs: flat index, refresh_mode auto, 0 tombstones
    NOTICE 2005:  vvector: index docs: 7 journal rows in the delta, read in 8 ms
    NOTICE 2005:  vvector: index docs: sizing: index 0 MB, all indexes 495 MB, build about 0 MB, smallest node 35155 MB of memory (31496 MB free or cache), 8 cores

`status` reports the rows in the delta and how long they take to read, open
transactions that write to the table, versions in the future (a sign that an
application sets the version column itself), and a sizing check: index size
against node memory, the memory a refresh needs against
`FencedUDxMemoryLimitMB` and free memory, `threads_default` against cores, and
all indexes together against the page cache. Each problem is a WARNING with
the recommended fix. It never changes anything. `refresh_index` runs it first.

`sizing` estimates the memory of an index before you load the table; anyone
may call it:

    CALL vvector.sizing(10000000, 768, 'flat', 'none');

    NOTICE 2005:  vvector.sizing: 10000000 vectors of 768 dimensions (768 floats per row): vectors 29296.9 MB, ids 76.3 MB, graph 0.0 MB, sq8 codes 0.0 MB
    NOTICE 2005:  vvector.sizing: snapshot and cache file 29373.2 MB per node; build memory about 29411.3 MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)
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
the delta view, `vector_count`, `dims`, `index_bytes`, `built_at`,
`build_seconds`, `format_version`).

    SELECT index_name, source_table, metric, index_type, active_snapshot, vector_count, dims, index_bytes, build_seconds
    FROM vvector.manifest WHERE index_name = 'docs';

     index_name | source_table | metric | index_type | active_snapshot | vector_count | dims | index_bytes | build_seconds
    ------------+--------------+--------+------------+-----------------+--------------+------+-------------+---------------
     docs       | app.docs     | cosine | flat       |             101 |            5 |    3 |         640 |         0.223

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
| radius | off | a number | only neighbours within it, at most k: l2 and l1 `score <= radius`; cosine and dot `score >= radius` |
| threads | 0 | 0 (one per core) to 64 | threads for one statement |
| precision | fast | fast, balanced, best, exact | the speed and recall trade-off of HNSW (M2); a flat index is always exact |
| ef_search | 0 (preset) | 0 to 100000 | HNSW candidate list (M2); no effect on a flat index |
| exact | false | true, false | force the exact search on an HNSW index (M2) |
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

`app.docs_snap` is one row that carries the active snapshot id. Pass the
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

One statement with 1000 queries searches 1M vectors at about 950 queries per
second on 8 cores: much faster than 1000 statements.

### With the changes since the refresh

After the refresh above, vector 6 is added and vector 2 is deleted:

    INSERT INTO app.docs (id, vec) VALUES (6, ARRAY[1.0, 0.25, 0.0]);
    INSERT INTO app.docs (id, del) VALUES (2, TRUE);
    COMMIT;

The delta view now holds these rows and the sentinel:

    SELECT id, del, ver IS NOT NULL AS has_ver, snapshot_id FROM app.docs_delta ORDER BY id;

     id | del | has_ver | snapshot_id
    ----+-----+---------+-------------
        |     | f       |         102
      2 | t   | t       |         102
      6 | f   | t       |         102

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

At most k rows are returned; raise k for a wide radius.

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

Milestone M1 has the `flat` index: every search compares the query with every
vector of the snapshot (exactly, with SIMD instructions and all cores). It is
exact by definition, so `precision`, `ef_search`, `exact`, `rescore` and
`oversampling` change nothing on it; they are accepted so that the same SQL
works when the index becomes HNSW (milestone M2), which searches a graph
instead of every vector.

| If you want | Set |
|---|---|
| the lowest latency for single queries | the `query` parameter, `FROM <index>_snap`, deploy with `FENCED=mixed` |
| results that include every committed change | `freshness='exact'` and `FROM <index>_delta` (per query, session or index) |
| many queries at once | one statement with all query rows (a table), not one statement per query |
| fewer cores for one statement | `threads=N` |
| the same default for every user of an index | `set_index_options` |

Memory:

| What | How much |
|---|---|
| snapshot and cache file per node | 256 bytes + (4 x row_stride + 8) bytes per vector; row_stride = dims rounded up to a multiple of 16 |
| a refresh (on one node) | about the snapshot size + 4 bytes per vector; fenced it counts against `FencedUDxMemoryLimitMB` (-1 = no limit) |
| a query | 4 x row_stride bytes per query and per journal row, plus 1 bit per vector when the journal has rows |

1,000,000 vectors of 128 dimensions take 496 MB; of 768 dimensions, 2.9 GB.
Queries read the cache file through the operating system's page cache: keep
all indexes of a node in memory (`status` warns when they do not fit).

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
  of 128 numbers. `status` warns above 100,000 rows or when reading the delta
  is slow; refresh more often then.
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
  Tested on one node so far; the 3-node Eon test comes with milestone M2.
- **What each node has**:

      SELECT node_name, index_name, snapshot_id, vector_count, dims, metric, index_type, freshness_default, loaded
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='docs') OVER(PARTITION NODES) FROM vvector.probe) i;

         node_name    | index_name | snapshot_id | vector_count | dims | metric | index_type | freshness_default | loaded
      ----------------+------------+-------------+--------------+------+--------+------------+-------------------+--------
       v_vdb_node0001 | docs       |         101 |            5 |    3 | cosine | flat       | exact             | t

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
| `vbuild(id, vec, del USING PARAMETERS index_name, metric, index_type, max_ver, m, ef_construction, threads, quantization, base_snapshot, cache_dir) OVER()` | PUBLIC | turns (id, vector) rows into a snapshot; returns (byte_offset, chunk, vector_count, dims, max_ver, format_version), chunks of 8 MB. Rows with `del = true` are left out. No ORDER BY: it sorts by id itself. `metric` l2 (default), cosine, dot, l1 |
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

Every message starts with the function or procedure that raised it.

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
| `vbuild: metric must be l2, cosine, dot or l1`, `index_type ...`, `quantization ...`, `m ...`, `ef_construction ...` | bad build parameter | see the parameter tables |
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
with ground truth, TEXMEX corpus), metric l2, k = 10, flat index (milestone
M1). Three numbers, reported separately:

**Engine alone** (`make bench DATA_DIR=...`, no Vertica):

| Measurement | Result |
|---|---:|
| build of the snapshot | 0.08 s |
| 1 query, 1 thread | 8.6 ms (59 GB/s of vectors, 69% of the measured memory bandwidth of one thread) |
| 1 query, 8 threads | 2.3 ms (222 GB/s, 100% of the machine's memory bandwidth) |
| batch of 1000 queries, 8 threads | 1000 queries/s |
| recall@10 against the ground truth | 0.999 (exact search; the rest are ties in distance) |

**One statement** (`scripts/latency.sh`, median at the client, 200 runs):

| Statement | Fenced | Unfenced (or mixed) |
|---|---:|---:|
| `SELECT 1` (the floor of any statement) | 0.8 ms | 0.8 ms |
| vsearch, `query` parameter, `FROM sift_snap` | 12.3 ms | 5.0 ms |
| the same over `sift_delta`, `freshness='exact'`, empty delta | 13.5 ms | 6.2 ms |
| the same with 1000 journal rows | 16.4 ms | 7.8 ms |
| SQL full scan (`ORDER BY VECTOR_L2(vec, q) LIMIT 10`) | 7267 ms | |

**Throughput and refresh**:

| Measurement | Fenced | Unfenced |
|---|---:|---:|
| 1000 queries in one statement | 1044 ms (957 queries/s) | 1059 ms (944 queries/s) |
| `refresh_index`, 1M vectors | 9.4 s | |
| recall@10 of vsearch against the ground truth | 0.9994 | |

Fenced mode adds about 6 ms per statement, more than the search itself; for
single searches `FENCED=mixed` gives the unfenced latency while the memory-
heavy build stays fenced. For batches the modes are within 2%. Where each
millisecond goes is in [docs/design.md](docs/design.md).

To reproduce (loads SIFT1M into schema VVBENCH, 10 to 20 minutes):

    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
    make && make tools && make deploy
    scripts/benchmark.sh --data_dir=$PWD/sift

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
- Flat index only (HNSW: M2). A search reads the whole snapshot: about 2 to
  3 ms per million vectors of 128 dimensions on 8 cores.
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
  test VM 1M x 128 takes 9.4 s.

## Files

| File | What it does |
|---|---|
| `Makefile` | `make`, `make test`, `make bench`, `make tools`, `make deploy [FENCED=yes\|no\|mixed]`, `make undeploy` |
| `src/engine/` | pure C++17, no Vertica includes: snapshot format, node cache, distance kernels, flat search, threads, query text, HNSW (stub) |
| `src/udx/` | the Vertica adapters: one small file per SQL function |
| `sql/` | `install.sql`, `procedures.sql`, `uninstall.sql` |
| `scripts/` | `deploy.sh`, `register.sh`, `refresh.sh`, `load_dataset.sh`, `latency.sh`, `benchmark.sh` |
| `tools/fvecs.cpp` | converts `.fvecs`, `.ivecs`, `.bvecs` files (SIFT1M) to text for COPY |
| `tests/engine/` | unit tests (`make test`) and the engine benchmark (`make bench`) |
| `tests/sql/` | integration tests: `test_snapshot.sh`, `test_freshness.sh`, `test_search.sh`; `run_all.sh` runs them in every mode |
| `docs/` | `design.md` (decisions, measurements), `format.md` (snapshot format), `VERTICA_NOTES.md` (verified Vertica behaviour) |

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
