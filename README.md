# vertica-vector-db (vvector)

Nearest-neighbour vector search inside Vertica: a C++ UDx library that keeps
an index of the vectors stored in a Vertica table and answers "the k vectors
closest to this one" from SQL, like a built-in function.

**Status: early work (milestone M0).** The plumbing works: build a snapshot
of a vector table, store it in Vertica, load it on every node, keep it fresh.
The search itself is not written yet: `vsearch` checks its input and the node
cache and then stops with "not implemented yet". The HNSW index is a stub.
Do not use this for anything but testing.

## Why

Vertica 26.2 has no vector type and no vector index. Vectors are stored as
arrays (`ARRAY[FLOAT]`, `ARRAY[INT]` or `ARRAY[NUMERIC]`), and the built-in
functions `COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and
`VECTOR_MAGNITUDE` compare two arrays. A nearest-neighbour query in SQL is a
full scan of the table:

    SELECT id, VECTOR_L2(vec, ARRAY[0.1, 0.2, 0.3]) AS distance
    FROM app.docs ORDER BY distance LIMIT 10;

The goal of this project is the same answer without the full scan, from an
index that lives inside Vertica and is never stale.

## What works today

| Piece | State |
|---|---|
| `vvector.vversion()` | library version, format version, build flags |
| `vvector.vbuild(id, vec)` | builds a snapshot (ids and float32 vectors) from the rows of a table |
| `vvector.vload(...)` | writes the snapshot to a cache file on every node |
| `vvector.vinfo()` | what every node has cached |
| `vvector.vsearch(...)` | reads its input and the cache, checks that the cache is current; **no search yet** |
| `register_index`, `refresh_index`, `load_all`, `status`, `schedule_refresh`, `unregister_index` | stored procedures: register a table, rebuild and load the snapshot, keep a view of the rows written since the last refresh |
| HNSW | stub: `index_type='hnsw'` says "not implemented yet" |

Tested on one Vertica 26.2 node (aarch64, Rocky Linux 9, g++ 11.5), fenced
and unfenced. Multi-node tests follow on a 3-node Eon cluster (milestone M2).

## Install

On a Vertica node (Vertica 26.x with the C++ SDK in `/opt/vertica/sdk`, g++
with C++17, GNU make):

    make
    make test                 # engine unit tests, no Vertica needed
    make deploy               # fenced mode, the default
    make deploy FENCED=no     # unfenced: faster, but inside the Vertica process
    make undeploy             # removes library and functions; schema vvector stays

Everything is created in schema `vvector`. Building, loading and the
procedures need the role `vvector_admin`.

Every script reads the connection from the environment (`VSQL_HOST`,
`VSQL_PORT`, `VSQL_USER`, `VSQL_PASSWORD`, `VSQL_DATABASE`). Every script
accepts `--help`, and every script that changes the database accepts
`--echo_only`: it prints the commands and changes nothing.

## Use (draft: the interface may still change)

A table of vectors, used as a journal: rows are only inserted, a delete is a
row with `del = TRUE`, and for every id the latest row wins.

    CREATE TABLE app.docs (
        id   INT NOT NULL,
        vec  ARRAY[FLOAT],                                     -- every vector has the same length
        del  BOOLEAN NOT NULL DEFAULT FALSE,
        ts   TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP()    -- version: set by the database
    )
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE
    GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);

    -- index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin
    CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', NULL);
    CALL vvector.refresh_index('docs');
    CALL vvector.schedule_refresh('docs', '0 * * * *');     -- every hour

    SELECT * FROM vvector.manifest;
    SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe;

Planned query (M1), reading the snapshot plus the rows written since the last
refresh from the view `app.docs_delta`:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', k=10) OVER()
    FROM (SELECT * FROM app.docs_delta
          UNION ALL SELECT 1, ARRAY[0.1, 0.2, 0.3], NULL, NULL, NULL, NULL, NULL) q;

`metric` is `l2`, `cosine` or `dot`, with the meaning of `VECTOR_L2`,
`COSINE_SIMILARITY` and `DOT_PRODUCT`.

## Files

| File | What it does |
|---|---|
| `Makefile` | builds `build/libvvector.so`, runs the unit tests, installs or removes the library |
| `src/engine/` | pure C++17, no Vertica includes: snapshot format, node cache, HNSW (stub) |
| `src/udx/` | the Vertica adapters: one small file per SQL function |
| `sql/` | `install.sql`, `procedures.sql`, `uninstall.sql` |
| `scripts/` | `deploy.sh`, `register.sh`, `refresh.sh` |
| `tests/engine/` | unit tests of the engine (`make test`) |
| `tests/sql/` | integration tests against a database: `test_snapshot.sh`, `test_freshness.sh` |
| `docs/` | `design.md` (freshness, cache rules), `format.md` (snapshot format), `VERTICA_NOTES.md` (verified Vertica behaviour) |

## Not supported

- Vectors of different lengths in one index.
- Ids other than Vertica INT.
- Vectors longer than 8,125 elements are untested: that is Vertica's default
  array bound (see `docs/VERTICA_NOTES.md`).
- Big-endian hosts.

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
