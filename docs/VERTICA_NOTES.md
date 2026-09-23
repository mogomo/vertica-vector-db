# Vertica behaviour verified for this project

Everything here was tested on Vertica 26.2, not taken from memory. Items
marked (graph project) were measured in the earlier project this code comes
from (github.com/mogomo/vertica-graph-udx), on single node and 3-node Eon.
Check again on other versions.

## Vectors (verified on 26.2.0-1, single node, 2026-09-23)
- There is no VECTOR type: `CREATE TABLE t (v VECTOR(3))` fails with
  "Type VECTOR does not exist".
- Vectors are arrays: `ARRAY[FLOAT]`, `ARRAY[INT]`, `ARRAY[NUMERIC(p,s)]`.
  Unbounded arrays get the default bound `DefaultArrayBinarySize` = 65000
  bytes (catalog type `array[float8](65000)`: 8125 FLOAT elements). A bound can
  be given: `ARRAY[FLOAT, 4096]` (elements) or `ARRAY[FLOAT](200000)` (bytes).
  Inserting more elements than the bound fails ("Output array isn't big enough").
- Built-in functions, all in library VectorOpsLib, schema public, argument
  type Any, result FLOAT: `COSINE_SIMILARITY(a, b)`, `DOT_PRODUCT(a, b)`,
  `VECTOR_L2(a, b)` (Euclidean distance), `VECTOR_MAGNITUDE(a)`. They accept
  FLOAT, INT and NUMERIC arrays and mixes of them; the result is FLOAT also
  for INT arrays.
- Arrays of different lengths: error "Arrays must be the same length (2 vs 3)".
  A NULL array or a NULL element gives NULL. A zero vector gives cosine 0.
  An empty array gives magnitude 0.
- No vector index: `CREATE INDEX` is not supported (only CREATE TEXT INDEX).
  `ORDER BY VECTOR_L2(v, q) LIMIT k` is a full scan followed by SORT [TOPK].
- Casts: `ARRAY[1, 2]::ARRAY[FLOAT]` and `ARRAY[NUMERIC]` to `ARRAY[FLOAT]`
  work. A cast to plain `ARRAY[FLOAT]` gives the default bound (65000 bytes).
- `NULL::ARRAY[FLOAT]` has the type `array[float8, 0]`. In a UNION ALL with a
  longer array the result type takes the larger bound, so a sentinel row of
  NULL arrays does not truncate query vectors.
- A UNION ALL of an `ARRAY[INT]` and an `ARRAY[NUMERIC]` column fails inside
  Vertica: "INTERNAL 5445: VIAssert(dt.oid == ((BaseDataOID) 6)) failed"
  (ParseClause.cpp). Each type alone works.
- A transform function declared with an `ARRAY[FLOAT]` argument
  (`addArrayType(Float8OID)`, catalog type `Float8Array1D`) also accepts
  `ARRAY[INT]` and `ARRAY[NUMERIC]` input without a cast.
- GRANT and REVOKE cannot name a function with an array argument: `GRANT
  EXECUTE ON TRANSFORM FUNCTION s.f(INT, ARRAY[FLOAT])` is a syntax error at
  "ARRAY"; `Float8Array1D` is "not a type". `GRANT EXECUTE ON ALL FUNCTIONS IN
  SCHEMA s TO r` works, and it also grants the stored procedures of the schema.
- An array column in the sort order of a table gives the warning "Sort clause
  contains a Array attribute ... data loads may be slowed significantly".
- `ARRAY_LENGTH(a)` and `APPLY_COUNT_ELEMENTS(a)` return the number of elements.
  `IMPLODE(x) WITHIN GROUP (ORDER BY ...)` builds an array from rows.
  `ARRAY[RANDOM(), RANDOM()]` builds random vectors in an INSERT ... SELECT.

## UDx SDK
- The SDK headers compile with `-std=c++17` (g++ 11.5; g++ 8.5 in the graph
  project). Flags come from `/opt/vertica/sdk/examples/makefile`;
  `_GLIBCXX_USE_CXX11_ABI=1`.
- Arrays: `#include "Arrays/Accessors.h"`, `argTypes.addArrayType(Float8OID)`,
  `Array::ArrayReader a = reader.getArrayRef(col)`, then `a->hasData()`,
  `a->isNull(0)`, `a->getFloatRef(0)`, `a->next()`. `Arrays/Arrays.cpp` is
  included by `Vertica.cpp`. `Float8OID` is a macro that names `BaseDataOID`
  without its namespace: outside `using namespace Vertica` it needs
  `using Vertica::BaseDataOID;`.
- A transform function with no arguments and no FROM works: `SELECT f() OVER()`.
- Vertica does not call a transform function on empty input. Hence the sentinel
  row in the delta view, and the `_snap` view (the sentinel alone) or `FROM dual`
  for snapshot-only queries.
- Every input column costs row transfer time (graph project: dropping two
  columns halved a 100M-row build).
- A large generic lambda around a hot output loop stopped inlining and cost 30%
  (graph project). Keep hot output loops in small functions; batch.
- Session parameter for the library: `getUDSessionParamReader("library")`,
  set with `ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'`. It is
  also visible inside stored procedures of the same session.
- `vt_report_error` inside `try` is fine; messages reach the client with file
  and line. (vvector throws `std::runtime_error` inside and reports once, in the catch: see below.)

## SQL
- `OVER(PARTITION NODES)` over an UNSEGMENTED table runs on one node only. Over
  a segmented table it runs on every node that holds rows (graph project, Eon).
- A scalar subquery next to an aggregate in the select list is rejected
  (ERROR 4817). Cross join one-row subqueries instead.
- A subquery in an ON clause is rejected (ERROR 4816). Put it in WHERE.
- `CREATE ROLE` has no IF NOT EXISTS. `CREATE SEQUENCE IF NOT EXISTS` exists;
  use `CACHE 1`, or ids jump by 250,000 per session.
- `epoch` cannot be selected into or sort a projection ("Column name epoch is
  reserved").
- vsql prints only the last result of several statements given with one `-c`.
- `/*+LABEL(x)*/` works in dynamic SQL inside procedures; find times in
  `v_monitor.query_requests` by label and `CURRENT_SESSION()`.
- The table-level hint goes after the table name: `FROM t /*+PROJS('s.p')*/`.
- `CREATE LOCAL TEMPORARY TABLE ... ON COMMIT PRESERVE ROWS AS SELECT ... KSAFE 0`
  keeps a large result in the database without a buddy projection.
- The catalog of libraries is `v_monitor.user_libraries` (not v_catalog).

## Time and order
- `CLOCK_TIMESTAMP()`: real clock, evaluated per row, accepted as a column
  DEFAULT. `SYSDATE()`/`GETDATE()`: statement start. `NOW()`/`CURRENT_TIMESTAMP`:
  transaction start.
- `CLOCK_TIMESTAMP()::TIMESTAMP` depends on the session time zone. Use TIMESTAMPTZ.
- `PARTITION BY ts::DATE` is rejected for TIMESTAMPTZ (non-deterministic);
  `(ts AT TIME ZONE 'UTC')::DATE` is accepted.
- Sequence numbers do not follow write order across sessions: three rows
  written in order by two sessions got 250001, 500001, 250002.
- `v_monitor.locks` (columns node_names, object_name = 'Table:schema.table',
  transaction_id, lock_mode, request_timestamp, grant_timestamp) shows the
  insert locks (mode I) of other sessions' open transactions, for INSERT and
  COPY. Uncommitted rows are invisible to every query.

## Storage and the delta read (graph project, 100M-row table, 1000 new rows)
- New rows in their own ROS container: 1 to 3 ms.
- After mergeout into the big container: 12 to 46 ms with default encoding,
  5 s with `ENCODING RLE` on the key column.
- Partitioned by version date: containers of different partitions are never
  merged, 1 ms, with any encoding. To be measured again with array columns.
- `DO_TM_TASK('mergeout', 'schema.table')` forces the case for tests.

## PL/vSQL
- `v := EXECUTE 'SELECT ...';` assigns from dynamic SQL. `x := (SELECT ...)`
  fails with "Query returned 0 rows" when nothing matches: use MAX().
- A local variable with the name of a column used in static SQL is an error
  ("Column reference ... is ambiguous").
- `refresh` and `load_snapshot` are built-in function names and cannot be
  procedure names.
- `PERFORM COMMIT;` works inside a procedure. NOTICEs of nested CALLs are not
  shown to the caller.
- A BOOLEAN concatenated into SQL text becomes `t`/`f`: write `true`/`false`.
- Scheduling: `CREATE SCHEDULE s USING CRON '...'` and `CREATE TRIGGER t ON
  SCHEDULE s EXECUTE PROCEDURE p('arg') AS DEFINER`; rows in
  `v_catalog.stored_proc_triggers`.
- REVOKE of a privilege that was never granted is a NOTICE, not an error.

## Memory
- `FencedUDxMemoryLimitMB` is -1 by default (no limit). UDx heap is outside the
  resource pools; an mmap'ed file is page cache, shared by all queries and
  reclaimable.

## Verified in milestone M1 (26.2.0-1, 2026-09-23)
- SDK, arrays: for a 1-D FLOAT array cell, `Array::ArrayReader a = in.getArrayRef(col)` gives
  `a->getNumRows()` (the element count) and `a->getFloatPtr(0)`, a pointer to the first element;
  `a->getColStride(0)` is 8: the elements are one contiguous run of float8 values in the input
  block. vvector reads vectors through that pointer, not with a call per element. A NULL element is
  the float8 value `vfloat_null` (a NaN pattern; test with `vfloatIsNull`).
- SDK: `TransformFunctionFactory::Properties` has `isExploder` ("expands rows 1:N"), set through
  `getFunctionProperties`; marked INTERNAL in the header. Tested with a small test library: with
  `isExploder = true` a transform function can be called without OVER() (`SELECT f(a) FROM t`),
  beside other columns (`SELECT b, f(a) FROM t`: every output row carries b of its input row), and
  with `OVER(PARTITION ROW)` or `OVER(PARTITION BEST)`. `LATERAL` is a syntax error. This is the
  shape for vknn (milestone M2).
- Parsing an ARRAY literal of 128 FLOAT numbers in a statement costs Vertica about 7 ms (measured
  as the difference between two otherwise equal vsearch statements); a VARCHAR parameter of the
  same numbers costs nothing measurable. Hence the `query` parameter of vsearch.
- A fenced transform function costs about 6 ms per statement more than an unfenced one
  (`vversion() OVER()`: 7.0 ms against 1.1 ms at the client, VM, 26.2.0-1).
- SDK: session parameters arrive as strings (`getUDSessionParamReader("library").getStringRef`),
  whatever the value looks like; vvector parses numbers and booleans itself.
- SDK: `vt_report_error` throws through a function pointer set by the server; the exception type is
  not visible to the library. The adapters throw `std::runtime_error` and report once, in the catch.
- `ALTER TABLE t ADD COLUMN IF NOT EXISTS c INT DEFAULT NULL` works; a second run gives
  "NOTICE 8778: Duplicate column name; nothing was done". install.sql upgrades the manifest this way.
- Procedures can be overloaded by argument count: `p(x INT, y INT)` and `p(x INT)` side by side,
  and one can `PERFORM CALL` the other. register_index uses it for the optional index_type.
- PL/vSQL: `x := (SELECT ...)::INT` (a cast outside the subquery) fails with "ERROR 5301:
  Unsupported use of sub-queries"; put the cast inside. A subquery inside an `IF` condition works.
- `GET_CONFIG_PARAMETER(...)` cannot be nested in another function or cast ("ERROR 2009: ... can
  not be used in function int8"). Read `v_monitor.configuration_parameters.current_value` instead.
- PL/vSQL: an `IF` whose condition is NULL (for example `IF x = 'a'` with x NULL) raises
  "ERROR 10268: Query returned null where a value was expected"; it is not treated as false.
- `v_monitor.host_resources` has `processor_core_count`, `total_memory_bytes`,
  `total_memory_free_bytes`, `total_memory_cache_bytes`. `GET_CONFIG_PARAMETER('FencedUDxMemoryLimitMB')`
  returns the limit as text ('-1' = none); so does `current_value` in
  `v_monitor.configuration_parameters`.
- `TO_JSON(vec)` prints an ARRAY[FLOAT] as `[0.12345678901234568,-1e-20,3.0]`: 17 significant
  digits, so the text reads back as the same double. `ARRAY_TO_STRING` does not accept FLOAT arrays
  and an array cannot be cast to VARCHAR (ERROR 2366). Use TO_JSON to build a `query` parameter.
- A correlated subquery with `<>` in its condition is rejected (ERROR 2788: "Correlated subquery in
  expression with operator <> is not supported"). Window functions (LAG, LEAD) do the same job.
- A view without FROM (`CREATE VIEW v AS SELECT NULL::INT AS qid, ..., 21 AS snapshot_id`) is
  allowed and is a valid input of a transform function: the `_snap` view.

## Verified in milestone M2 (26.2.0-1 single node and 26.2.0-2 3-node Eon, 2026-09-23)
- SDK, exploder (`isExploder = true`, used by vknn): the rows a transform function writes before it
  calls `in.next()` are paired with the columns of the current input row that are selected beside
  the function (`SELECT q.qid, vvector.vknn(q.qvec ...) FROM q`: every output row carries the qid
  of its input row). An input row for which the function writes nothing does not appear in the
  result. Vertica runs several instances in parallel, on every node that holds input rows. Works
  fenced and unfenced, on one node and on 3 nodes.
- `target_clones` works in a library loaded by Vertica: on an x86 node with AVX-512 the kernels
  resolve to the `avx512f` clone inside the Vertica process (vversion reports `kernels=avx512f`),
  fenced and unfenced.
- Eon, 3 nodes: a transform function with `OVER()` over the delta view (journal rows `WHERE ver >
  literal` UNION ALL one sentinel row) scans the journal on every node and sends the rows to the
  initiator (EXPLAIN: `Send`/`Recv` below the function), also when no journal row qualifies. That
  exchange costs about 15 ms per statement on the test cluster (1.1 ms on one node for the
  whole delta read). Over the `_snap` view (no FROM) or `FROM dual` there is no exchange.
- Eon: a projection added to a table that already has data is not used until it is refreshed:
  `CREATE PROJECTION` prints "WARNING 4468: Projection ... is not available for query processing.
  Execute the select start_refresh() function to copy data into this projection."
- Eon, 3 nodes: an UNSEGMENTED ALL NODES projection of the journal, ordered by the version column,
  removes that exchange, but only after two steps: `SELECT REFRESH('schema.table');` (synchronous;
  3 s for 100k rows of 128 FLOAT) fills it, and `SELECT ANALYZE_STATISTICS('schema.table');` makes
  the planner choose it. Before the statistics the planner kept the segmented projection. Then
  EXPLAIN shows the scan with "Execute on: Query Initiator" and no Send/Recv; the empty delta costs
  about 4 ms instead of 17 ms. `v_monitor.projection_storage` lists the full copy on every node.
- The planner kept using that projection after the table had doubled without new statistics
  (estimate 21K rows for the delta scan). A refresh build (`ROW_NUMBER() OVER(PARTITION BY id ...)`
  over the whole journal) still reads the segmented projection on all nodes.
- Eon: an UNSEGMENTED ALL NODES projection lives in the `replica` shard: `v_monitor.storage_containers`
  lists each of its containers once per node with the same `sal_storage_id` (one copy in communal
  storage); `v_monitor.projection_storage` shows the full size on every node (the depot).
  Bulk `INSERT ... SELECT` into the table was about 2.5 times slower with it (20k rows of 128 FLOAT:
  830 against 250 to 380 ms); single-row INSERT + COMMIT about the same.
- `v_monitor.disk_storage` (node_name, storage_usage 'DATA,TEMP' / 'DEPOT' / 'CATALOG',
  disk_space_free_mb) gives the free space per storage location, on Eon and on a single node.

## PL/vSQL (verified 2026-09-23, 26.2.0-1)
- `BEGIN ... EXCEPTION WHEN OTHERS THEN ... END;` inside a procedure catches an error of dynamic SQL
  (`EXECUTE 'CREATE PROJECTION ...'` on a name that exists: SQLSTATE 42710, SQLERRM `Object "x"
  already exists`); the procedure continues. `CREATE PROJECTION`, `SELECT REFRESH('t')`,
  `SELECT ANALYZE_STATISTICS('t.col')` and `DROP PROJECTION IF EXISTS` work through `EXECUTE`.
- Eon 26.2.0-2: with another session holding an open INSERT on a table, `CREATE PROJECTION` on it
  and `SELECT REFRESH('t')` wait until that transaction ends (12 s behind a writer that committed
  after 15 s); `ANALYZE_STATISTICS('t.col')` (60 ms), `DROP PROJECTION` and `SELECT` do not wait.

## Verified in milestone M3 (26.2.0-1 single node, 2026-09-23)
- `UPDATE` writes a new row version with a new commit epoch: the `epoch` pseudo-column of the
  updated row changed from 83557 to 83559, and the row count of the table stays the same. So a
  row count cannot see a physical UPDATE of an old journal row; incremental refresh therefore
  detects only rows that are removed or added with old versions (count of rows up to the
  previous boundary), and a physical UPDATE needs `refresh_index(name, 'full')`. Not used: row
  epochs for detection. What mergeout does to the epoch of rows older than the AHM was not
  verified (on the test VM the AHM did not move for several minutes, and `MAKE_AHM_NOW()`
  would have changed the whole database's history).
- `DO_TM_TASK('mergeout', 'schema.table')` runs a mergeout of one table.
- Deleted rows of `vvector.snapshot` (an old snapshot of 574 to 612 MB every refresh) were purged
  by the Tuple Mover by itself during 100 refreshes in 15 minutes: the disk use stayed within
  7 GB of its start. Vertica stores the 8 MB chunks compressed to about half.
- PL/vSQL: `x := SPLIT_PART(text_var, ' ', 1)::INT;` works (a cast on an expression, not a subquery).
- `OCTET_LENGTH(chunk)` of LONG VARBINARY reads the value: `SUM(OCTET_LENGTH(chunk))` over a
  574 MB snapshot takes 1.1 s. `MAX(byte_offset)` plus the length of that one chunk takes
  milliseconds.
- Stored procedures run with the rights of the caller, also for nested CALLs: a user with only the
  role vvector_admin got "ERROR 3457: Function vvector.register_index_core(...) does not exist, or
  permission is denied" until EXECUTE on the internal procedures was granted to vvector_admin
  (found in M3; the gap existed since M1, when every test ran as dbadmin). With the grants such a
  user registers, refreshes (full and incremental), asks status and unregisters, given USAGE and
  CREATE on the schema of the source table and SELECT on it.
- `v_monitor.locks` shows a non-superuser the locks of other users' sessions too (an open INSERT of
  dbadmin was visible to a user with only vvector_admin): the delta boundary works for them.
