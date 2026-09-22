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
  row in the delta view and `FROM dual` for snapshot-only queries.
- Every input column costs row transfer time (graph project: dropping two
  columns halved a 100M-row build).
- A large generic lambda around a hot output loop stopped inlining and cost 30%
  (graph project). Keep hot output loops in small functions; batch.
- Session parameter for the library: `getUDSessionParamReader("library")`,
  set with `ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'`. It is
  also visible inside stored procedures of the same session.
- `vt_report_error` inside `try` is fine; messages reach the client with file
  and line.

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
