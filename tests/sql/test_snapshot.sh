#!/usr/bin/env bash
# Integration test of the snapshot path: vbuild -> vvector.snapshot -> vload ->
# vinfo -> vsearch reading the node cache (vsearch is a stub: it checks the cache
# and its input, then stops with "not implemented yet").
#
#   tests/sql/test_snapshot.sh [--rows=N] [--dims=N] [--schema=NAME] [--cache_dir=DIR] [--keep] [--echo_only]
#
# Test data: N random vectors of DIMS elements in <schema>.vectors (schema VVTEST
# or --schema=NAME; dropped and recreated unless --keep is given). The test index
# is called vvtest. Its snapshot rows are removed at the end.
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD,
# VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVTEST
CACHE_DIR=/tmp/vvector
KEEP=no
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --dims=*)      DIMS="${arg#--dims=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --keep)        KEEP=yes ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "test_snapshot.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

. tests/sql/lib.sh
CD=", cache_dir='$CACHE_DIR'"
# Input shape of vsearch: (qid, qvec, id, vec, del, ver, snapshot_id)
Q="1, $(random_vector "$DIMS")::ARRAY[FLOAT], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT"

if [ "$KEEP" = no ]; then
    expect "test data: $ROWS vectors of $DIMS elements" "^vectors: $ROWS$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.vectors (id INT NOT NULL, vec ARRAY[FLOAT] NOT NULL) ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.vectors SELECT id, $(random_vector "$DIMS") FROM ($(row_numbers "$ROWS")) g;
COMMIT;
SELECT 'vectors: ' || COUNT(*) FROM $SCHEMA.vectors;"
fi

echo "== build and load"
expect "vbuild into vvector.snapshot" "^chunks: [1-9]" "
DELETE FROM vvector.snapshot WHERE index_name = 'vvtest';
INSERT INTO vvector.snapshot
SELECT 'vvtest', 1, byte_offset, chunk FROM (
  SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest', metric='cosine', max_ver=4711) OVER(ORDER BY id)
  FROM $SCHEMA.vectors) b;
COMMIT;
SELECT 'chunks: ' || COUNT(*) FROM vvector.snapshot WHERE index_name = 'vvtest';"

expect "vload on every node" "^loaded on all nodes" "
SELECT CASE WHEN l.loaded = u.up THEN 'loaded on all nodes' ELSE 'loaded on ' || l.loaded || ' of ' || u.up || ' nodes' END
FROM (SELECT COUNT(DISTINCT node_name) AS loaded
      FROM (SELECT vvector.vload(byte_offset, chunk USING PARAMETERS index_name='vvtest'$CD, snapshot_id=1) OVER(PARTITION NODES)
            FROM (SELECT s.byte_offset, s.chunk FROM vvector.snapshot s CROSS JOIN vvector.probe p
                  WHERE s.index_name = 'vvtest' AND s.snapshot_id = 1
                    AND p.k IN (SELECT k FROM (SELECT vvector.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) g WHERE status = 'loaded') l
CROSS JOIN (SELECT COUNT(*) AS up FROM nodes WHERE node_state = 'UP') u;"

expect "vload again (idempotent)" "^loaded$" "
SELECT DISTINCT status FROM (SELECT vvector.vload(byte_offset, chunk USING PARAMETERS index_name='vvtest'$CD, snapshot_id=1) OVER(PARTITION NODES)
      FROM (SELECT s.byte_offset, s.chunk FROM vvector.snapshot s CROSS JOIN vvector.probe p
                  WHERE s.index_name = 'vvtest' AND s.snapshot_id = 1
                    AND p.k IN (SELECT k FROM (SELECT vvector.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) l;"

expect "vinfo: count, dims, metric and watermark match on every node" "^vinfo ok" "
SELECT CASE WHEN i.nodes_reporting = u.up AND i.min_count = t.n AND i.max_count = t.n AND i.min_dims = $DIMS
             AND i.max_dims = $DIMS AND i.metrics = 'cosine' AND i.min_ver = 4711 AND i.all_loaded = 1
            THEN 'vinfo ok' ELSE 'vinfo mismatch: ' || i.nodes_reporting || ' of ' || u.up || ' nodes, count ' ||
                 i.min_count || '..' || i.max_count || ' vs ' || t.n || ', dims ' || i.min_dims || ', metric ' || i.metrics END
FROM (SELECT COUNT(DISTINCT node_name) AS nodes_reporting, MIN(vector_count) AS min_count, MAX(vector_count) AS max_count,
             MIN(dims) AS min_dims, MAX(dims) AS max_dims, MAX(metric) AS metrics, MIN(max_ver) AS min_ver, MIN(loaded::INT) AS all_loaded
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='vvtest'$CD) OVER(PARTITION NODES) FROM vvector.probe) g) i
CROSS JOIN (SELECT COUNT(*) AS up FROM nodes WHERE node_state = 'UP') u
CROSS JOIN (SELECT COUNT(*) AS n FROM $SCHEMA.vectors) t;"

echo "== vbuild input rules"
expect "ARRAY[INT] vectors are accepted (Vertica converts them to ARRAY[FLOAT])" "^built: 2 vectors of 3$" "
SELECT 'built: ' || MAX(vector_count) || ' vectors of ' || MAX(dims) FROM (
  SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
  FROM (SELECT 1 AS id, ARRAY[1, 2, 3] AS vec UNION ALL SELECT 2, ARRAY[4, 5, 6]) v) b;"
# Not in one query with the INT arrays: a UNION of ARRAY[INT] and ARRAY[NUMERIC] fails inside Vertica 26.2.
expect "ARRAY[NUMERIC] vectors are accepted" "^built: 2 vectors of 3$" "
SELECT 'built: ' || MAX(vector_count) || ' vectors of ' || MAX(dims) FROM (
  SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
  FROM (SELECT 1 AS id, ARRAY[1.5, 2.5, 3.5]::ARRAY[NUMERIC(6,2)] AS vec UNION ALL SELECT 2, ARRAY[1, 2, 3]::ARRAY[NUMERIC(6,2)]) v) b;"
expect "a repeated id is refused" "id 7 appears twice" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
FROM (SELECT 7 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 7, ARRAY[3.0, 4.0]) v;"
expect "vectors of different lengths are refused" "has 3 elements, the ones before have 2" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 2, ARRAY[3.0, 4.0, 5.0]) v;"
expect "a NULL vector is refused" "id and vec must not be NULL" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 2, NULL::ARRAY[FLOAT]) v;"
expect "a NULL element is refused" "NULL element" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x') OVER(ORDER BY id)
FROM (SELECT 1 AS id, ARRAY[1.0, NULL]::ARRAY[FLOAT] AS vec) v;"
expect "an unknown metric is refused" "metric must be l2, cosine or dot" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x', metric='hamming') OVER(ORDER BY id)
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "index_type hnsw says it is not implemented yet" "HNSW is not implemented yet" "
SELECT vvector.vbuild(id, vec USING PARAMETERS index_name='vvtest_x', index_type='hnsw') OVER(ORDER BY id)
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"

echo "== vsearch (stub) and cache rules"
expect "vsearch reads the cache and its input, then says the search is not implemented yet" \
       "search is not implemented yet (1 query rows, 0 journal rows read)" "
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest'$CD, k=5) OVER() FROM dual;"

expect "a query vector of another length is refused" "has $DIMS dimensions, the query vectors have 2" "
SELECT vvector.vsearch(1, ARRAY[1.0, 2.0], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT, NULL::INT
                       USING PARAMETERS index_name='vvtest'$CD) OVER() FROM dual;"

expect "session parameter cache_dir is used" "no snapshot cache for index 'vvtest' in /tmp/vvector_not_there" "
ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/tmp/vvector_not_there';
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest') OVER() FROM dual;"

expect "function parameter cache_dir wins over the session parameter" "search is not implemented yet" "
ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/tmp/vvector_not_there';
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest', cache_dir='$CACHE_DIR') OVER() FROM dual;"

expect "stale cache is refused" "snapshot cache stale on .*: run vload" "
SELECT vvector.vsearch($Q, 999 USING PARAMETERS index_name='vvtest'$CD) OVER() FROM dual;"

expect "unknown index is refused" "no snapshot cache for index 'vvtest_none'" "
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest_none'$CD) OVER() FROM dual;"

expect "bad index name is refused" "is not valid" "
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='../etc') OVER() FROM dual;"

run_sql "cleanup" "DELETE FROM vvector.snapshot WHERE index_name = 'vvtest'; COMMIT;" > /dev/null

finish_tests test_snapshot
