#!/usr/bin/env bash
# Integration test of the snapshot path: vbuild -> vvector.snapshot -> vload ->
# vinfo -> vsearch reading the node cache; the input rules of vbuild and vsearch, and
# the cache rules (session parameter, stale cache, unknown index); the index option cache_dir
# (the procedures load into it, a query without cache_dir finds it through the default directory,
# a bad or unusable directory is refused, back to the default).
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
# The chunks joined to one probe row per node, as load_on_nodes does: the snapshot table is segmented,
# so on more than one node the chunks are broadcast (the hint gives only a warning on one node).
NODES=1
[ "$ECHO_ONLY" = yes ] || NODES=$(vsql -X -A -t -c "SELECT COUNT(*) FROM nodes WHERE node_state = 'UP'" 2>/dev/null || echo 1)
if [ "${NODES:-1}" -gt 1 ]; then
    CHUNKS="SELECT /*+SYNTACTIC_JOIN*/ s.byte_offset, s.chunk FROM vvector.probe p JOIN /*+DISTRIB(L,B)*/ vvector.snapshot s ON TRUE"
else
    CHUNKS="SELECT s.byte_offset, s.chunk FROM vvector.probe p JOIN vvector.snapshot s ON TRUE"
fi
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
expect "vvector.snapshot is segmented (one copy per refresh, not one per node)" "^segmented$" "
SELECT CASE WHEN COUNT(*) > 0 AND MIN(is_segmented::INT) = 1 THEN 'segmented' ELSE 'not segmented' END
FROM v_catalog.projections WHERE projection_schema = 'vvector' AND anchor_table_name = 'snapshot';"
expect "vbuild into vvector.snapshot" "^chunks: [1-9]" "
DELETE FROM vvector.snapshot WHERE index_name = 'vvtest';
INSERT INTO vvector.snapshot
SELECT 'vvtest', 1, byte_offset, chunk FROM (
  SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest', metric='cosine', max_ver=4711) OVER()
  FROM $SCHEMA.vectors) b;
COMMIT;
SELECT 'chunks: ' || COUNT(*) FROM vvector.snapshot WHERE index_name = 'vvtest';"

expect "vload on every node" "^loaded on all nodes" "
SELECT CASE WHEN l.loaded = u.up THEN 'loaded on all nodes' ELSE 'loaded on ' || l.loaded || ' of ' || u.up || ' nodes' END
FROM (SELECT COUNT(DISTINCT node_name) AS loaded
      FROM (SELECT vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='vvtest'$CD, snapshot_id=1) OVER(PARTITION NODES)
            FROM ($CHUNKS
                  WHERE s.index_name = 'vvtest' AND s.snapshot_id = 1
                    AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) g WHERE status = 'loaded') l
CROSS JOIN (SELECT COUNT(*) AS up FROM nodes WHERE node_state = 'UP') u;"

expect "vload again (idempotent)" "^loaded$" "
SELECT DISTINCT status FROM (SELECT vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='vvtest'$CD, snapshot_id=1) OVER(PARTITION NODES)
      FROM ($CHUNKS
                  WHERE s.index_name = 'vvtest' AND s.snapshot_id = 1
                    AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) l;"

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
  SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
  FROM (SELECT 1 AS id, ARRAY[1, 2, 3] AS vec UNION ALL SELECT 2, ARRAY[4, 5, 6]) v) b;"
# Not in one query with the INT arrays: a UNION of ARRAY[INT] and ARRAY[NUMERIC] fails inside Vertica 26.2.
expect "ARRAY[NUMERIC] vectors are accepted" "^built: 2 vectors of 3$" "
SELECT 'built: ' || MAX(vector_count) || ' vectors of ' || MAX(dims) FROM (
  SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
  FROM (SELECT 1 AS id, ARRAY[1.5, 2.5, 3.5]::ARRAY[NUMERIC(6,2)] AS vec UNION ALL SELECT 2, ARRAY[1, 2, 3]::ARRAY[NUMERIC(6,2)]) v) b;"
expect "a repeated id is refused" "id 7 appears twice" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 7 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 7, ARRAY[3.0, 4.0]) v;"
expect "vectors of different lengths are refused" "has 3 elements, the ones before have 2" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 2, ARRAY[3.0, 4.0, 5.0]) v;"
expect "a NULL vector is refused" "the vector of id 2 is NULL (a delete needs del = true)" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec UNION ALL SELECT 2, NULL::ARRAY[FLOAT]) v;"
expect "a NULL element is refused" "the vector of id 1 has a NULL element" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, NULL]::ARRAY[FLOAT] AS vec) v;"
expect "an unknown metric is refused" "metric must be l2, cosine, dot or l1" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', metric='hamming') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "index_type hnsw builds a graph, also of one vector" "^built: 1 vectors of 2$" "
SELECT 'built: ' || MAX(vector_count) || ' vectors of ' || MAX(dims) FROM (
  SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', index_type='hnsw') OVER()
  FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v) b;"
expect "an unknown index_type is refused" "index_type must be flat or hnsw, not 'ivf'" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', index_type='ivf') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "quantization sq8 builds a snapshot with codes (test_sq8.sh tests them)" "^chunks 1, vectors 1$" "
SELECT 'chunks ' || COUNT(*) || ', vectors ' || MAX(vector_count) FROM (
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', quantization='sq8') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v) b;"
expect "an unknown quantization is refused" "vbuild: quantization must be none or sq8, not 'pq'" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', quantization='pq') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "an incremental build needs its base snapshot in the node cache" "base snapshot 5 is not usable in the cache of .*refresh with mode full" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', base_snapshot=5$CD) OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "rows with del = true are left out of a full build; l1 is a metric" "^built: 2 vectors of 2$" "
SELECT 'built: ' || MAX(vector_count) || ' vectors of ' || MAX(dims) FROM (
  SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='vvtest_x', metric='l1') OVER()
  FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec, FALSE AS del UNION ALL SELECT 2, ARRAY[3.0, 4.0], TRUE
        UNION ALL SELECT 3, ARRAY[5.0, 6.0], NULL) v) b;"
expect "build_in='file' builds the same snapshot as a build in memory (sizes and counts)" "^same$" "
SELECT CASE WHEN r.v = f.v THEN 'same' ELSE 'ram ' || r.v || ', file ' || f.v END FROM
 (SELECT COUNT(*) || ' chunks ' || SUM(OCTET_LENGTH(chunk)) || ' bytes ' || MAX(vector_count) || ' vectors' AS v FROM (
   SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', index_type='hnsw'$CD) OVER() FROM $SCHEMA.vectors) b) r
 CROSS JOIN
 (SELECT COUNT(*) || ' chunks ' || SUM(OCTET_LENGTH(chunk)) || ' bytes ' || MAX(vector_count) || ' vectors' AS v FROM (
   SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', index_type='hnsw', build_in='file'$CD) OVER() FROM $SCHEMA.vectors) b) f;"
expect "an unknown build_in is refused" "build_in must be ram or file, not 'disk'" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x', build_in='disk') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 2.0] AS vec) v;"
expect "a NaN element is refused" "element 2 is not a finite float32 value" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 1 AS id, ARRAY[1.0, 'NaN'::FLOAT] AS vec) v;"
expect "a value beyond the float32 range is refused" "element 1 is not a finite float32 value" "
SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='vvtest_x') OVER()
FROM (SELECT 1 AS id, ARRAY[1e300, 1.0] AS vec) v;"

echo "== vsearch and cache rules"
expect "vsearch returns k rows ranked 1 to k, the closest first" "^rows 5, ranks 1 to 5, ordered$" "
SELECT 'rows ' || COUNT(*) || ', ranks ' || MIN(rank) || ' to ' || MAX(rank) || CASE WHEN MIN(ok) = 1 THEN ', ordered' ELSE ', NOT ordered' END
FROM (SELECT rank, CASE WHEN LAG(score) OVER(ORDER BY rank) IS NULL OR LAG(score) OVER(ORDER BY rank) >= score THEN 1 ELSE 0 END AS ok
      FROM (SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest'$CD, k=5) OVER() FROM dual) r) x;"

expect "a query vector of another length is refused" "index 'vvtest' has $DIMS dimensions, query 1 has 2" "
SELECT vvector.vsearch(1, ARRAY[1.0, 2.0], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT, NULL::INT
                       USING PARAMETERS index_name='vvtest'$CD) OVER() FROM dual;"

expect "a journal row without an id is refused (only the sentinel row has none)" "a journal row has no id" "
SELECT vvector.vsearch(NULL::INT, NULL::ARRAY[FLOAT], NULL::INT, ARRAY[1.0, 2.0], FALSE, NULL::INT, NULL::INT
                       USING PARAMETERS index_name='vvtest'$CD) OVER() FROM dual;"

expect "session parameter cache_dir is used" "no snapshot cache for index 'vvtest' in /tmp/vvector_not_there" "
ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/tmp/vvector_not_there';
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest') OVER() FROM dual;"

expect "function parameter cache_dir wins over the session parameter" "^rows: 3$" "
ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/tmp/vvector_not_there';
SELECT 'rows: ' || COUNT(*) FROM (SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest', cache_dir='$CACHE_DIR', k=3) OVER() FROM dual) r;"

expect "stale cache is refused" "snapshot cache stale on .*: run vload" "
SELECT vvector.vsearch($Q, 999 USING PARAMETERS index_name='vvtest'$CD) OVER() FROM dual;"

expect "unknown index is refused" "no snapshot cache for index 'vvtest_none'" "
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='vvtest_none'$CD) OVER() FROM dual;"

expect "bad index name is refused" "is not valid" "
SELECT vvector.vsearch($Q, NULL::INT USING PARAMETERS index_name='../etc') OVER() FROM dual;"

echo "== the index option cache_dir"
# The directory is inside the test's cache directory, so it is on the same disk (on a cluster the
# default may be a link to a data disk). ".alt" is not a valid index name: never listed as an index.
ALT="$CACHE_DIR/.alt"
IXC=vvtest_cd
SEARCH_C="SELECT 'rows: ' || COUNT(*) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='$IXC', query='[$(printf '0.5, %.0s' $(seq 1 $((DIMS - 1))))0.5]', k=3) OVER() FROM $SCHEMA.${IXC}_snap) r;"
vinfo_file() {  # PATTERN: every UP node reports the snapshot file of $IXC under PATTERN (vinfo without cache_dir)
    echo "SELECT CASE WHEN COUNT(DISTINCT v.node_name) = MAX(u.up) AND MIN(CASE WHEN v.cache_file LIKE '$1/$IXC/%' THEN 1 ELSE 0 END) = 1
                 THEN 'every node: $1' ELSE 'cache_file ' || MAX(v.cache_file) || ' on ' || COUNT(DISTINCT v.node_name) || ' nodes' END
          FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IXC') OVER(PARTITION NODES) FROM vvector.probe) v
          CROSS JOIN (SELECT COUNT(*) AS up FROM nodes WHERE node_state = 'UP') u;"
}
run_sql "cleanup of an earlier run" "CALL vvector.unregister_index('$IXC');" > /dev/null
[ "$ECHO_ONLY" = yes ] || rm -rf "${ALT:?}/$IXC"
expect "register a static index with cache_dir $ALT, refresh" "index $IXC refreshed: snapshot [0-9]*, full build" "
CALL vvector.register_index('$IXC', '$SCHEMA.vectors', 'id', 'vec', NULL, NULL, 'l2', NULL, 'flat');
CALL vvector.set_index_options('$IXC', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '$ALT');
CALL vvector.refresh_index('$IXC');"
expect "vinfo without cache_dir finds the snapshot in that directory" "^every node: $ALT$" "$(vinfo_file "$ALT")"
expect "a search without cache_dir finds it too" "^rows: 3$" "$SEARCH_C"
if [ "$ECHO_ONLY" = no ]; then
    if [ -f "$ALT/$IXC/ACTIVE" ] && ! ls /tmp/vvector/$IXC/*.vv > /dev/null 2>&1 && grep -q "cache_dir=$ALT" /tmp/vvector/$IXC/OPTIONS; then
        echo "PASS  this node: the snapshot is in $ALT, the default directory holds only OPTIONS naming it"
    else
        echo "FAIL  this node: the snapshot is in $ALT, the default directory holds only OPTIONS naming it"
        ls -la "$ALT/$IXC" /tmp/vvector/$IXC 2>&1 | sed 's/^/      got: /' | head -12
        FAILED=$((FAILED + 1))
    fi
fi
expect "the stale check works through the redirect" "snapshot cache stale on .*: run vload" "
SELECT vvector.vsearch($Q, 999999999 USING PARAMETERS index_name='$IXC') OVER() FROM dual;"
expect "status names the directory" "node cache directory: $ALT (index option cache_dir)" "CALL vvector.status('$IXC');"
expect "a path with .. is refused" "cache_dir must be an absolute path of letters, digits and / . _ - without . or .. parts, or default" "
CALL vvector.set_index_options('$IXC', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '/tmp/../etc');"
expect "a directory that cannot be used is refused with vload's message" "vload: on .*cannot create directory /proc/vvector_no" "
CALL vvector.set_index_options('$IXC', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '/proc/vvector_no');"
expect "... and the option stays" "node cache directory: $ALT (index option cache_dir)" "CALL vvector.status('$IXC');"
expect "... and searches still work" "^rows: 3$" "$SEARCH_C"
expect "back to the default directory: loaded there at once" "^every node: /tmp/vvector$" "
CALL vvector.set_index_options('$IXC', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'default');
$(vinfo_file /tmp/vvector)"
expect "a search without cache_dir uses it" "^rows: 3$" "$SEARCH_C"
expect "unregister names the directory the cache files stay in" "Cache files under <cache_dir>/$IXC stay" "CALL vvector.unregister_index('$IXC');"
[ "$ECHO_ONLY" = yes ] || rm -rf "${ALT:?}/$IXC" "/tmp/vvector/$IXC"

run_sql "cleanup" "DELETE FROM vvector.snapshot WHERE index_name = 'vvtest'; COMMIT;" > /dev/null

finish_tests test_snapshot
