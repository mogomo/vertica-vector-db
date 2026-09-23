#!/usr/bin/env bash
# Integration test of vsearch: exact results against Vertica's own functions.
#
#   tests/sql/test_search.sh [--rows=N] [--schema=NAME] [--cache_dir=DIR] [--echo_only]
#
# For the metrics l2, cosine and dot, vsearch must return what
#   ORDER BY VECTOR_L2 | COSINE_SIMILARITY | DOT_PRODUCT (vec, q) LIMIT k
# returns over the live rows of the journal (latest row per id, not deleted); for l1 the same with
# the SQL expression SUM(ABS(vec[i] - q[i])). Compared: the ids (a different id is allowed only
# where scores tie at the k-th place), the scores (1e-5 relative, at least 1e-5 absolute), and the
# ranks (a different rank only between scores that tie within that tolerance).
# Checked before and after 1000 journaled adds, 1000 deletes and 500 replacements without a
# refresh, with 20 queries, a batch of 1000 queries, the query parameter, radius, k > rows,
# thread counts, parameter precedence (function, session, index default), and the error messages.
#
# Test data: schema VVSEARCH (or --schema=NAME), dropped and recreated: a journal of N random
# vectors of 16 elements (default 20000), registered as flat indexes vs_l2, vs_cos, vs_dot and
# vs_l1 (tests/sql/test_hnsw.sh does the same for HNSW indexes).
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVSEARCH
CACHE_DIR=/tmp/vvector
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "test_search.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh

IX_PREFIX=vs_
. tests/sql/search_lib.sh

Q1="(SELECT * FROM $SCHEMA.vs_l2_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 1) x"

echo "== test data: journal of $ROWS vectors, four indexes"
run_sql "cleanup of an earlier run" "
CALL vvector.unregister_index('vs_l2'); CALL vvector.unregister_index('vs_cos');
CALL vvector.unregister_index('vs_dot'); CALL vvector.unregister_index('vs_l1'); CALL vvector.unregister_index('vs_small');" > /dev/null
expect "journal, queries, small table" "^journal $ROWS, queries 20, batch 1000$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.journal (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                              ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.journal (id, vec, ts) SELECT id, $VEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers "$ROWS")) g;
CREATE TABLE $SCHEMA.queries (qid INT, qvec ARRAY[FLOAT]);
INSERT INTO $SCHEMA.queries SELECT id, $VEC FROM ($(row_numbers 20)) g;
CREATE TABLE $SCHEMA.batch (qid INT, qvec ARRAY[FLOAT]);
INSERT INTO $SCHEMA.batch SELECT id, $VEC FROM ($(row_numbers 1000)) g;
CREATE TABLE $SCHEMA.batch_sample AS SELECT * FROM $SCHEMA.batch WHERE qid % 10 = 3;
CREATE TABLE $SCHEMA.small (id INT, vec ARRAY[FLOAT]);
INSERT INTO $SCHEMA.small SELECT id, $VEC FROM ($(row_numbers 7)) g;
COMMIT;
SELECT 'journal ' || (SELECT COUNT(*) FROM $SCHEMA.journal) || ', queries ' || (SELECT COUNT(*) FROM $SCHEMA.queries)
       || ', batch ' || (SELECT COUNT(*) FROM $SCHEMA.batch);"
for m in l2 cos dot l1; do
    metric=$m; [ "$m" = cos ] && metric=cosine
    expect "register and refresh vs_$m ($metric)" "index vs_$m refreshed" "
CALL vvector.register_index('vs_$m', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', '$metric', NULL, 'flat');
CALL vvector.refresh_index('vs_$m');"
done

echo "== exact results on the snapshot"
for m in l2 cos dot l1; do compare "vs_$m: 20 queries, k 10, equal to the full scan" "$m" queries 10 exact; done
compare "vs_l2: k 100" l2 queries 100 exact
compare "vs_cos: k 1" cos queries 1 exact

echo "== 1000 adds, 1000 deletes, 500 replacements in the journal, no refresh"
expect "journal the changes" "^delta rows: 2500$" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id, TRUE FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 1000 + id, $VEC FROM ($(row_numbers 500)) g;
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM $SCHEMA.vs_l2_delta;"
for m in l2 cos dot l1; do compare "vs_$m: freshness exact equals the full scan of the live rows" "$m" queries 10 exact; done
compare "vs_dot: k 50 with the journal" dot queries 50 exact
expect "freshness snapshot ignores the journal: deleted ids come back" "^deleted in results: [1-9]" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000, freshness='snapshot'" "$Q1")) r WHERE id <= 1000;"
expect "freshness exact: no deleted id" "^deleted in results: 0$" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000, freshness='exact'" "$Q1")) r WHERE id <= 1000;"

echo "== many queries in one statement"
compare "vs_l2: a batch of 1000 queries (100 of them checked) equals the full scan" l2 batch 10 exact "" batch_sample
compare "vs_cos: a batch of 1000 queries, 1 thread (100 checked)" cos batch 10 exact ", threads=1" batch_sample
expect "a batch of 1000 queries gives 10 rows for each" "^rows 10000, queries 1000$" "
SELECT 'rows ' || COUNT(*) || ', queries ' || COUNT(DISTINCT qid) FROM $SCHEMA.got;"
expect "the same results with 1 and with 8 threads" "^differences: 0$" "
SELECT 'differences: ' || COUNT(*) FROM (
  (SELECT * FROM ($(search_sql vs_dot ", k=20, threads=1, freshness='exact'" "(SELECT * FROM $SCHEMA.vs_dot_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.batch) x")) a
   EXCEPT SELECT * FROM ($(search_sql vs_dot ", k=20, threads=8, freshness='exact'" "(SELECT * FROM $SCHEMA.vs_dot_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.batch) x")) b)
) d;"

echo "== the query parameter, the _snap view, radius, k"
QTEXT=$(printf "SELECT TO_JSON(qvec) FROM %s.queries WHERE qid = 1;" "$SCHEMA" | { [ "$ECHO_ONLY" = yes ] && echo "[0.1]" || { printf '%s\n' "$PRE"; cat; } | vsql -X -A -t -q; })
expect "query parameter gives the same ids and scores as the query row (snapshot only, _snap view)" "^differences: 0, rows: 10$" "
SELECT 'differences: ' || (SELECT COUNT(*) FROM (
    SELECT id, rank, score::NUMERIC(20,6) FROM ($(search_sql vs_l2 ", k=10, query='$QTEXT'" "$SCHEMA.vs_l2_snap")) a
    EXCEPT SELECT id, rank, score::NUMERIC(20,6) FROM ($(search_sql vs_l2 ", k=10" "$Q1")) b) d)
  || ', rows: ' || (SELECT COUNT(*) FROM ($(search_sql vs_l2 ", k=10, query='$QTEXT'" "$SCHEMA.vs_l2_snap")) c);"
expect "the query parameter has qid 0" "^qids: 0$" "
SELECT 'qids: ' || MAX(qid) FROM ($(search_sql vs_l2 ", k=3, query='$QTEXT'" "$SCHEMA.vs_l2_snap")) r;"
# Halfway between the 5th and the 6th distance: exactly 5 rows are within it.
RADIUS=$(printf "SELECT (MAX(CASE WHEN rank = 5 THEN score END) + MAX(CASE WHEN rank = 6 THEN score END)) / 2 FROM (%s) r;" "$(search_sql vs_l2 ", k=6, freshness='exact'" "$Q1")" | { [ "$ECHO_ONLY" = yes ] && echo 1 || { printf '%s\n' "$PRE"; cat; } | vsql -X -A -t -q; })
expect "radius: only rows within it (l2: score <= radius), at most k" "^rows 5, outside 0$" "
SELECT 'rows ' || COUNT(*) || ', outside ' || SUM(CASE WHEN score > $RADIUS THEN 1 ELSE 0 END)
FROM ($(search_sql vs_l2 ", k=10, freshness='exact', radius=$RADIUS" "$Q1")) r;"
expect "radius on cosine: score >= radius" "^rows 0 to 50, below 0$" "
SELECT 'rows ' || CASE WHEN COUNT(*) <= 50 THEN '0 to 50' ELSE COUNT(*)::VARCHAR END || ', below ' || SUM(CASE WHEN score < 0.5 THEN 1 ELSE 0 END)
FROM ($(search_sql vs_cos ", k=50, radius=0.5, freshness='exact'" "(SELECT * FROM $SCHEMA.vs_cos_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 2) x")) r;"
expect "k larger than the index: every vector once" "^rows 7, ranks 1 to 7$" "
CALL vvector.register_index('vs_small', '$SCHEMA.small', 'id', 'vec', NULL, NULL, 'l2', NULL);
CALL vvector.refresh_index('vs_small');
SELECT 'rows ' || COUNT(DISTINCT id) || ', ranks ' || MIN(rank) || ' to ' || MAX(rank)
FROM ($(search_sql vs_small ", k=20, query='$QTEXT'" "$SCHEMA.vs_small_snap")) r;"
expect "a static index has only the _snap view" "^views: 1$" "
SELECT 'views: ' || COUNT(*) FROM v_catalog.views WHERE LOWER(table_schema) = LOWER('$SCHEMA') AND LOWER(table_name) LIKE 'vs\\_small\\_%';"

echo "== parameter precedence: function, then session, then index default, then built-in"
expect "session parameter k" "^rows: 3$" "
ALTER SESSION SET UDPARAMETER FOR vvector k = '3';
SELECT 'rows: ' || COUNT(*) FROM ($(search_sql vs_l2 "" "$Q1")) r;"
expect "function parameter k wins over the session" "^rows: 4$" "
ALTER SESSION SET UDPARAMETER FOR vvector k = '3';
SELECT 'rows: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=4" "$Q1")) r;"
expect "built-in k is 10" "^rows: 10$" "SELECT 'rows: ' || COUNT(*) FROM ($(search_sql vs_l2 "" "$Q1")) r;"
run_sql "set_index_options freshness_default exact" "CALL vvector.set_index_options('vs_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'exact', NULL, NULL);" > /dev/null
wait_cache_check
expect "index default freshness exact (set_index_options) applies the journal" "^deleted in results: 0$" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000" "$Q1")) r WHERE id <= 1000;"
expect "session parameter freshness snapshot wins over the index default" "^deleted in results: [1-9]" "
ALTER SESSION SET UDPARAMETER FOR vvector freshness = 'snapshot';
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000" "$Q1")) r WHERE id <= 1000;"
expect "function parameter wins over the session" "^deleted in results: 0$" "
ALTER SESSION SET UDPARAMETER FOR vvector freshness = 'snapshot';
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000, freshness='exact'" "$Q1")) r WHERE id <= 1000;"
run_sql "set_index_options freshness_default default" "CALL vvector.set_index_options('vs_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'default', NULL, NULL);" > /dev/null
wait_cache_check
expect "index default back to the built-in (snapshot)" "^deleted in results: [1-9]" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vs_l2 ", k=2000" "$Q1")) r WHERE id <= 1000;"
expect "precision levels are accepted; on a flat index all are exact" "^differences: 0$" "
SELECT 'differences: ' || COUNT(*) FROM (
  SELECT id, rank FROM ($(search_sql vs_l2 ", precision='fast', ef_search=50" "$Q1")) a
  EXCEPT SELECT id, rank FROM ($(search_sql vs_l2 ", precision='exact', exact=true" "$Q1")) b) d;"
expect "a bad session value names its source" "session parameter threads = 'many' is not an integer" "
ALTER SESSION SET UDPARAMETER FOR vvector threads = 'many';
$(search_sql vs_l2 "" "$Q1");"

echo "== error messages"
expect "k 0 is refused" "k must be 1 to 16384, not 0" "$(search_sql vs_l2 ", k=0" "$Q1");"
expect "k 16385 is refused" "k must be 1 to 16384, not 16385" "$(search_sql vs_l2 ", k=16385" "$Q1");"
expect "an unknown precision is refused" "precision must be fast, balanced, best or exact, not 'high'" "$(search_sql vs_l2 ", precision='high'" "$Q1");"
expect "an unknown freshness is refused" "freshness must be snapshot or exact, not 'now'" "$(search_sql vs_l2 ", freshness='now'" "$Q1");"
expect "threads 65 is refused" "threads must be 0 (one per core) to 64" "$(search_sql vs_l2 ", threads=65" "$Q1");"
expect "oversampling 0.5 is refused" "oversampling must be 1 to 100" "$(search_sql vs_l2 ", oversampling=0.5" "$Q1");"
expect "a malformed query parameter is refused" "query vector text: expected ',' or ']' at character 6" "$(search_sql vs_l2 ", query='[1.0 2.0]'" "$SCHEMA.vs_l2_snap");"
expect "a query parameter of another length is refused" "index 'vs_l2' has 16 dimensions, the query parameter has 2" "$(search_sql vs_l2 ", query='[1, 2]'" "$SCHEMA.vs_l2_snap");"
expect "no query at all is refused" "no query: give the query parameter, or query rows (qid, qvec)" "$(search_sql vs_l2 "" "$SCHEMA.vs_l2_snap");"
expect "a query row without qid is refused" "a query row has a vector (qvec) but no qid" "
$(search_sql vs_l2 "" "(SELECT NULL::INT AS qid, qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, NULL::BOOLEAN AS del, NULL::INT AS ver, NULL::INT AS snapshot_id FROM $SCHEMA.queries WHERE qid = 1) x");"
expect "an allow-list row says when filtered search comes" "allow-list rows (filtered search) are not implemented yet (milestone M5)" "
$(search_sql vs_l2 "" "(SELECT qid, qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, NULL::BOOLEAN AS del, NULL::INT AS ver, NULL::INT AS snapshot_id FROM $SCHEMA.queries WHERE qid = 1 UNION ALL SELECT NULL, NULL, 5, NULL, NULL, NULL, NULL) x");"
expect "a journal vector of another length is refused" "the journal vector of id 5 has 2" "
$(search_sql vs_l2 ", freshness='exact'" "(SELECT qid, qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, NULL::BOOLEAN AS del, NULL::INT AS ver, NULL::INT AS snapshot_id FROM $SCHEMA.queries WHERE qid = 1 UNION ALL SELECT NULL, NULL, 5, ARRAY[1.0, 2.0], FALSE, 1, NULL) x");"

run_sql "cleanup" "
CALL vvector.unregister_index('vs_l2'); CALL vvector.unregister_index('vs_cos'); CALL vvector.unregister_index('vs_dot');
CALL vvector.unregister_index('vs_l1'); CALL vvector.unregister_index('vs_small');
DROP SCHEMA $SCHEMA CASCADE;" > /dev/null

finish_tests test_search
