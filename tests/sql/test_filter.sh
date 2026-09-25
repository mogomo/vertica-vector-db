#!/usr/bin/env bash
# Integration test of filtered search (allow-list rows) and range search (radius) on HNSW.
#
#   tests/sql/test_filter.sh [--rows=N] [--schema=NAME] [--cache_dir=DIR] [--echo_only]
#
# On a journal of N random vectors of 16 elements (default 30000), registered as vf_l2 (HNSW),
# vf_cos (HNSW with sq8 codes), vf_dot (flat) and vf_l1 (HNSW):
# - allow-lists of 2 ids (under 0.01%), 1% and 50% of the ids: precision exact gives what the full
#   scan over the allowed rows gives; so does the default precision for 2 ids and 1% (the allowed rows
#   are searched exactly below max(10000, sqrt(64 x ef_search x vectors)) of them: 13,856 here); at
#   50% (the graph with the other positions masked) recall@10 >= 0.95;
# - with 500 journaled adds, 500 deletes and 200 replacements and freshness exact: journal rows count
#   only when their id is allowed; precision exact still equals the full scan over the allowed live rows;
# - an empty allow-list with filtered=true returns nothing; ids that are not in the index are ignored;
# - range search: radius with k 16384 on HNSW returns only vectors within the radius, with recall
#   >= 0.95 of the full scan within the radius; the same with vknn and with sq8 codes; with an
#   allow-list and a radius, exactly the allowed rows within the radius.
#
# Test data: schema VVFILTER (or --schema=NAME), dropped at the end.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=30000
DIMS=16
SCHEMA=VVFILTER
CACHE_DIR=/tmp/vvector
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "test_filter.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh
IX_PREFIX=vf_
. tests/sql/search_lib.sh

HALF=$((ROWS / 2))
Q1="(SELECT * FROM $SCHEMA.vf_l2_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 1"

echo "== test data: journal of $ROWS vectors, four indexes, three allow-lists"
run_sql "cleanup of an earlier run" "
CALL vvector.unregister_index('vf_l2'); CALL vvector.unregister_index('vf_cos');
CALL vvector.unregister_index('vf_dot'); CALL vvector.unregister_index('vf_l1');" > /dev/null
expect "journal, queries, allow-lists" "^journal $ROWS, queries 100, allowed 2 $((ROWS / 100)) $HALF$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.journal (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                              ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.journal (id, vec, ts) SELECT id, $VEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers "$ROWS")) g;
CREATE TABLE $SCHEMA.queries (qid INT, qvec ARRAY[FLOAT]);
INSERT INTO $SCHEMA.queries SELECT id, $VEC FROM ($(row_numbers 100)) g;
CREATE TABLE $SCHEMA.queries20 AS SELECT * FROM $SCHEMA.queries WHERE qid <= 20;
CREATE TABLE $SCHEMA.a_tiny AS SELECT id FROM $SCHEMA.journal WHERE id IN (7, $((ROWS - 3)));
CREATE TABLE $SCHEMA.a_pct AS SELECT id FROM $SCHEMA.journal WHERE id % 100 = 3;
CREATE TABLE $SCHEMA.a_half AS SELECT id FROM $SCHEMA.journal WHERE id % 2 = 0;
CREATE TABLE $SCHEMA.a_none (id INT);
COMMIT;
SELECT 'journal ' || (SELECT COUNT(*) FROM $SCHEMA.journal) || ', queries ' || (SELECT COUNT(*) FROM $SCHEMA.queries)
       || ', allowed ' || (SELECT COUNT(*) FROM $SCHEMA.a_tiny) || ' ' || (SELECT COUNT(*) FROM $SCHEMA.a_pct) || ' ' || (SELECT COUNT(*) FROM $SCHEMA.a_half);"
expect "register and refresh vf_l2 (hnsw), vf_cos (hnsw, sq8), vf_dot (flat), vf_l1 (hnsw)" "index vf_l1 refreshed" "
CALL vvector.register_index('vf_l2', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', NULL);
CALL vvector.register_index('vf_cos', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', NULL);
CALL vvector.set_index_options('vf_cos', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
CALL vvector.register_index('vf_dot', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'dot', NULL, 'flat');
CALL vvector.register_index('vf_l1', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l1', NULL);
CALL vvector.refresh_index('vf_l2'); CALL vvector.refresh_index('vf_cos'); CALL vvector.refresh_index('vf_dot');
CALL vvector.refresh_index('vf_l1');"

echo "== filtered search on the snapshot"
for a in a_tiny a_pct a_half; do
    ALLOW=$a
    for m in l2 cos dot l1; do compare "vf_$m, allow-list $a: precision exact equals the full scan over the allowed rows" "$m" queries20 10 snapshot ", precision='exact'"; done
done
for a in a_tiny a_pct; do
    ALLOW=$a
    for m in l2 cos l1; do compare "vf_$m, allow-list $a: the default precision searches the allowed rows exactly" "$m" queries20 10 snapshot; done
done
ALLOW=a_half
for m in l2 cos l1; do recall "vf_$m, allow-list a_half: the graph with the rest masked, recall@10 >= 0.95" "$m" queries 10 "" 0.95; done
ALLOW=a_pct
compare "vf_l2, allow-list a_pct, k 500: more than the allowed rows gives every allowed row" l2 queries20 500 snapshot
ALLOW=

expect "an empty allow-list with filtered=true returns nothing" "^rows 0$" "
SELECT 'rows ' || COUNT(*) FROM ($(search_sql vf_l2 ", filtered=true" "$Q1) x")) r;"
expect "without filtered=true, no allow-list row means no filter" "^rows 10$" "
SELECT 'rows ' || COUNT(*) FROM ($(search_sql vf_l2 "" "$Q1 UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.a_none) x")) r;"
expect "allow-list ids that are not in the index are ignored" "^rows 2, ids 7 $((ROWS - 3))$" "
SELECT 'rows ' || COUNT(*) || ', ids ' || MIN(id) || ' ' || MAX(id) FROM ($(search_sql vf_l2 ", filtered=true" \
    "$Q1 UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.a_tiny UNION ALL SELECT NULL, NULL, -5, NULL, NULL, NULL, NULL
     UNION ALL SELECT NULL, NULL, 999999999, NULL, NULL, NULL, NULL UNION ALL SELECT NULL, NULL, 7, NULL, NULL, NULL, NULL) x")) r;"
expect "the query parameter with an allow-list" "^rows 2$" "
SELECT 'rows ' || COUNT(*) FROM ($(search_sql vf_l2 ", query='[$(printf '0.1, %.0s' $(seq 15))0.1]'" \
    "(SELECT * FROM $SCHEMA.vf_l2_snap UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.a_tiny) x")) r;"

echo "== range search on HNSW (radius with k 16384)"
# The radius: the distance of the 30th neighbour of query 1 (l2), of the 30th best score (cosine).
R_L2=$(run_sql "radius l2" "SELECT d FROM (SELECT VECTOR_L2(l.vec, q.qvec) AS d FROM $SCHEMA.queries q CROSS JOIN ($LIVE) l
                             WHERE q.qid = 1 ORDER BY 1 LIMIT 30) x ORDER BY d DESC LIMIT 1;")
R_COS=$(run_sql "radius cosine" "SELECT d FROM (SELECT COSINE_SIMILARITY(l.vec, q.qvec) AS d FROM $SCHEMA.queries q CROSS JOIN ($LIVE) l
                              WHERE q.qid = 1 ORDER BY 1 DESC LIMIT 30) x ORDER BY d LIMIT 1;")
# The allow-list check below keeps 1% of the ids: within the 30th neighbour's radius the 20 queries
# have about six allowed rows in all, and none once in a while (seen on Eon, session 21), so it uses
# the 1000th neighbour's radius (some 200 allowed rows).
R_PCT=$(run_sql "radius l2 for the allow-list" "SELECT d FROM (SELECT VECTOR_L2(l.vec, q.qvec) AS d FROM $SCHEMA.queries q CROSS JOIN ($LIVE) l
                             WHERE q.qid = 1 ORDER BY 1 LIMIT 1000) x ORDER BY d DESC LIMIT 1;")
[ "$ECHO_ONLY" = yes ] && { R_L2=1.5; R_COS=0.7; R_PCT=2; }
# range NAME INDEX METRIC RADIUS SQL_OF_THE_SEARCH: every result within the radius (1e-5), and recall
# >= 0.95 against the full scan within the radius, for queries 1 to 20.
range() {
    local cmp="<=" sc
    sc=$(score_expr "$3")
    [ "$3" = cos ] && cmp=">="
    expect "$1" "^range ok" "
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS $5;
CREATE TABLE $SCHEMA.ref AS SELECT q.qid, l.id FROM $SCHEMA.queries20 q CROSS JOIN ($LIVE) l WHERE $sc $cmp $4;
SELECT CASE WHEN outside = 0 AND n_ref > 100 AND hit >= 0.95 * n_ref THEN 'range ok ' ELSE 'range bad ' END
       || 'results ' || n_got || ', within the radius ' || n_ref || ', found ' || hit || ', outside ' || outside FROM (
  SELECT (SELECT COUNT(*) FROM $SCHEMA.got) AS n_got, (SELECT COUNT(*) FROM $SCHEMA.ref) AS n_ref,
         (SELECT COUNT(*) FROM $SCHEMA.got g JOIN $SCHEMA.ref r ON g.qid = r.qid AND g.id = r.id) AS hit,
         (SELECT COUNT(*) FROM $SCHEMA.got WHERE NOT (score $cmp $4 $( [ "$3" = cos ] && echo "- 1e-5" || echo "+ 1e-5"))) AS outside) x;"
}
range "vf_l2: radius $R_L2 with k 16384 (balanced)" vf_l2 l2 "$R_L2" \
    "$(search_sql vf_l2 ", k=16384, radius=$R_L2" "(SELECT * FROM $SCHEMA.vf_l2_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries20) x")"
range "vf_l2: radius $R_L2 with k 16384 (fast)" vf_l2 l2 "$R_L2" \
    "$(search_sql vf_l2 ", k=16384, radius=$R_L2, precision='fast'" "(SELECT * FROM $SCHEMA.vf_l2_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries20) x")"
range "vf_cos (sq8): radius $R_COS with k 16384" vf_cos cos "$R_COS" \
    "$(search_sql vf_cos ", k=16384, radius=$R_COS" "(SELECT * FROM $SCHEMA.vf_cos_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries20) x")"
range "vknn on vf_l2: radius $R_L2 with k 16384" vf_l2 l2 "$R_L2" \
    "SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vf_l2', k=16384, radius=$R_L2) FROM $SCHEMA.queries20 q"

expect "vf_l2, allow-list a_pct with the radius: every allowed row within it, nothing else" "^filtered range ok" "
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS $(search_sql vf_l2 ", k=16384, radius=$R_PCT" "(SELECT * FROM $SCHEMA.vf_l2_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries20
    UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.a_pct) x");
CREATE TABLE $SCHEMA.ref AS SELECT q.qid, l.id, VECTOR_L2(l.vec, q.qvec) AS d FROM $SCHEMA.queries20 q CROSS JOIN ($LIVE) l
    WHERE l.id IN (SELECT id FROM $SCHEMA.a_pct) AND VECTOR_L2(l.vec, q.qvec) <= $R_PCT + 1e-5;
SELECT CASE WHEN missing = 0 AND foreign_ids = 0 AND n_ref > 0 THEN 'filtered range ok ' ELSE 'filtered range bad ' END
       || 'results ' || n_got || ', allowed within the radius ' || n_ref || ', missing ' || missing || ', not allowed or outside ' || foreign_ids FROM (
  SELECT (SELECT COUNT(*) FROM $SCHEMA.got) AS n_got, (SELECT COUNT(*) FROM $SCHEMA.ref WHERE d <= $R_PCT - 1e-5) AS n_ref,
         (SELECT COUNT(*) FROM $SCHEMA.ref r WHERE r.d <= $R_PCT - 1e-5 AND NOT EXISTS (SELECT 1 FROM $SCHEMA.got g WHERE g.qid = r.qid AND g.id = r.id)) AS missing,
         (SELECT COUNT(*) FROM $SCHEMA.got g WHERE NOT EXISTS (SELECT 1 FROM $SCHEMA.ref r WHERE r.qid = g.qid AND r.id = g.id)) AS foreign_ids) x;"

echo "== filtered search with the journal (500 adds, 500 deletes, 200 replacements, no refresh)"
expect "journal the changes; the new ids 900000002, 900000004, ... join allow-list a_half" "^delta rows: 1200$" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 500)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id * 2, TRUE FROM ($(row_numbers 500)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 2000 + id, $VEC FROM ($(row_numbers 200)) g;
INSERT INTO $SCHEMA.a_half SELECT 900000000 + id FROM ($(row_numbers 500)) g WHERE id % 2 = 0;
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM $SCHEMA.vf_l2_delta;"
for a in a_pct a_half; do
    ALLOW=$a
    for m in l2 cos dot; do compare "vf_$m, allow-list $a, freshness exact: precision exact equals the full scan over the allowed live rows" "$m" queries20 10 exact ", precision='exact'"; done
done
ALLOW=a_pct
compare "vf_l1, allow-list a_pct, freshness exact: the default precision is exact" l1 queries20 10 exact
ALLOW=a_half
recall "vf_l2, allow-list a_half, freshness exact: recall@10 >= 0.95" l2 queries 10 ", freshness='exact'" 0.95
ALLOW=
expect "journal adds outside the allow-list never come back" "^outside: 0$" "
SELECT 'outside: ' || COUNT(*) FROM ($(search_sql vf_l2 ", k=100, freshness='exact'" \
    "(SELECT * FROM $SCHEMA.vf_l2_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid <= 5
      UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.a_pct) x")) r WHERE id NOT IN (SELECT id FROM $SCHEMA.a_pct);"

run_sql "cleanup" "
CALL vvector.unregister_index('vf_l2'); CALL vvector.unregister_index('vf_cos'); CALL vvector.unregister_index('vf_dot');
CALL vvector.unregister_index('vf_l1');
DROP SCHEMA $SCHEMA CASCADE;" > /dev/null

finish_tests test_filter
