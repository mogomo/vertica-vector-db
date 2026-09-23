#!/usr/bin/env bash
# Integration test of HNSW indexes: build through refresh_index, vinfo, exact search on an HNSW
# index, recall of the precision levels, the journal overlay, thread independence, options.
#
#   tests/sql/test_hnsw.sh [--rows=N] [--schema=NAME] [--cache_dir=DIR] [--sift=SCHEMA] [--echo_only]
#
# On a journal of N random vectors of 16 elements (default 20000), registered as HNSW indexes
# vh_l2, vh_cos, vh_dot and vh_l1:
# - precision='exact' and exact=true give what the full scan gives (tests/sql/search_lib.sh:
#   same ids, scores within 1e-5, ranks), before and after 1000 journaled adds, 1000 deletes and
#   500 replacements without a refresh; so does an ef_search larger than the index (the graph
#   reaches every vector);
# - recall@10 of the precision levels fast, balanced and best against the full scan;
# - freshness='exact' never returns a deleted id; 1 and 8 threads give the same results;
# - vknn (no OVER(), one row in, k rows out) gives what vsearch gives with freshness snapshot;
# - set_index_options m and ef_construction take effect at the next refresh;
# - error messages of the HNSW parameters.
# --sift=SCHEMA: also registers the SIFT1M journal SCHEMA.sift_base (scripts/load_dataset.sh) as
# HNSW index sift_hnsw, refreshes it and asserts recall@10 >= 0.95 at precision balanced against
# SCHEMA.sift_gt for 1000 queries; prints the recall of every level. Takes a few minutes. The
# index is kept for scripts/benchmark.sh.
#
# Test data: schema VVHNSW (or --schema=NAME), dropped and recreated.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVHNSW
CACHE_DIR=/tmp/vvector
SIFT=
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --sift=*)      SIFT="${arg#--sift=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "test_hnsw.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh
IX_PREFIX=vh_
. tests/sql/search_lib.sh

search_sql() { echo "SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$1'$2) OVER() FROM $3"; }
Q1="(SELECT * FROM $SCHEMA.vh_l2_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 1) x"
BIG=$((ROWS * 2))

echo "== test data: journal of $ROWS vectors, four HNSW indexes"
run_sql "cleanup of an earlier run" "
CALL vvector.unregister_index('vh_l2'); CALL vvector.unregister_index('vh_cos');
CALL vvector.unregister_index('vh_dot'); CALL vvector.unregister_index('vh_l1');" > /dev/null
expect "journal and queries" "^journal $ROWS, queries 100$" "
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
COMMIT;
SELECT 'journal ' || (SELECT COUNT(*) FROM $SCHEMA.journal) || ', queries ' || (SELECT COUNT(*) FROM $SCHEMA.queries);"
for m in l2 cos dot l1; do
    metric=$m; [ "$m" = cos ] && metric=cosine
    expect "register (index_type defaults to hnsw) and refresh vh_$m ($metric)" "index vh_$m refreshed" "
CALL vvector.register_index('vh_$m', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', '$metric', NULL);
CALL vvector.refresh_index('vh_$m');"
done
expect "manifest and vinfo: hnsw with a graph" "^hnsw hnsw [1-9][0-9]* 16 200$" "
SELECT m.index_type || ' ' || i.index_type || ' ' || i.graph_bytes || ' ' || m.hnsw_m || ' ' || m.hnsw_ef_construction
FROM vvector.manifest m JOIN (SELECT * FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) v) i
     ON i.index_name = m.index_name WHERE m.index_name = 'vh_l2' LIMIT 1;"

echo "== exact search on an HNSW index"
for m in l2 cos dot l1; do compare "vh_$m: precision exact equals the full scan" "$m" queries20 10 exact ", precision='exact'"; done
compare "vh_l2: exact=true equals the full scan, k 100" l2 queries20 100 exact ", exact=true"
compare "vh_cos: ef_search larger than the index reaches every vector" cos queries20 10 exact ", ef_search=$BIG"

echo "== recall of the precision levels (random data, 16 dimensions)"
recall "vh_l2: precision fast (the default) recall@10 >= 0.90" l2 queries 10 "" 0.90
recall "vh_l2: precision balanced recall@10 >= 0.97" l2 queries 10 ", precision='balanced'" 0.97
recall "vh_l2: precision best recall@10 >= 0.995" l2 queries 10 ", precision='best'" 0.995
recall "vh_cos: precision balanced recall@10 >= 0.97" cos queries 10 ", precision='balanced'" 0.97
recall "vh_dot: precision balanced recall@10 >= 0.97" dot queries 10 ", precision='balanced'" 0.97
recall "vh_l1: precision balanced recall@10 >= 0.97" l1 queries 10 ", precision='balanced'" 0.97
recall "vh_l2: ef_search 400 wins over precision fast" l2 queries 10 ", precision='fast', ef_search=400" 0.995

echo "== 1000 adds, 1000 deletes, 500 replacements in the journal, no refresh"
expect "journal the changes" "^delta rows: 2500$" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id, TRUE FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 1000 + id, $VEC FROM ($(row_numbers 500)) g;
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM $SCHEMA.vh_l2_delta;"
for m in l2 dot; do compare "vh_$m: freshness exact, precision exact equals the full scan of the live rows" "$m" queries20 10 exact ", precision='exact'"; done
compare "vh_cos: freshness exact, ef_search larger than the index equals the full scan" cos queries20 10 exact ", ef_search=$BIG"
recall "vh_l2: freshness exact, precision balanced recall@10 >= 0.97 on the live rows" l2 queries 10 ", freshness='exact', precision='balanced'" 0.97
expect "freshness exact: no deleted id, even with k 2000" "^deleted in results: 0$" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vh_l2 ", k=2000, freshness='exact'" "$Q1")) r WHERE id <= 1000;"
expect "freshness exact: added ids are found (1000 of 21000 vectors, k 2000)" "^added in results: [1-9]" "
SELECT 'added in results: ' || COUNT(*) FROM ($(search_sql vh_l2 ", k=2000, freshness='exact'" "$Q1")) r WHERE id > 900000000;"
expect "the same results with 1 and with 8 threads (100 queries, journal applied)" "^differences: 0, rows: 1000$" "
SELECT 'differences: ' || (SELECT COUNT(*) FROM (
  (SELECT * FROM ($(search_sql vh_dot ", k=10, threads=1, freshness='exact'" "(SELECT * FROM $SCHEMA.vh_dot_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x")) a
   EXCEPT SELECT * FROM ($(search_sql vh_dot ", k=10, threads=8, freshness='exact'" "(SELECT * FROM $SCHEMA.vh_dot_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x")) b)) d)
  || ', rows: ' || (SELECT COUNT(*) FROM ($(search_sql vh_dot ", k=10, threads=8" "(SELECT * FROM $SCHEMA.vh_dot_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x")) c);"
expect "radius on HNSW: only rows within it" "^outside: 0$" "
SELECT 'outside: ' || COUNT(*) FROM ($(search_sql vh_l2 ", k=50, radius=1.0, freshness='exact'" "$Q1")) r WHERE score > 1.0;"

echo "== vknn: one vector in, k rows out, no OVER()"
SNAPQ="(SELECT * FROM $SCHEMA.vh_cos_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x"
expect "vknn gives what vsearch gives with freshness snapshot (100 queries, beside the qid column)" "^differences: 0, rows: 1000$" "
SELECT 'differences: ' || (SELECT COUNT(*) FROM (
    SELECT qid, id, rank, score::NUMERIC(20,6) FROM (SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vh_cos', k=10) FROM $SCHEMA.queries q) a
    EXCEPT SELECT qid, id, rank, score::NUMERIC(20,6) FROM ($(search_sql vh_cos ", k=10" "$SNAPQ")) b) d)
  || ', rows: ' || (SELECT COUNT(*) FROM (SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vh_cos', k=10) FROM $SCHEMA.queries q) c);"
expect "vknn with precision exact equals vsearch exact" "^differences: 0$" "
SELECT 'differences: ' || COUNT(*) FROM (
    SELECT qid, id, rank FROM (SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vh_cos', k=10, precision='exact') FROM $SCHEMA.queries20 q) a
    EXCEPT SELECT qid, id, rank FROM ($(search_sql vh_cos ", k=10, precision='exact'" "(SELECT * FROM $SCHEMA.vh_cos_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries20) x")) b) d;"
QTEXT=$(printf "SELECT TO_JSON(qvec) FROM %s.queries WHERE qid = 1;" "$SCHEMA" | { [ "$ECHO_ONLY" = yes ] && echo "[0.1]" || { printf '%s\n' "$PRE"; cat; } | vsql -X -A -t -q; })
expect "vknn with the query parameter FROM dual" "^rows 5, ranks 1 to 5$" "
SELECT 'rows ' || COUNT(*) || ', ranks ' || MIN(rank) || ' to ' || MAX(rank)
FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='vh_cos', k=5, query='$QTEXT') FROM dual) r;"
expect "vknn: a NULL vector gives no rows" "^rows: 0$" "
SELECT 'rows: ' || COUNT(*) FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='vh_cos') FROM dual) r;"
expect "vknn: a vector of another length is refused" "vknn: index 'vh_cos' has 16 dimensions, the query vector has 2" "
SELECT vvector.vknn(ARRAY[1.0, 2.0] USING PARAMETERS index_name='vh_cos') FROM dual;"

echo "== options and refresh"
expect "set_index_options m 8, ef_construction 40, then refresh: smaller graph, same exactness" "^m 8, efc 40, smaller: t$" "
CREATE LOCAL TEMP TABLE before ON COMMIT PRESERVE ROWS AS SELECT graph_bytes FROM vvector.manifest WHERE index_name = 'vh_l2';
CALL vvector.set_index_options('vh_l2', NULL, 8, 40, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
CALL vvector.refresh_index('vh_l2');
SELECT 'm ' || hnsw_m || ', efc ' || hnsw_ef_construction || ', smaller: ' || (graph_bytes < (SELECT graph_bytes FROM before))
FROM vvector.manifest WHERE index_name = 'vh_l2';"
compare "vh_l2 (m 8): precision exact equals the full scan after the refresh" l2 queries20 10 exact ", precision='exact'"
expect "set_index_options index_type flat, then refresh: no graph" "^flat 0$" "
CALL vvector.set_index_options('vh_l2', 'flat', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
CALL vvector.refresh_index('vh_l2');
SELECT index_type || ' ' || graph_bytes FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) v WHERE index_name = 'vh_l2' LIMIT 1;"

echo "== error messages"
expect "ef_search above 100000 is refused" "ef_search must be 0 (preset) to 100000" "$(search_sql vh_cos ", ef_search=100001" "$SCHEMA.vh_cos_snap");"
expect "set_index_options refuses m 1" "m must be 2 to 256" "CALL vvector.set_index_options('vh_cos', NULL, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
expect "vbuild refuses m 300" "vbuild: m must be 2 to 256" "
SELECT COUNT(*) FROM (SELECT vvector.vbuild(id, vec, FALSE USING PARAMETERS index_name='vh_err', index_type='hnsw', m=300) OVER()
FROM $SCHEMA.journal) b;"

if [ -n "$SIFT" ]; then
    echo "== SIFT1M ($SIFT.sift_base): HNSW index sift_hnsw, m 16, ef_construction 200"
    expect "register and refresh sift_hnsw" "index sift_hnsw refreshed" "
CALL vvector.unregister_index('sift_hnsw');
CALL vvector.register_index('sift_hnsw', '$SIFT.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw');
CALL vvector.refresh_index('sift_hnsw');"
    for p in fast balanced best; do
        run_sql "recall $p" "SELECT 'SIFT1M precision $p: recall@10 ' || (COUNT(*) / 10000.0)::NUMERIC(6,4) FROM
     (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10, precision='$p') OVER()
      FROM (SELECT * FROM $SIFT.sift_hnsw_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SIFT.sift_query WHERE qid < 1000) x) r
     JOIN $SIFT.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10;" | sed 's/^/      /'
    done
    expect "SIFT1M: recall@10 >= 0.95 at precision balanced (ef_search 100)" "^recall ok" "
SELECT CASE WHEN r >= 0.95 THEN 'recall ok ' ELSE 'recall too low ' END || r::NUMERIC(6,4) FROM (SELECT COUNT(*) / 10000.0 AS r FROM
     (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10, precision='balanced') OVER()
      FROM (SELECT * FROM $SIFT.sift_hnsw_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SIFT.sift_query WHERE qid < 1000) x) r
     JOIN $SIFT.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10) y;"
fi

run_sql "cleanup" "
CALL vvector.unregister_index('vh_l2'); CALL vvector.unregister_index('vh_cos');
CALL vvector.unregister_index('vh_dot'); CALL vvector.unregister_index('vh_l1');
DROP SCHEMA $SCHEMA CASCADE;" > /dev/null

finish_tests test_hnsw
