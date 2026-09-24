#!/usr/bin/env bash
# Integration test of int8 quantisation (quantization sq8): build through refresh_index, vinfo, exact
# results through rescoring, recall of the precision levels, scores without rescoring, the journal
# overlay, incremental refresh, memory_mode compact, and the errors.
#
#   tests/sql/test_sq8.sh [--rows=N] [--schema=NAME] [--cache_dir=DIR] [--sift=SCHEMA] [--echo_only]
#
# On a journal of N random vectors of 16 elements (default 20000), registered as HNSW indexes with
# sq8 codes vq_l2, vq_cos, vq_dot, vq_l1 and a flat index with sq8 codes vqf_l2:
# - precision='exact' gives what the full scan gives (tests/sql/search_lib.sh: same ids, scores
#   within 1e-5, ranks); so does a search that rescores the 100 x k best candidates by their codes
#   (the graph walk sees every vector): the rescored scores are the float scores (the engine test
#   proves it with every vector a candidate);
# - recall@10 of fast (codes only), balanced (the default, 2 x k rescored) and best (4 x k);
# - rescore=false returns the approximate scores, close to the exact ones;
# - the same after 1000 journaled adds, 1000 deletes and 500 replacements without a refresh, and
#   after the incremental refresh that follows (which keeps the codes);
# - vknn gives what vsearch gives; 1 and 8 threads give the same results;
# - memory_mode compact needs sq8; quantization none again is a full build; vbuild refuses a base
#   snapshot whose quantization differs.
# --sift=SCHEMA: also registers the SIFT1M journal SCHEMA.sift_base (scripts/load_dataset.sh) as an
# HNSW index with sq8 codes, sift_sq8, and asserts recall@10 >= 0.97 at precision balanced against
# SCHEMA.sift_gt for 1000 queries (the float index has 0.98); prints the recall of every level. The
# index is kept for scripts/benchmark.sh.
#
# Test data: schema VVSQ8 (or --schema=NAME), dropped and recreated.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVSQ8
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
        -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "test_sq8.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh
IX_PREFIX=vq_
. tests/sql/search_lib.sh

# 1000 of the 20000 vectors rescored, the 1000 best by their codes: on HNSW an ef_search above the
# number of vectors makes the graph walk see every vector (as in test_hnsw.sh), so only the codes
# choose the candidates; the true 10 nearest are far inside that list.
WIDE=", ef_search=$((ROWS * 2)), oversampling=100"
SQ8="NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL"
Q1="(SELECT * FROM $SCHEMA.vq_l2_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 1) x"
QALL() { echo "(SELECT * FROM $SCHEMA.$1_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x"; }

echo "== test data: journal of $ROWS vectors, four HNSW indexes and a flat index with sq8 codes"
run_sql "cleanup of an earlier run" "
CALL vvector.unregister_index('vq_l2'); CALL vvector.unregister_index('vq_cos'); CALL vvector.unregister_index('vq_dot');
CALL vvector.unregister_index('vq_l1'); CALL vvector.unregister_index('vqf_l2');" > /dev/null
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
    expect "register vq_$m ($metric, hnsw), quantization sq8, refresh" "index vq_$m refreshed" "
CALL vvector.register_index('vq_$m', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', '$metric', 0, 'hnsw');
CALL vvector.set_index_options('vq_$m', $SQ8);
CALL vvector.refresh_index('vq_$m');"
done
expect "register vqf_l2 (l2, flat), quantization sq8, refresh" "index vqf_l2 refreshed" "
CALL vvector.register_index('vqf_l2', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', 0, 'flat');
CALL vvector.set_index_options('vqf_l2', $SQ8);
CALL vvector.refresh_index('vqf_l2');"
expect "manifest and vinfo on every node: sq8" "^sq8 on [1-9][0-9]* of [1-9][0-9]* nodes, same: t$" "
SELECT q || ' on ' || n || ' of ' || up || ' nodes, same: ' || (n = up) FROM
    (SELECT MAX(m.quantization) AS q, COUNT(DISTINCT i.node_name) AS n
     FROM vvector.manifest m JOIN (SELECT * FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) v) i
          ON i.index_name = m.index_name AND i.quantization = 'sq8' WHERE m.index_name = 'vq_l2') a
    CROSS JOIN (SELECT COUNT(*) AS up FROM v_catalog.nodes WHERE node_state = 'UP') b;"

echo "== exact results"
for m in l2 cos dot l1; do compare "vq_$m: precision exact equals the full scan" "$m" queries20 10 exact ", precision='exact'"; done
for m in l2 cos dot l1; do
    compare "vq_$m: 1000 candidates rescored: the results are the full scan" "$m" queries20 10 exact "$WIDE"
done
IX_PREFIX=vqf_
compare "vqf_l2 (flat): 1000 candidates rescored: the results are the full scan" l2 queries20 10 exact "$WIDE"
IX_PREFIX=vq_

echo "== recall of the precision levels (random data, 16 dimensions)"
recall "vq_l2: precision fast (codes only) recall@10 >= 0.80" l2 queries 10 ", precision='fast'" 0.80
recall "vq_l2: precision balanced (the default) recall@10 >= 0.95" l2 queries 10 "" 0.95
recall "vq_l2: precision best recall@10 >= 0.98" l2 queries 10 ", precision='best'" 0.98
recall "vq_cos: precision balanced recall@10 >= 0.95" cos queries 10 ", precision='balanced'" 0.95
recall "vq_dot: precision balanced recall@10 >= 0.95" dot queries 10 ", precision='balanced'" 0.95
recall "vq_l1: precision balanced recall@10 >= 0.95" l1 queries 10 ", precision='balanced'" 0.95
IX_PREFIX=vqf_
recall "vqf_l2 (flat): precision balanced recall@10 >= 0.95" l2 queries 10 "" 0.95
IX_PREFIX=vq_
expect "rescore=false: approximate scores, within 0.05 of VECTOR_L2 (scale x 4 x 1.6)" "^off by at most 0.0[0-4]" "
SELECT 'off by at most ' || MAX(ABS(r.score - VECTOR_L2(j.vec, q.qvec)))::NUMERIC(8,5) || ', some differ: ' || (MAX(ABS(r.score - VECTOR_L2(j.vec, q.qvec))) > 0)
FROM ($(search_sql vq_l2 ", k=10, rescore=false" "$(QALL vq_l2)")) r
JOIN $SCHEMA.queries q ON q.qid = r.qid JOIN $SCHEMA.journal j ON j.id = r.id;"
expect "rescore=true (the default): the scores are VECTOR_L2's" "^off by at most 0.00000" "
SELECT 'off by at most ' || MAX(ABS(r.score - VECTOR_L2(j.vec, q.qvec)))::NUMERIC(8,5)
FROM ($(search_sql vq_l2 ", k=10" "$(QALL vq_l2)")) r
JOIN $SCHEMA.queries q ON q.qid = r.qid JOIN $SCHEMA.journal j ON j.id = r.id;"

echo "== 1000 adds, 1000 deletes, 500 replacements in the journal, no refresh"
expect "journal the changes" "^delta rows: 2500$" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id, TRUE FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 1000 + id, $VEC FROM ($(row_numbers 500)) g;
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM $SCHEMA.vq_l2_delta;"
for m in l2 dot; do
    compare "vq_$m: freshness exact, 1000 candidates rescored equal the full scan of the live rows" "$m" queries20 10 exact "$WIDE"
done
recall "vq_l2: freshness exact, precision balanced recall@10 >= 0.95 on the live rows" l2 queries 10 ", freshness='exact'" 0.95
expect "freshness exact: no deleted id, even with k 2000" "^deleted in results: 0$" "
SELECT 'deleted in results: ' || COUNT(*) FROM ($(search_sql vq_l2 ", k=2000, freshness='exact'" "$Q1")) r WHERE id <= 1000;"

echo "== incremental refresh"
for m in l2 cos; do
    expect "vq_$m: the refresh is incremental and keeps the codes" "^incremental sq8 sq8$" "
CALL vvector.refresh_index('vq_$m');
SELECT CASE WHEN m.refresh_note LIKE '%incremental from snapshot%' THEN 'incremental' ELSE m.refresh_note END || ' ' || m.quantization || ' ' || i.quantization
FROM vvector.manifest m JOIN (SELECT * FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) v) i
     ON i.index_name = m.index_name WHERE m.index_name = 'vq_$m' LIMIT 1;"
    compare "vq_$m: after the incremental refresh, 1000 candidates rescored equal the full scan" "$m" queries20 10 exact "$WIDE"
done
recall "vq_l2: after the incremental refresh, precision balanced recall@10 >= 0.95" l2 queries 10 "" 0.95

echo "== vknn and threads"
SNAPQ="(SELECT * FROM $SCHEMA.vq_cos_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries) x"
expect "vknn gives what vsearch gives with freshness snapshot (100 queries)" "^differences: 0, rows: 1000$" "
SELECT 'differences: ' || (SELECT COUNT(*) FROM (
    SELECT qid, id, rank, score::NUMERIC(20,6) FROM (SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vq_cos', k=10) FROM $SCHEMA.queries q) a
    EXCEPT SELECT qid, id, rank, score::NUMERIC(20,6) FROM ($(search_sql vq_cos ", k=10" "$SNAPQ")) b) d)
  || ', rows: ' || (SELECT COUNT(*) FROM (SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='vq_cos', k=10) FROM $SCHEMA.queries q) c);"
for p in "" ", precision='fast'"; do
    expect "the same results with 1 and with 8 threads${p:+ (fast)}" "^differences: 0, rows: 1000$" "
SELECT 'differences: ' || (SELECT COUNT(*) FROM (
  (SELECT * FROM ($(search_sql vq_dot ", k=10, threads=1, freshness='exact'$p" "$(QALL vq_dot)")) a
   EXCEPT SELECT * FROM ($(search_sql vq_dot ", k=10, threads=8, freshness='exact'$p" "$(QALL vq_dot)")) b)) d)
  || ', rows: ' || (SELECT COUNT(*) FROM ($(search_sql vq_dot ", k=10, threads=8$p" "$(QALL vq_dot)")) c);"
done

echo "== memory_mode, options and errors"
expect "memory_mode compact on an index with sq8, then a refresh: searches work" "^rows: 10$" "
CALL vvector.set_index_options('vq_l1', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'compact', NULL, NULL, NULL, NULL);
INSERT INTO $SCHEMA.journal (id, vec) SELECT 800000000 + id, $VEC FROM ($(row_numbers 10)) g; COMMIT;
CALL vvector.refresh_index('vq_l1');
SELECT 'rows: ' || COUNT(*) FROM ($(search_sql vq_l1 ", k=10" "(SELECT * FROM $SCHEMA.vq_l1_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.queries WHERE qid = 3) x")) r;"
expect "memory_mode compact without sq8 is refused" "memory_mode compact needs quantization sq8" "
CALL vvector.set_index_options('vq_l1', NULL, NULL, NULL, 'none', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
expect "quantization none (and memory_mode ram): the next refresh is a full build without codes" "^full build none$" "
CALL vvector.set_index_options('vq_l1', NULL, NULL, NULL, 'none', NULL, NULL, NULL, 'ram', NULL, NULL, NULL, NULL);
CALL vvector.refresh_index('vq_l1');
SELECT CASE WHEN m.refresh_note LIKE '%full build%' THEN 'full build' ELSE m.refresh_note END || ' ' || i.quantization
FROM vvector.manifest m JOIN (SELECT * FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) v) i
     ON i.index_name = m.index_name WHERE m.index_name = 'vq_l1' LIMIT 1;"
compare "vq_l1 without codes: precision exact equals the full scan" l1 queries20 10 exact ", precision='exact'"
BASE=$( [ "$ECHO_ONLY" = yes ] && echo 1 || { printf '%s\n' "$PRE" "SELECT active_snapshot FROM vvector.manifest WHERE index_name = 'vq_l2';" | vsql -X -A -t -q; })
expect "vbuild refuses a base snapshot with other codes" "the base snapshot has quantization sq8, not none: refresh with mode full" "
SELECT COUNT(*) FROM (SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='vq_l2', metric='l2', index_type='hnsw',
    quantization='none', base_snapshot=$BASE) OVER() FROM $SCHEMA.journal WHERE id = 5) b;"
expect "set_index_options refuses quantization pq" "quantization must be none or sq8" "
CALL vvector.set_index_options('vq_l2', NULL, NULL, NULL, 'pq', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
expect "oversampling below 1 is refused" "oversampling must be 1 to 100" "$(search_sql vq_l2 ", oversampling=0.5" "$SCHEMA.vq_l2_snap");"

if [ -n "$SIFT" ]; then
    echo "== SIFT1M ($SIFT.sift_base): HNSW index with sq8 codes sift_sq8, m 16, ef_construction 200"
    expect "register and refresh sift_sq8" "index sift_sq8 refreshed" "
CALL vvector.unregister_index('sift_sq8');
CALL vvector.register_index('sift_sq8', '$SIFT.sift_base', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');
CALL vvector.set_index_options('sift_sq8', $SQ8);
CALL vvector.refresh_index('sift_sq8');"
    for p in fast balanced best; do
        run_sql "recall $p" "SELECT 'SIFT1M sq8 precision $p: recall@10 ' || (COUNT(*) / 10000.0)::NUMERIC(6,4) FROM
     (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_sq8', k=10, precision='$p') OVER()
      FROM (SELECT * FROM $SIFT.sift_sq8_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SIFT.sift_query WHERE qid < 1000) x) r
     JOIN $SIFT.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10;" | sed 's/^/      /'
    done
    expect "SIFT1M sq8: recall@10 >= 0.97 at precision balanced (float: 0.98)" "^recall ok" "
SELECT CASE WHEN r >= 0.97 THEN 'recall ok ' ELSE 'recall too low ' END || r::NUMERIC(6,4) FROM (SELECT COUNT(*) / 10000.0 AS r FROM
     (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_sq8', k=10, precision='balanced') OVER()
      FROM (SELECT * FROM $SIFT.sift_sq8_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SIFT.sift_query WHERE qid < 1000) x) r
     JOIN $SIFT.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10) y;"
fi

run_sql "cleanup" "
CALL vvector.unregister_index('vq_l2'); CALL vvector.unregister_index('vq_cos'); CALL vvector.unregister_index('vq_dot');
CALL vvector.unregister_index('vq_l1'); CALL vvector.unregister_index('vqf_l2');
DROP SCHEMA $SCHEMA CASCADE;" > /dev/null

finish_tests test_sq8
