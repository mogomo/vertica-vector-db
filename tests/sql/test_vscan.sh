#!/usr/bin/env bash
# Integration test of vscan: exact search over a table without an index (milestone M7).
#
#   tests/sql/test_vscan.sh [--rows=N] [--schema=NAME] [--echo_only]
#
# On a table of N random vectors of 16 elements (default 20000) with 20 queries, vscan merged by
# the SQL around it must return what ORDER BY VECTOR_L2 | COSINE_SIMILARITY | DOT_PRODUCT (vec, q)
# LIMIT k returns, and for l1 the SQL expression SUM(ABS(vec[i] - q[i])) (compared as
# test_search.sh: ids, scores within 1e-5 relative, ranks; ties only where scores tie). Checked
# with OVER(PARTITION BEST) and OVER(), several queries in the `queries` parameter (a LONG VARCHAR:
# 300 queries), one query in `query`, a WHERE filter, radius, k above the rows, NULL ids and
# vectors (skipped), equal vectors under two ids (both returned, by id), an empty input, and the
# error messages (a wrong number of elements, a bad metric, k, no query).
#
# Test data: schema VVSCAN (or --schema=NAME), dropped and recreated. No index is registered.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVSCAN
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "test_vscan.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE=""
. tests/sql/lib.sh

VEC=$(random_vector "$DIMS")
L1_EXPR=""
for ((i = 0; i < DIMS; i++)); do L1_EXPR+="${L1_EXPR:+ + }ABS(l.vec[$i] - q.qvec[$i])"; done

score_expr() {   # METRIC -> the SQL score of l.vec against q.qvec
    case "$1" in
        l2) echo "VECTOR_L2(l.vec, q.qvec)" ;;
        cosine) echo "COSINE_SIMILARITY(l.vec, q.qvec)" ;;
        dot) echo "DOT_PRODUCT(l.vec, q.qvec)" ;;
        l1) echo "($L1_EXPR)" ;;
    esac
}
order_of() { case "$1" in l2|l1) echo "ASC" ;; *) echo "DESC" ;; esac; }

# The query vectors as text (one line per query: '[a, b, ...]'), the same numbers the queries
# table holds: generated here, inserted as literals.
gen_queries() {   # N SEED -> N lines
    awk -v n="$1" -v d="$DIMS" -v seed="$2" 'BEGIN { srand(seed); for (i = 1; i <= n; i++) { s = "["; for (j = 1; j <= d; j++) { s = s (j > 1 ? ", " : "") sprintf("%.6f", rand() * 2 - 1) } print s "]" } }'
}
QUERY_LINES=$(gen_queries 20 41)
INSERTS=""
qid=1
while IFS= read -r line; do
    INSERTS+="INSERT INTO $SCHEMA.queries VALUES ($qid, ARRAY${line});"$'\n'
    qid=$((qid + 1))
done <<< "$QUERY_LINES"
QUERIES_PARAM=$(echo "$QUERY_LINES" | paste -sd ';' -)
Q1=$(echo "$QUERY_LINES" | sed -n '1p')

# vscan_sql METRIC K OVER_CLAUSE PARAMS [INPUT]: the merged result (qid, id, score, rank) of vscan.
vscan_sql() {
    local m=$1 k=$2 over=$3 extra=${4:-} input=${5:-$SCHEMA.vecs}
    echo "SELECT qid, id, score, rank FROM (
              SELECT qid, id, score, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score $(order_of "$m"), id) AS rank
              FROM (SELECT vvector.vscan(id, vec USING PARAMETERS queries='$QUERIES_PARAM', k=$k, metric='$m'$extra) $over FROM $input) s) r
          WHERE rank <= $k"
}

# compare NAME METRIC K OVER_CLAUSE [PARAMS [INPUT [REF_INPUT]]]: vscan against the SQL reference.
compare() {
    local name=$1 m=$2 k=$3 over=$4 extra=${5:-} input=${6:-$SCHEMA.vecs} ref=${7:-${6:-$SCHEMA.vecs}}
    expect "$name" "^mismatch: missing 0, extra 0, score 0, rank 0, queries 20$" "
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS $(vscan_sql "$m" "$k" "$over" "$extra" "$input");
CREATE TABLE $SCHEMA.ref AS SELECT qid, id, score, rank,
    COALESCE(ABS(score - LAG(score) OVER(PARTITION BY qid ORDER BY rank)) <= 1e-5 * GREATEST(1, ABS(score)), FALSE)
    OR COALESCE(ABS(LEAD(score) OVER(PARTITION BY qid ORDER BY rank) - score) <= 1e-5 * GREATEST(1, ABS(score)), FALSE) AS tie
    FROM (SELECT qid, id, score, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score $(order_of "$m"), id) AS rank
          FROM (SELECT q.qid, l.id, $(score_expr "$m") AS score FROM $SCHEMA.queries q CROSS JOIN (SELECT id, vec FROM $ref WHERE id IS NOT NULL AND vec IS NOT NULL) l) s) r WHERE rank <= $k;
SELECT 'mismatch: missing ' || SUM(missing) || ', extra ' || SUM(extra) || ', score ' || SUM(bad_score) || ', rank ' || SUM(bad_rank)
       || ', queries ' || COUNT(DISTINCT qid)
FROM (
    SELECT COALESCE(r.qid, g.qid) AS qid,
           CASE WHEN g.id IS NULL AND (kth.score IS NULL OR ABS(r.score - kth.score) > 1e-5 * GREATEST(1, ABS(kth.score))) THEN 1 ELSE 0 END AS missing,
           CASE WHEN r.id IS NULL AND (kth2.score IS NULL OR ABS(g.score - kth2.score) > 1e-5 * GREATEST(1, ABS(kth2.score))) THEN 1 ELSE 0 END AS extra,
           CASE WHEN r.id IS NOT NULL AND g.id IS NOT NULL AND ABS(r.score - g.score) > 1e-5 * GREATEST(1, ABS(r.score)) THEN 1 ELSE 0 END AS bad_score,
           CASE WHEN r.id IS NOT NULL AND g.id IS NOT NULL AND r.rank <> g.rank AND NOT r.tie THEN 1 ELSE 0 END AS bad_rank
    FROM $SCHEMA.ref r FULL OUTER JOIN $SCHEMA.got g ON r.qid = g.qid AND r.id = g.id
    LEFT JOIN (SELECT qid, score FROM $SCHEMA.ref WHERE rank = $k) kth ON kth.qid = r.qid
    LEFT JOIN (SELECT qid, score FROM $SCHEMA.ref WHERE rank = $k) kth2 ON kth2.qid = g.qid
) c;"
}

echo "== test data: a table of $ROWS vectors of $DIMS elements, 20 queries (no index)"
expect "table and queries" "^vecs $ROWS, queries 20$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.vecs (id INT, vec ARRAY[FLOAT], grp INT) ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.vecs SELECT id, $VEC, id % 7 FROM ($(row_numbers "$ROWS")) n;
CREATE TABLE $SCHEMA.queries (qid INT, qvec ARRAY[FLOAT]);
$INSERTS
COMMIT;
SELECT 'vecs ' || (SELECT COUNT(*) FROM $SCHEMA.vecs) || ', queries ' || (SELECT COUNT(*) FROM $SCHEMA.queries);"

echo "== exact results against the built-ins, OVER(PARTITION BEST) and OVER()"
for m in l2 cosine dot l1; do compare "$m: vscan OVER(PARTITION BEST) equals the full scan" "$m" 10 "OVER(PARTITION BEST)"; done
compare "l2: vscan OVER() (one instance) equals the full scan" l2 10 "OVER()"
compare "cosine: k 100" cosine 100 "OVER(PARTITION BEST)"
compare "dot: k above the rows returns every row in order" dot 100 "OVER(PARTITION BEST)" "" "(SELECT * FROM $SCHEMA.vecs WHERE id <= 50) f"

echo "== one query, a filter, radius"
expect "the query parameter: qid 0, the same ids as query 1 of the batch" "^same 10 of 10$" "
SELECT 'same ' || COUNT(*) || ' of 10' FROM
    (SELECT id FROM (SELECT vvector.vscan(id, vec USING PARAMETERS query='$Q1', k=10, metric='l2') OVER(PARTITION BEST) FROM $SCHEMA.vecs) s
     WHERE qid = 0 ORDER BY score LIMIT 10) a
    JOIN ($(vscan_sql l2 10 "OVER(PARTITION BEST)")) b ON b.id = a.id AND b.qid = 1;"
compare "a WHERE filter on the table (true pre-filtering)" l2 10 "OVER(PARTITION BEST)" "" "(SELECT * FROM $SCHEMA.vecs WHERE grp = 3) f"
# Every instance returns up to k rows per query (its own k best): the merged result (rank <= k) is
# what the user sees; the checks below look at that.
expect "radius: only rows within it, at most k after the merge, some" "^within: t, at most k: t, some: t$" "
SELECT 'within: ' || (MAX(score) <= 1.9)::VARCHAR || ', at most k: ' || (MAX(n) <= 5)::VARCHAR || ', some: ' || (SUM(n) > 0)::VARCHAR
FROM (SELECT qid, COUNT(*) AS n, MAX(score) AS score FROM ($(vscan_sql l2 5 "OVER(PARTITION BEST)" ", radius=1.9")) r GROUP BY qid) g;"
expect "radius equals the full scan cut at the radius (l2 <= 1.9), k 16384" "^mismatch: 0$" "
SELECT 'mismatch: ' || COUNT(*) FROM (
    (SELECT qid, id FROM ($(vscan_sql l2 16384 "OVER(PARTITION BEST)" ", radius=1.9")) s
     EXCEPT SELECT q.qid, l.id FROM $SCHEMA.queries q CROSS JOIN $SCHEMA.vecs l WHERE VECTOR_L2(l.vec, q.qvec) <= 1.9)
    UNION ALL
    (SELECT q.qid, l.id FROM $SCHEMA.queries q CROSS JOIN $SCHEMA.vecs l WHERE VECTOR_L2(l.vec, q.qvec) <= 1.9 - 1e-5
     EXCEPT SELECT qid, id FROM ($(vscan_sql l2 16384 "OVER(PARTITION BEST)" ", radius=1.9")) s)
) d;"

echo "== NULLs, ties, an empty input, a long queries parameter"
expect "NULL ids and NULL vectors are skipped" "^mismatch: 0$" "
INSERT INTO $SCHEMA.vecs VALUES (NULL, ARRAY[$(printf '0.5, %.0s' $(seq 1 $((DIMS - 1))))0.5], 0);
INSERT INTO $SCHEMA.vecs VALUES (-1, NULL, 0);
COMMIT;
SELECT 'mismatch: ' || COUNT(*) FROM (
    SELECT qid, id FROM ($(vscan_sql l2 10 "OVER(PARTITION BEST)")) a
    EXCEPT SELECT qid, id FROM ($(vscan_sql l2 10 "OVER(PARTITION BEST)" "" "(SELECT * FROM $SCHEMA.vecs WHERE id > 0) f")) b) d;"
expect "equal vectors under two ids: both returned, ordered by id" "^1000001 1000002$" "
INSERT INTO $SCHEMA.vecs SELECT 1000001, ARRAY${Q1}, 0;
INSERT INTO $SCHEMA.vecs SELECT 1000002, ARRAY${Q1}, 0;
COMMIT;
SELECT MIN(CASE WHEN rank = 1 THEN id END) || ' ' || MIN(CASE WHEN rank = 2 THEN id END) FROM ($(vscan_sql l2 2 "OVER(PARTITION BEST)")) r WHERE qid = 1;"
expect "an empty input gives no rows" "^0 rows$" "
SELECT COUNT(*) || ' rows' FROM (SELECT vvector.vscan(id, vec USING PARAMETERS query='$Q1', k=10, metric='l2') OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id < -100) s;"
# 500 queries of 16 elements: about 85 KB, more than a VARCHAR parameter holds (65000 bytes).
LONG_LINES=$(gen_queries 500 43)
LONG_PARAM=$(echo "$LONG_LINES" | paste -sd ';' -)
MERGED_LONG="SELECT qid, id FROM (SELECT qid, id, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score, id) AS rank
             FROM (SELECT vvector.vscan(id, vec USING PARAMETERS queries='$LONG_PARAM', k=10, metric='l2') OVER(PARTITION BEST) FROM $SCHEMA.vecs) s) r WHERE rank <= 10"
expect "500 queries in one LONG VARCHAR parameter ($(echo -n "$LONG_PARAM" | wc -c | tr -d ' ') bytes): 500 x 10 rows after the merge, query 7 equals its single search" "^5000 rows, same 10 of 10$" "
SELECT (SELECT COUNT(*) FROM ($MERGED_LONG) m) || ' rows, same '
    || (SELECT COUNT(*) FROM ($MERGED_LONG) a
        JOIN (SELECT id FROM (SELECT id, ROW_NUMBER() OVER(ORDER BY score, id) AS rank
                              FROM (SELECT vvector.vscan(id, vec USING PARAMETERS query='$(echo "$LONG_LINES" | sed -n '7p')', k=10, metric='l2') OVER(PARTITION BEST) FROM $SCHEMA.vecs) s) r
              WHERE rank <= 10) b ON b.id = a.id
        WHERE a.qid = 7)
    || ' of 10';"

echo "== error messages"
expect "a vector with the wrong number of elements is refused" "vscan: the query has $DIMS elements, the vector of id 5000001 has 3" "
INSERT INTO $SCHEMA.vecs VALUES (5000001, ARRAY[1.0, 2.0, 3.0], 0);
SELECT vvector.vscan(id, vec USING PARAMETERS query='$Q1', k=10, metric='l2') OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id = 5000001;"
expect "a bad metric is refused" "vscan: metric must be l2, cosine, dot or l1" "
SELECT vvector.vscan(id, vec USING PARAMETERS query='$Q1', k=10, metric='euclid') OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id < 10;"
expect "k 0 is refused" "vscan: k must be 1 to 16384" "
SELECT vvector.vscan(id, vec USING PARAMETERS query='$Q1', k=0) OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id < 10;"
expect "no query is refused" "vscan: a query is required" "
SELECT vvector.vscan(id, vec USING PARAMETERS k=10) OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id < 10;"
expect "queries of different lengths are refused" "vscan: query 2 has 3 elements, the first has $DIMS" "
SELECT vvector.vscan(id, vec USING PARAMETERS queries='$Q1;[1, 2, 3]', k=10) OVER(PARTITION BEST) FROM $SCHEMA.vecs WHERE id < 10;"

finish_tests "test_vscan"
