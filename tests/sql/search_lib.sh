# Shared by test_search.sh and test_hnsw.sh: the full-scan reference and the comparison with it.
# Sourced after lib.sh. Expects SCHEMA (with a table journal (id, vec, del, ts)), DIMS, and
# IX_PREFIX: the indexes are named <IX_PREFIX>l2, <IX_PREFIX>cos, <IX_PREFIX>dot, <IX_PREFIX>l1.
# ALLOW (optional, test_filter.sh): a table (id) in SCHEMA; when set, compare and recall search with
# its ids as allow-list rows (filtered search) and the reference is limited to them.
ALLOW="${ALLOW:-}"

VEC=$(random_vector "$DIMS")
L1_EXPR=""
for ((i = 0; i < DIMS; i++)); do L1_EXPR+="${L1_EXPR:+ + }ABS(l.vec[$i] - q.qvec[$i])"; done

# The live vectors of the journal: latest row per id by version, a delete wins a tie.
LIVE="SELECT id, vec FROM (SELECT id, vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, del DESC) AS rn
      FROM $SCHEMA.journal) j WHERE rn = 1 AND NOT del"

allow_rows() { [ -n "$ALLOW" ] && echo "UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.$ALLOW"; }
live_rows() { if [ -n "$ALLOW" ]; then echo "SELECT * FROM ($LIVE) a WHERE id IN (SELECT id FROM $SCHEMA.$ALLOW)"; else echo "$LIVE"; fi; }

score_expr() {   # METRIC -> the SQL score of l.vec against q.qvec
    case "$1" in
        l2) echo "VECTOR_L2(l.vec, q.qvec)" ;;
        cos) echo "COSINE_SIMILARITY(l.vec, q.qvec)" ;;
        dot) echo "DOT_PRODUCT(l.vec, q.qvec)" ;;
        l1) echo "($L1_EXPR)" ;;
    esac
}
order_of() { case "$1" in l2|l1) echo "ASC" ;; *) echo "DESC" ;; esac; }

# compare NAME METRIC QUERY_TABLE K FRESHNESS [EXTRA_PARAMS [REF_QUERY_TABLE]]: vsearch over the
# delta view with the queries of QUERY_TABLE against the full-scan reference over the live rows,
# for the queries of REF_QUERY_TABLE (default: all; the reference in SQL is slow for many queries).
compare() {
    local name=$1 m=$2 qt=$3 k=$4 fresh=$5 extra=${6:-} rqt=${7:-$3}
    expect "$name" "^mismatch: missing 0, extra 0, score 0, rank 0, queries [1-9]" "
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='${IX_PREFIX}$m', k=$k, freshness='$fresh'$extra) OVER()
    FROM (SELECT * FROM $SCHEMA.${IX_PREFIX}${m}_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.$qt $(allow_rows)) x;
CREATE TABLE $SCHEMA.ref AS SELECT qid, id, score, rank,
    COALESCE(ABS(score - LAG(score) OVER(PARTITION BY qid ORDER BY rank)) <= 1e-5 * GREATEST(1, ABS(score)), FALSE)
    OR COALESCE(ABS(LEAD(score) OVER(PARTITION BY qid ORDER BY rank) - score) <= 1e-5 * GREATEST(1, ABS(score)), FALSE) AS tie
    FROM (SELECT qid, id, score, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score $(order_of "$m"), id) AS rank
          FROM (SELECT q.qid, l.id, $(score_expr "$m") AS score FROM $SCHEMA.$rqt q CROSS JOIN ($(live_rows)) l) s) r WHERE rank <= $k;
SELECT 'mismatch: missing ' || SUM(missing) || ', extra ' || SUM(extra) || ', score ' || SUM(bad_score) || ', rank ' || SUM(bad_rank)
       || ', queries ' || COUNT(DISTINCT qid)
FROM (
    SELECT COALESCE(r.qid, g.qid) AS qid,
           CASE WHEN g.id IS NULL AND (kth.score IS NULL OR ABS(r.score - kth.score) > 1e-5 * GREATEST(1, ABS(kth.score))) THEN 1 ELSE 0 END AS missing,
           CASE WHEN r.id IS NULL AND (kth2.score IS NULL OR ABS(g.score - kth2.score) > 1e-5 * GREATEST(1, ABS(kth2.score))) THEN 1 ELSE 0 END AS extra,
           CASE WHEN r.id IS NOT NULL AND g.id IS NOT NULL AND ABS(r.score - g.score) > 1e-5 * GREATEST(1, ABS(r.score)) THEN 1 ELSE 0 END AS bad_score,
           CASE WHEN r.id IS NOT NULL AND g.id IS NOT NULL AND r.rank <> g.rank AND NOT r.tie THEN 1 ELSE 0 END AS bad_rank
    FROM $SCHEMA.ref r FULL OUTER JOIN (SELECT * FROM $SCHEMA.got WHERE qid IN (SELECT qid FROM $SCHEMA.$rqt)) g
         ON r.qid = g.qid AND r.id = g.id
    LEFT JOIN (SELECT qid, score FROM $SCHEMA.ref WHERE rank = $k) kth ON kth.qid = r.qid
    LEFT JOIN (SELECT qid, score FROM $SCHEMA.ref WHERE rank = $k) kth2 ON kth2.qid = g.qid
) c;"
}

# search_sql INDEX PARAMS INPUT: a vsearch statement.
search_sql() { echo "SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$1'$2) OVER() FROM $3"; }

# recall NAME METRIC QUERY_TABLE K PARAMS MIN: recall@K of vsearch (over the delta view, with the
# extra PARAMS) against the full scan of the live rows; passes when it is at least MIN. Prints
# "recall <value>" in the output of a failure.
recall() {
    local name=$1 m=$2 qt=$3 k=$4 extra=$5 min=$6
    expect "$name" "^recall ok" "
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='${IX_PREFIX}$m', k=$k$extra) OVER()
    FROM (SELECT * FROM $SCHEMA.${IX_PREFIX}${m}_delta UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.$qt $(allow_rows)) x;
CREATE TABLE $SCHEMA.ref AS SELECT qid, id FROM (SELECT qid, id, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score $(order_of "$m"), id) AS rank
    FROM (SELECT q.qid, l.id, $(score_expr "$m") AS score FROM $SCHEMA.$qt q CROSS JOIN ($(live_rows)) l) s) r WHERE rank <= $k;
SELECT CASE WHEN r >= $min THEN 'recall ok ' ELSE 'recall too low ' END || r::NUMERIC(6,4) FROM
    (SELECT (SELECT COUNT(*) FROM $SCHEMA.got g JOIN $SCHEMA.ref r ON g.qid = r.qid AND g.id = r.id) / (SELECT COUNT(*) FROM $SCHEMA.ref)::FLOAT AS r) x;"
}
