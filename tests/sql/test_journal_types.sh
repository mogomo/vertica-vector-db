#!/usr/bin/env bash
# Integration test of the journal column types the other tests do not use, and of the grants on the
# views of an index.
#
#   tests/sql/test_journal_types.sh [--schema=NAME] [--cache_dir=DIR] [--echo_only]
#
# - vec ARRAY[INT], del INT (+1 add, -1 delete), ver plain TIMESTAMP: register (flat, l2, margin 0),
#   refresh; exact searches through the delta view equal the full scan of the live rows with
#   VECTOR_L2 (ids and ranks, ties by id), before and after journaled adds and deletes;
# - vec ARRAY[NUMERIC], del BOOLEAN, ver INT with margin 5: the rows of the last 5 versions stay in the
#   delta; exact searches equal the full scan with COSINE_SIMILARITY; k larger than the live rows
#   returns every live row once;
# - SELECT granted on both views survives a refresh (the refresh recreates them).
#
# Test data: schema VVTYPES (or --schema=NAME), dropped and recreated; indexes vt_int, vt_num; role
# vvtypes_viewer (dropped at the end).
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

SCHEMA=VVTYPES
CACHE_DIR=/tmp/vvector
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "test_journal_types.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh

DIMS=8
INTVEC="ARRAY[$(for i in $(seq 1 $DIMS); do printf '(RANDOM() * 10)::INT'; [ $i -lt $DIMS ] && printf ', '; done)]"
NUMVEC="ARRAY[$(for i in $(seq 1 $DIMS); do printf '(RANDOM() * 2 - 1)::NUMERIC(6,2)'; [ $i -lt $DIMS ] && printf ', '; done)]::ARRAY[NUMERIC(6,2)]"
QI="ARRAY[$(printf '5, %.0s' $(seq 1 $((DIMS - 1))))5]"
QF="ARRAY[$(printf '0.5, %.0s' $(seq 1 $((DIMS - 1))))-0.5]"

# same_top INDEX TABLE QUERY K ORDER_EXPR LIVE: vvector's exact search through the delta view against
# the full scan of the live rows ordered by ORDER_EXPR (then id). Prints "same K" or what differs.
same_top() {
    echo "SELECT CASE WHEN COUNT(*) = $4 AND SUM(CASE WHEN r.id = v.id THEN 1 ELSE 0 END) = $4 THEN 'same $4'
                 ELSE 'differ: ' || SUM(CASE WHEN r.id = v.id THEN 1 ELSE 0 END) || ' of ' || COUNT(*) END
          FROM (SELECT id, ROW_NUMBER() OVER(ORDER BY $5, id) AS rank FROM ($6) l ORDER BY $5, id LIMIT $4) r
          FULL JOIN (SELECT id, rank FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                         USING PARAMETERS index_name='$1', k=$4, freshness='exact', precision='exact') OVER()
                       FROM (SELECT * FROM $SCHEMA.$1_delta UNION ALL SELECT 1, $3::ARRAY[FLOAT], NULL, NULL, NULL, NULL, NULL) q) s) v
          ON r.rank = v.rank;"
}
LIVE_INT="SELECT id, vec FROM (SELECT id, vec, op, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, (op < 0) DESC) AS rn
          FROM $SCHEMA.ja) j WHERE rn = 1 AND op > 0"
LIVE_NUM="SELECT id, vec FROM (SELECT id, vec::ARRAY[FLOAT] AS vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY v DESC, del DESC) AS rn
          FROM $SCHEMA.jn) j WHERE rn = 1 AND NOT del"

echo "== test data"
run_sql "cleanup of an earlier run" "CALL vvector.unregister_index('vt_int'); CALL vvector.unregister_index('vt_num'); DROP ROLE vvtypes_viewer CASCADE;" > /dev/null
expect "journals: ARRAY[INT] with an INT delete flag and TIMESTAMP versions; ARRAY[NUMERIC] with INT versions" "^ja 331, jn 100$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.ja (id INT NOT NULL, vec ARRAY[INT], op INT NOT NULL DEFAULT 1, ts TIMESTAMP NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.ja (id, vec, ts) SELECT id, $INTVEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers 300)) g;
INSERT INTO $SCHEMA.ja (id, op, ts) SELECT id, -1, CLOCK_TIMESTAMP() - INTERVAL '23 hours' FROM ($(row_numbers 30)) g;
INSERT INTO $SCHEMA.ja (id, vec, ts) VALUES (7, $QI, CLOCK_TIMESTAMP() - INTERVAL '22 hours');
CREATE TABLE $SCHEMA.jn (id INT NOT NULL, vec ARRAY[NUMERIC(6,2)], del BOOLEAN NOT NULL DEFAULT FALSE, v INT NOT NULL)
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.jn (id, vec, v) SELECT id, $NUMVEC, id FROM ($(row_numbers 100)) g;
COMMIT;
SELECT 'ja ' || (SELECT COUNT(*) FROM $SCHEMA.ja) || ', jn ' || (SELECT COUNT(*) FROM $SCHEMA.jn);"

echo "== ARRAY[INT], INT delete flag (+1 / -1), TIMESTAMP versions"
expect "register and refresh vt_int (flat, l2, margin 0)" "index vt_int refreshed: snapshot [0-9]*, full build (first build), 271 vectors of $DIMS dimensions" "
CALL vvector.register_index('vt_int', '$SCHEMA.ja', 'id', 'vec', 'op', 'ts', 'l2', 0, 'flat');
CALL vvector.refresh_index('vt_int');"
expect "exact search equals VECTOR_L2 over the live rows (id 7 re-added, ids 1-30 deleted)" "^same 20$" "$(same_top vt_int ja "$QI" 20 "VECTOR_L2(vec, $QI)" "$LIVE_INT")"
expect "journaled adds near the query and deletes, no refresh: still equal" "^same 20$" "
INSERT INTO $SCHEMA.ja (id, vec) SELECT 1000 + id, ARRAY[$(printf '5, %.0s' $(seq 1 $((DIMS - 1))))4] FROM ($(row_numbers 5)) g;
INSERT INTO $SCHEMA.ja (id, op) SELECT 30 + id, -1 FROM ($(row_numbers 30)) g;
COMMIT;
$(same_top vt_int ja "$QI" 20 "VECTOR_L2(vec, $QI)" "$LIVE_INT")"
expect "the refresh folds them in: incremental" "index vt_int refreshed: snapshot [0-9]*, incremental from snapshot [0-9]* (5 vectors appended, 30 tombstoned)" "
CALL vvector.refresh_index('vt_int');"
expect "... and the search still equals the full scan" "^same 20$" "$(same_top vt_int ja "$QI" 20 "VECTOR_L2(vec, $QI)" "$LIVE_INT")"

echo "== ARRAY[NUMERIC], BOOLEAN delete flag, INT versions with margin 5"
expect "register and refresh vt_num (flat, cosine, margin 5): versions above 95 stay in the delta" "index vt_num refreshed: snapshot [0-9]*, full build (first build), 95 vectors" "
CALL vvector.register_index('vt_num', '$SCHEMA.jn', 'id', 'vec', 'del', 'v', 'cosine', 5, 'flat');
CALL vvector.refresh_index('vt_num');"
expect "exact search equals COSINE_SIMILARITY over the live rows" "^same 10$" "$(same_top vt_num jn "$QF" 10 "COSINE_SIMILARITY(vec, $QF) DESC" "$LIVE_NUM")"
expect "deletes of 10 ids (versions 200 on): still equal" "^same 10$" "
INSERT INTO $SCHEMA.jn (id, del, v) SELECT id, TRUE, 199 + id FROM ($(row_numbers 10)) g;
COMMIT;
$(same_top vt_num jn "$QF" 10 "COSINE_SIMILARITY(vec, $QF) DESC" "$LIVE_NUM")"
expect "k above the live rows returns every live row once" "^rows 90, ids 90$" "
SELECT 'rows ' || COUNT(*) || ', ids ' || COUNT(DISTINCT id) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='vt_num', k=1000, freshness='exact', precision='exact') OVER()
    FROM (SELECT * FROM $SCHEMA.vt_num_delta UNION ALL SELECT 1, $QF::ARRAY[FLOAT], NULL, NULL, NULL, NULL, NULL) q) s;"

echo "== grants on the views survive a refresh"
expect "SELECT granted on vt_int_snap and vt_int_delta" "^grants 2$" "
CREATE ROLE vvtypes_viewer;
GRANT USAGE ON SCHEMA $SCHEMA TO vvtypes_viewer;
GRANT SELECT ON $SCHEMA.vt_int_snap, $SCHEMA.vt_int_delta TO vvtypes_viewer;
SELECT 'grants ' || COUNT(*) FROM v_catalog.grants WHERE grantee = 'vvtypes_viewer' AND object_name IN ('vt_int_snap', 'vt_int_delta')
       AND privileges_description ILIKE '%SELECT%';"
expect "after a refresh with a new snapshot (both views made again): still granted" "^grants 2$" "
INSERT INTO $SCHEMA.ja (id, vec) VALUES (2000, $QI); COMMIT;
CALL vvector.refresh_index('vt_int');
SELECT 'grants ' || COUNT(*) FROM v_catalog.grants WHERE grantee = 'vvtypes_viewer' AND object_name IN ('vt_int_snap', 'vt_int_delta')
       AND privileges_description ILIKE '%SELECT%';"

run_sql "cleanup" "CALL vvector.unregister_index('vt_int'); CALL vvector.unregister_index('vt_num'); DROP ROLE vvtypes_viewer CASCADE; DROP SCHEMA $SCHEMA CASCADE;" > /dev/null
finish_tests test_journal_types
