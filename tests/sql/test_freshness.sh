#!/usr/bin/env bash
# Integration test of the freshness model: journal, delta view, register_index,
# refresh_index, load_all, status, schedule_refresh, unregister_index, the
# delta boundary with an open writer (two sessions), and the journal replica (auto on one
# node and on a cluster, on, off, shared by two indexes, postponed while a writer is open, a failure,
# dropped by unregister).
# What is tested: the delta view holds exactly the rows a query must apply, every
# refresh builds exactly the live vectors (latest row per id, not deleted), and
# vsearch applies the journal with freshness='exact' and ignores it by default.
# Exactness of the scores against the built-in functions: tests/sql/test_search.sh.
#
#   tests/sql/test_freshness.sh [--rows=N] [--dims=N] [--schema=NAME] [--cache_dir=DIR] [--schedule_fires] [--echo_only]
#
# --schedule_fires  also waits (up to 2.5 minutes) until a schedule of every minute has run a refresh
#                   by itself (tests/sql/run_all.sh passes it in its first mode only).
#
# Test data: schema VVTEST (or --schema=NAME) gets a journal table <schema>.journal;
# it is registered as index vvfresh. The schema's journal is dropped and recreated.
# Run it on a database node: one test removes a local cache file.
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD,
# VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVTEST
CACHE_DIR=/tmp/vvector
ECHO_ONLY=no
FIRES=no
IX=vvfresh

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --dims=*)      DIMS="${arg#--dims=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --schedule_fires) FIRES=yes ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "test_freshness.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Every vsql session of this test uses the same cache directory, also inside the procedures.
PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh

active_snapshot() { printf "SELECT active_snapshot FROM vvector.manifest WHERE index_name = '%s';\n" "$IX" | vsql -X -A -t -q; }
VEC=$(random_vector "$DIMS")
Q="1, ${VEC}::ARRAY[FLOAT], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT, NULL::INT"
SEARCH="SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', k=5) OVER()
FROM (SELECT * FROM $SCHEMA.${IX}_delta UNION ALL SELECT $Q) q;"
COUNT5="SELECT 'rows: ' || COUNT(*) FROM (${SEARCH%;}) r;"
# How many of the ids 6 to 1000 (deleted in the journal below) are among the 2000 nearest.
deleted_found() {   # FRESHNESS
    echo "SELECT 'deleted in results: ' || COUNT(*) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
          USING PARAMETERS index_name='$IX', k=2000, freshness='$1') OVER()
          FROM (SELECT * FROM $SCHEMA.${IX}_delta UNION ALL SELECT $Q) q) r WHERE id BETWEEN 6 AND 1000;"
}
# The live vectors of the journal: the latest row per id, not deleted. Every refresh must build exactly these.
LIVE="SELECT COUNT(*) AS n FROM (SELECT id, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC) AS rn FROM $SCHEMA.journal) j WHERE rn = 1 AND NOT del"
built_equals_live() {   # NAME
    expect "$1" "^built = live: [1-9][0-9]*$" "
SELECT CASE WHEN m.vector_count = l.n AND i.n = l.n AND m.dims = $DIMS THEN 'built = live: ' || l.n
            ELSE 'built ' || m.vector_count || ' (manifest), ' || i.n || ' (vinfo), live ' || l.n || ', dims ' || m.dims END
FROM (SELECT vector_count, dims FROM vvector.manifest WHERE index_name = '$IX') m
CROSS JOIN (SELECT MAX(vector_count) AS n FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) v) i
CROSS JOIN ($LIVE) l;"
}

echo "== journal table, register, first refresh"
run_sql "cleanup of an earlier run" "CALL vvector.unregister_index('$IX');" > /dev/null
[ "$ECHO_ONLY" = yes ] || rm -rf "${CACHE_DIR:?}/$IX"
expect "journal table (BOOLEAN del, TIMESTAMPTZ version from CLOCK_TIMESTAMP, partitioned by version date)" "^journal rows: $ROWS$" "
CREATE SCHEMA IF NOT EXISTS $SCHEMA;
DROP TABLE IF EXISTS $SCHEMA.journal CASCADE;
CREATE TABLE $SCHEMA.journal (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                              ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.journal (id, vec, ts) SELECT id, $VEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers "$ROWS")) g;
COMMIT;
SELECT 'journal rows: ' || COUNT(*) FROM $SCHEMA.journal;"

expect "register_index (margin 0: every refresh below builds every row written before it)" "index $IX registered" "CALL vvector.register_index('$IX', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', 0);"
expect "register_index refuses a second registration" "already registered" "CALL vvector.register_index('$IX', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', NULL);"
expect "register_index refuses a bad identifier" "plain identifiers" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id; DROP TABLE x', 'vec', NULL, NULL, 'l2', NULL);"
expect "register_index refuses a column that is not an array" "must be ARRAY\[FLOAT\], ARRAY\[INT\] or ARRAY\[NUMERIC\]" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'ts', NULL, NULL, 'l2', NULL);"
expect "register_index refuses an unknown metric" "metric must be l2, cosine, dot or l1" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'vec', NULL, NULL, 'manhattan', NULL);"
expect "register_index refuses a column that does not exist" "column nope does not exist in $SCHEMA.journal" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'nope', NULL, NULL, 'l2', NULL);"
expect "register_index refuses an unknown index_type" "index_type must be flat or hnsw" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'ivf');"
expect "register_index whose views cannot be made (a table has the name) leaves no manifest row" "^refused, manifest rows: 0$" "
CREATE TABLE $SCHEMA.vvbad_delta (a INT);
CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', 0);
SELECT 'refused, manifest rows: ' || COUNT(*) FROM vvector.manifest WHERE index_name = 'vvbad';"
expect "... its message names the cause" "the views could not be made" "CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', 0);"
expect "... and a second try works once the name is free" "index vvbad registered\\." "
DROP TABLE $SCHEMA.vvbad_delta;
CALL vvector.register_index('vvbad', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', 0);"
run_sql "unregister vvbad" "CALL vvector.unregister_index('vvbad');" > /dev/null
expect "register_index refuses a version column that may be NULL" "ver_col ts must be NOT NULL" "
DROP TABLE IF EXISTS $SCHEMA.nullver;
CREATE TABLE $SCHEMA.nullver (id INT NOT NULL, vec ARRAY[FLOAT], ts TIMESTAMPTZ);
CALL vvector.register_index('vvbad', '$SCHEMA.nullver', 'id', 'vec', NULL, 'ts', 'l2', NULL);"
expect "a query before the first refresh says what to do" "run vload" "$SEARCH"
expect "refresh_index" "index $IX refreshed: snapshot [0-9]" "CALL vvector.refresh_index('$IX');"
built_equals_live "the snapshot holds every vector of the journal"
expect "delta view holds only the sentinel after the refresh" "^rows 1, journal rows 0$" "
SELECT 'rows ' || COUNT(*) || ', journal rows ' || COUNT(id) FROM $SCHEMA.${IX}_delta;"
expect "vsearch through the delta view: k rows" "^rows: 5$" "$COUNT5"
expect "the _snap view is one row with the active snapshot id" "^snap rows 1, active: t$" "
SELECT 'snap rows ' || COUNT(*) || ', active: ' || (MAX(s.snapshot_id) = MAX(m.active_snapshot))::VARCHAR
FROM $SCHEMA.${IX}_snap s CROSS JOIN (SELECT active_snapshot FROM vvector.manifest WHERE index_name = '$IX') m;"

echo "== 1000 journaled adds and 1000 journaled deletes, no refresh"
expect "journal the changes" "^delta rows: 2000$" "
SET SEARCH_PATH TO $SCHEMA, public;
INSERT INTO journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 1000)) g;
INSERT INTO journal (id, del) SELECT id, TRUE FROM ($(row_numbers 1000)) g;
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM ${IX}_delta;"
expect "freshness exact: no deleted id comes back" "^deleted in results: 0$" "$(deleted_found exact)"
expect "freshness snapshot (the default): the snapshot still has them" "^deleted in results: [1-9]" "$(deleted_found snapshot)"
expect "a delete and a re-add of one id are both in the delta, the add last" "^delta rows for id 5: 3, last is an add: t$" "
SET SEARCH_PATH TO $SCHEMA, public;
INSERT INTO journal (id, del) VALUES (5, TRUE); COMMIT;
INSERT INTO journal (id, vec) VALUES (5, $VEC); COMMIT;
SELECT 'delta rows for id 5: ' || COUNT(*) || ', last is an add: ' || (MAX(CASE WHEN NOT del THEN ver END) = MAX(ver))::VARCHAR
FROM ${IX}_delta WHERE id = 5;"
expect "status reports the delta" "journal rows in the delta" "CALL vvector.status('$IX');"
expect "status warns about a version in the future (a writer that sets the version itself)" "have a version in the future" "
INSERT INTO $SCHEMA.journal (id, vec, ts) VALUES (424242, $VEC, CLOCK_TIMESTAMP() + INTERVAL '3 days'); COMMIT;
CALL vvector.status('$IX');
DELETE FROM $SCHEMA.journal WHERE id = 424242; COMMIT;"

echo "== second refresh"
expect "refresh_index again" "index $IX refreshed: snapshot [0-9]" "CALL vvector.refresh_index('$IX');"
built_equals_live "the new snapshot holds exactly the live vectors (adds in, deletes out, id 5 back)"
expect "only the active and the previous snapshot are kept after a third refresh" "^snapshots kept: 2$" "
CALL vvector.refresh_index('$IX');
SELECT 'snapshots kept: ' || COUNT(DISTINCT snapshot_id) FROM vvector.snapshot WHERE index_name = '$IX';"
SID=$(active_snapshot)

echo "== cache rules"
if [ "$ECHO_ONLY" = yes ]; then
    echo "rm -f $CACHE_DIR/$IX/<active>.vv"
else
    rm -f "$CACHE_DIR/$IX/$SID.vv"
fi
wait_cache_check
expect "a deleted cache file gives a clear error" "run vload" "$SEARCH"
expect "load_all repairs it" "loaded on all nodes" "CALL vvector.load_all('$IX');"
expect "vinfo after load_all: every node has the active snapshot" "^nodes with the active snapshot: all$" "
SELECT 'nodes with the active snapshot: ' || CASE WHEN i.ok = u.up THEN 'all' ELSE i.ok || ' of ' || u.up END
FROM (SELECT COUNT(DISTINCT node_name) AS ok FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) g
      WHERE loaded AND snapshot_id = $SID) i
CROSS JOIN (SELECT COUNT(*) AS up FROM $NODES_UP) u;"
if [ "$ECHO_ONLY" = yes ]; then
    echo "cp $CACHE_DIR/$IX/<active>.vv $CACHE_DIR/$IX/1.vv; echo 1 > $CACHE_DIR/$IX/ACTIVE"
else
    # An older snapshot file that is still marked active: what a node looks like that missed a vload.
    cp "$CACHE_DIR/$IX/$SID.vv" "$CACHE_DIR/$IX/1.vv"; echo 1 > "$CACHE_DIR/$IX/ACTIVE"
fi
wait_cache_check
expect "a cache that points to an older snapshot is refused as stale" "snapshot cache stale on .*: run vload" "$SEARCH"
# Only this node's cache was changed: on a cluster the other nodes keep the active snapshot.
expect "vinfo shows exactly one node behind (this one), the others on the active snapshot" "^nodes behind: 1$" "
SELECT 'nodes behind: ' || (u.up - i.ok)
FROM (SELECT COUNT(DISTINCT node_name) AS ok FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) g
      WHERE loaded AND snapshot_id = $SID) i
CROSS JOIN (SELECT COUNT(*) AS up FROM $NODES_UP) u;"
expect "load_all repairs that too" "loaded on all nodes" "CALL vvector.load_all('$IX');"
expect "vsearch works again after the repairs" "^rows: 5$" "$COUNT5"

echo "== schedule and unregister"
expect "schedule_refresh creates schedule and trigger" "^triggers: 1$" "
CALL vvector.schedule_refresh('$IX', '*/30 * * * *');
SELECT 'triggers: ' || COUNT(*) FROM v_catalog.stored_proc_triggers WHERE schema_name = 'vvector' AND trigger_name ILIKE '${IX}_refresh_trigger';"
if [ "$FIRES" = yes ]; then
    # A schedule of every minute: the trigger runs refresh_index in a session of its own. Every
    # refresh moves the delta boundary (delta_from), also one that keeps the snapshot.
    before=$(run_sql "boundary before" "SELECT delta_from::VARCHAR FROM vvector.manifest WHERE index_name = '$IX';")
    run_sql "schedule every minute" "CALL vvector.schedule_refresh('$IX', '* * * * *');" > /dev/null
    fired=no
    if [ "$ECHO_ONLY" = no ]; then
        for i in $(seq 1 30); do
            sleep 5
            now=$(run_sql "boundary now" "SELECT delta_from::VARCHAR || '|' || CASE WHEN refresh_started_at IS NULL THEN 'free' ELSE 'marked' END
                                          FROM vvector.manifest WHERE index_name = '$IX';")
            if [ "${now%|*}" != "$before" ] && [ "${now#*|}" = free ]; then fired=yes; break; fi
        done
    fi
    expect "the schedule ran a refresh by itself, and it removed its mark" "^fired: yes$" "SELECT 'fired: $fired';"
fi
expect "unregister_index removes trigger, view, snapshots and manifest row" "^left: 0 0 0 0$" "
CALL vvector.unregister_index('$IX');
SELECT 'left: ' || (SELECT COUNT(*) FROM v_catalog.stored_proc_triggers WHERE trigger_name ILIKE '${IX}_refresh_trigger') || ' ' ||
       (SELECT COUNT(*) FROM v_catalog.views WHERE LOWER(table_name) IN ('${IX}_delta', '${IX}_snap')) || ' ' ||
       (SELECT COUNT(*) FROM vvector.snapshot WHERE index_name = '$IX') || ' ' || (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = '$IX');"

echo "== margin 0: the strictest case"
expect "delta view holds only the sentinel right after a refresh" "^rows 1, journal rows 0$" "
CALL vvector.register_index('$IX', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', 0);
CALL vvector.refresh_index('$IX');
SELECT 'rows ' || COUNT(*) || ', journal rows ' || COUNT(id) FROM $SCHEMA.${IX}_delta;"
# The chunks of a refresh that failed after storing them (a node could not load it) belong to no
# manifest row: the next refresh that makes a snapshot removes them.
expect "a snapshot left by a failed refresh is removed by the next refresh" "^snapshots: 2, orphans: 0$" "
INSERT INTO vvector.snapshot (index_name, snapshot_id, byte_offset, chunk) SELECT '$IX', NEXTVAL('vvector.snapshot_seq'), 0, HEX_TO_BINARY('00'); COMMIT;
INSERT INTO $SCHEMA.journal (id, vec) SELECT id, vec FROM $SCHEMA.journal WHERE NOT del ORDER BY id LIMIT 1; COMMIT;
CALL vvector.refresh_index('$IX');
SELECT 'snapshots: ' || COUNT(DISTINCT snapshot_id) || ', orphans: ' || SUM(CASE WHEN OCTET_LENGTH(chunk) = 1 THEN 1 ELSE 0 END)
FROM vvector.snapshot WHERE index_name = '$IX';"

# A writer that inserted before the refresh and commits after it. Its row is not in the snapshot,
# and its version is older than the refresh. It must still reach the delta view: refresh_index
# finds the open writer through its lock and starts the delta there.
if [ "$ECHO_ONLY" = yes ]; then
    echo "-- background session: INSERT INTO $SCHEMA.journal (id, vec) VALUES (900777777, ...); SELECT SLEEP(20); COMMIT;"
else
    ( printf "INSERT INTO $SCHEMA.journal (id, vec) VALUES (900777777, $VEC);\nSELECT SLEEP(20);\nCOMMIT;\n" | vsql -X -A -t -q > /dev/null 2>&1 ) &
    WRITER=$!
    sleep 4
fi
expect "status sees the open writer" "open transactions are writing" "CALL vvector.status('$IX');"
expect "refresh while the writer is open: its row is not visible yet" "^visible during the open transaction: 0$" "
CALL vvector.refresh_index('$IX');
SELECT 'visible during the open transaction: ' || COUNT(*) FROM $SCHEMA.${IX}_delta WHERE id = 900777777;"
[ "$ECHO_ONLY" = yes ] || wait $WRITER
expect "after the late commit the row is in the delta view, without another refresh" "^visible after the commit: 1$" "
SELECT 'visible after the commit: ' || COUNT(*) FROM $SCHEMA.${IX}_delta WHERE id = 900777777;"

echo "== the default margin (60 s): rows of the last minute before a refresh stay in the delta"
ALT=$(printf '5.0, -5.0, %.0s' $(seq 1 $((DIMS / 2)))); ALT="ARRAY[${ALT%, }]"
ALT_Q=$(printf '5, -5, %.0s' $(seq 1 $((DIMS / 2)))); ALT_Q="[${ALT_Q%, }]"
nearest_m() {  # VIEW FRESHNESS: the nearest id to ALT through that view of index ${IX}_m
    echo "SELECT id FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='${IX}_m',
          query='$ALT_Q', k=1, freshness='$2', precision='exact') OVER() FROM $SCHEMA.${IX}_m_$1) s;"
}
expect "register with the default margin and refresh right after a new row" "index ${IX}_m refreshed: snapshot [0-9]*, full build (first build)" "
CALL vvector.unregister_index('${IX}_m');
INSERT INTO $SCHEMA.journal (id, vec) VALUES (900999999, $ALT); COMMIT;
CALL vvector.register_index('${IX}_m', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', NULL, 'flat');
CALL vvector.refresh_index('${IX}_m');"
expect "the new row is not in the snapshot: its nearest there is another id" "^ok$" "SELECT CASE WHEN id <> 900999999 THEN 'ok' ELSE 'the row is in the snapshot' END FROM ($(nearest_m snap snapshot | sed 's/;$//')) x;"
expect "... and found through the delta with freshness='exact'" "^900999999$" "$(nearest_m delta exact)"
run_sql "unregister ${IX}_m" "CALL vvector.unregister_index('${IX}_m'); DELETE FROM $SCHEMA.journal WHERE id = 900999999; COMMIT;" > /dev/null

echo "== journal replica (an unsegmented projection of the journal, so the delta is read on one node)"
REP="${IX}_journal_rep"
has_rep() { echo "SELECT 'replica projections: ' || COUNT(DISTINCT projection_name) FROM v_catalog.projections
                   WHERE LOWER(projection_schema) = LOWER('$SCHEMA') AND LOWER(projection_name) = LOWER('$1');"; }
NODES=$( [ "$ECHO_ONLY" = yes ] && echo 1 || vsql -X -A -t -c "SELECT COUNT(*) FROM $NODES_UP" )
if [ "$NODES" -gt 1 ]; then
    expect "auto on $NODES nodes: register and refresh made the replica" "^replica projections: 1$" "$(has_rep "$REP")"
    Q1=$(printf '%s' "$VEC" | sed 's/RANDOM() \* 2 - 1/0.5/g')
    expect "the delta view is read from the replica, on the initiator only" "Projection: .*$REP" "
EXPLAIN SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', k=5, freshness='exact') OVER()
FROM (SELECT * FROM $SCHEMA.${IX}_delta UNION ALL SELECT 1, ${Q1}::ARRAY[FLOAT], NULL, NULL, NULL, NULL, NULL) q;"
else
    expect "auto on one node: no replica, and the reason" "journal replica auto: none: a single node" "CALL vvector.status('$IX');"
    expect "auto on one node: no projection made" "^replica projections: 0$" "$(has_rep "$REP")"
fi
expect "set_journal_replica on: the replica exists" "^replica projections: 1$" "
CALL vvector.set_journal_replica('$IX', 'on');
$(has_rep "$REP")"
expect "status reports it" "journal replica on: \(created\|kept\) $SCHEMA.$REP" "CALL vvector.status('$IX');"
expect "exact search with the replica: 5 rows" "^rows: 5$" "$COUNT5"
expect "exact search with the replica: deleted ids stay out" "^deleted in results: 0$" "$(deleted_found exact)"
expect "a second index on the same journal columns shares the replica" "^shared: t, projections: 1$" "
CALL vvector.unregister_index('${IX}_b');
CALL vvector.register_index('${IX}_b', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'cosine', 0, 'flat');
CALL vvector.set_journal_replica('${IX}_b', 'on');
SELECT 'shared: ' || (a.replica_projection = b.replica_projection)::VARCHAR || ', projections: '
       || (SELECT COUNT(DISTINCT projection_name) FROM v_catalog.projections WHERE LOWER(projection_schema) = LOWER('$SCHEMA') AND NOT is_segmented)
FROM vvector.manifest a, vvector.manifest b WHERE a.index_name = '$IX' AND b.index_name = '${IX}_b';"
expect "unregistering the second index keeps the shared replica" "^replica projections: 1$" "
CALL vvector.unregister_index('${IX}_b');
$(has_rep "$REP")"
expect "set_journal_replica off drops it" "^replica projections: 0, dropped$" "
CALL vvector.set_journal_replica('$IX', 'off');
SELECT 'replica projections: ' || (SELECT COUNT(DISTINCT projection_name) FROM v_catalog.projections
       WHERE LOWER(projection_schema) = LOWER('$SCHEMA') AND LOWER(projection_name) = LOWER('$REP'))
       || CASE WHEN replica_note LIKE '%dropped%' THEN ', dropped' ELSE ', note: ' || replica_note END
FROM vvector.manifest WHERE index_name = '$IX';"
# CREATE PROJECTION waits for open writers of the table: with one open, the replica is postponed.
if [ "$ECHO_ONLY" = yes ]; then
    echo "-- background session: INSERT INTO $SCHEMA.journal (id, vec) VALUES (900888888, ...); SELECT SLEEP(10); COMMIT;"
else
    ( printf "INSERT INTO $SCHEMA.journal (id, vec) VALUES (900888888, $VEC);\nSELECT SLEEP(10);\nCOMMIT;\n" | vsql -X -A -t -q > /dev/null 2>&1 ) &
    WRITER=$!
    sleep 3
fi
expect "with an open writer the replica is postponed, the call does not wait" "journal replica on: postponed: 1 open transactions write to" "
CALL vvector.set_journal_replica('$IX', 'on');"
[ "$ECHO_ONLY" = yes ] || wait $WRITER
expect "after the writer committed, the next call makes it" "^replica projections: 1$" "
CALL vvector.set_journal_replica('$IX', 'on');
$(has_rep "$REP")"
run_sql "replica off again" "CALL vvector.set_journal_replica('$IX', 'off');" > /dev/null
expect "a replica that cannot be made is reported, not an error, with the statements for a DBA" "journal replica on: not created: .*A DBA can run: CREATE PROJECTION" "
CREATE TABLE $SCHEMA.$REP (x INT);
CALL vvector.set_journal_replica('$IX', 'on');
DROP TABLE $SCHEMA.$REP;"
expect "a bad mode is refused" "mode must be auto, on or off" "CALL vvector.set_journal_replica('$IX', 'always');"
expect "unregister_index drops the replica" "^replica projections: 0$" "
CALL vvector.set_journal_replica('$IX', 'on');
CALL vvector.unregister_index('$IX');
$(has_rep "$REP")"

finish_tests test_freshness
