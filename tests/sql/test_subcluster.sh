#!/usr/bin/env bash
# test_subcluster.sh - Eon subclusters: every subcluster keeps node caches of its own (README
# "Operations", docs/design.md "Eon subclusters"). Needs a second vsql session on ANOTHER subcluster:
#   --secondary='<command>'   a command that reads SQL on stdin and runs it in a vsql session on a
#                             node of another subcluster, for example
#                             --secondary='vsql -h <address of a node of the secondary> -X -A -t -q'
# Without --secondary, or on a database with one subcluster, the test prints "skipped" and exits 0.
# Checks: a search on the other subcluster before a load (no cache), load_all there, the same results
# on both subclusters, load_all with nothing to load, a refresh on the primary followed by the stale
# error on the secondary and a load_all that applies the patch, a refresh from the secondary and the
# same on the primary, the status and refresh messages that name the subcluster.
#   tests/sql/test_subcluster.sh --secondary=CMD [--rows=N] [--dims=N] [--schema=NAME] [--echo_only]
set -uo pipefail
cd "$(dirname "$0")/../.."

ROWS=20000; DIMS=16; SCHEMA=vvtest_sub; ECHO_ONLY=no; SECONDARY=""
for a in "$@"; do
    case "$a" in
        --secondary=*) SECONDARY="${a#*=}" ;;
        --rows=*) ROWS="${a#*=}" ;;
        --dims=*) DIMS="${a#*=}" ;;
        --schema=*) SCHEMA="${a#*=}" ;;
        --echo_only) ECHO_ONLY=yes ;;
        --help|-h) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option $a (--help)"; exit 2 ;;
    esac
done
. tests/sql/lib.sh
IX=${SCHEMA}_ix

# run_sec NAME SQL / expect_sec NAME PATTERN SQL: as run_sql / expect, in the session of --secondary.
run_sec() {
    if [ "$ECHO_ONLY" = yes ]; then echo "-- $1 (secondary)"; echo "$2"; return 0; fi
    printf '%s\n' "$2" | $SECONDARY 2>&1
}
expect_sec() {
    local out
    out=$(run_sec "$1" "$3")
    if [ "$ECHO_ONLY" = yes ]; then echo "$out"; return 0; fi
    if echo "$out" | grep -q -- "$2"; then
        echo "PASS  $1"
    else
        echo "FAIL  $1"; echo "      wanted: $2"; echo "$out" | sed 's/^/      got: /' | head -6
        FAILED=$((FAILED + 1))
    fi
}
# same NAME A B: PASS when the two outputs are equal.
same() {
    if [ "$ECHO_ONLY" = yes ]; then return 0; fi
    if [ "$2" = "$3" ] && [ -n "$2" ]; then echo "PASS  $1"; else
        echo "FAIL  $1"; echo "$2" | sed 's/^/      primary:   /' | head -4; echo "$3" | sed 's/^/      secondary: /' | head -4
        FAILED=$((FAILED + 1))
    fi
}

if [ -z "$SECONDARY" ]; then
    echo "test_subcluster: skipped (no --secondary=<command for a vsql session on another subcluster>)"; exit 0
fi
if [ "$ECHO_ONLY" = no ]; then
    NSUB=$(run_sql "subclusters" "SELECT COUNT(DISTINCT subcluster_name) FROM v_catalog.nodes WHERE node_state = 'UP';")
    if [ "${NSUB:-0}" -lt 2 ]; then echo "test_subcluster: skipped (the database has ${NSUB:-0} subclusters)"; exit 0; fi
    HERE=$(run_sql "this subcluster" "SELECT subcluster_name FROM v_catalog.nodes WHERE node_name = local_node_name();")
    THERE=$(run_sec "that subcluster" "SELECT subcluster_name FROM v_catalog.nodes WHERE node_name = local_node_name();")
    if [ -z "$THERE" ] || [ "$THERE" = "$HERE" ]; then
        echo "test_subcluster: the --secondary command must open a session on another subcluster (this one: $HERE, it gave: ${THERE:-nothing})"; exit 2
    fi
    echo "== $(date -u +%FT%TZ) subclusters: primary session on $HERE, secondary session on $THERE"
fi

VEC=$(random_vector "$DIMS")
Q=$(printf '0.1%.0s,' $(seq 1 "$DIMS")); Q="[${Q%,}]"
SEARCH="SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', query='$Q', k=5, precision='exact') OVER() FROM $SCHEMA.${IX}_snap"
KNN="SELECT vvector.vknn(ARRAY$Q::ARRAY[FLOAT] USING PARAMETERS index_name='$IX', k=5, precision='exact')"
NEW="INSERT INTO $SCHEMA.t SELECT id + (SELECT MAX(id) FROM $SCHEMA.t), $VEC, false, CLOCK_TIMESTAMP() FROM ($(row_numbers 10)) r; COMMIT;"

echo "== build on the primary"
run_sql "cleanup of an earlier run" "CALL vvector.unregister_index('$IX'); DROP SCHEMA IF EXISTS $SCHEMA CASCADE;" > /dev/null
expect "table of $ROWS x $DIMS on the primary" "^rows: $ROWS$" "
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.t (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN, ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.t SELECT id, $VEC, false, CLOCK_TIMESTAMP() - INTERVAL '1 minute' FROM ($(row_numbers "$ROWS")) r;
COMMIT;
SELECT 'rows: ' || COUNT(*) FROM $SCHEMA.t;"
expect "register and refresh on the primary; the refresh names the other subclusters" "loaded on the nodes of subcluster .* only; every other subcluster loads it with CALL vvector.load_all" "
CALL vvector.register_index('$IX', '$SCHEMA.t', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');
CALL vvector.refresh_index('$IX');"
P1=$(run_sql "search on the primary" "$SEARCH;")
expect "the primary answers" "^rows: 5$" "SELECT 'rows: ' || COUNT(*) FROM ($SEARCH) s;"

echo "== the secondary before a load"
expect_sec "vsearch on the secondary before a load: no cache" "no snapshot cache for index '$IX'" "$SEARCH;"
expect_sec "vknn on the secondary before a load: no cache" "no snapshot cache for index '$IX'" "$KNN;"
expect_sec "vinfo on the secondary lists its own nodes, none loaded" "^loaded: 0 of [1-9][0-9]* nodes of $THERE" "
SELECT 'loaded: ' || SUM(CASE WHEN loaded THEN 1 ELSE 0 END) || ' of ' || COUNT(*) || ' nodes of ' || MAX(n.subcluster_name)
FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) i
JOIN v_catalog.nodes n ON n.node_name = i.node_name;"
expect_sec "status on the secondary names its subcluster and the missing cache" "cache of 0 of [1-9][0-9]* nodes of subcluster $THERE" "CALL vvector.status('$IX');"

echo "== load_all on the secondary"
expect_sec "load_all on the secondary loads its nodes" "loaded on all nodes of subcluster $THERE (1 of the chain" "CALL vvector.load_all('$IX');"
S1=$(run_sec "search on the secondary" "$SEARCH;")
same "the same 5 results on both subclusters" "$P1" "$S1"
expect_sec "load_all again: nothing to load" "already in the cache of all [1-9][0-9]* nodes of subcluster $THERE: nothing to load" "CALL vvector.load_all('$IX');"
expect_sec "status on the secondary: every node has the snapshot" "cache of \([1-9][0-9]*\) of \1 nodes of subcluster $THERE" "CALL vvector.status('$IX');"
expect "the primary's caches are untouched: every node of $HERE has the first snapshot" "^loaded: \([1-9][0-9]*\) of \1 nodes$" "
SELECT 'loaded: ' || SUM(CASE WHEN loaded THEN 1 ELSE 0 END) || ' of ' || COUNT(*) || ' nodes'
FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) i;"

echo "== a refresh on the primary, the secondary is behind"
expect "10 new rows and an incremental refresh on the primary" "incremental from snapshot" "$NEW
CALL vvector.refresh_index('$IX');"
P2=$(run_sql "search on the primary" "$SEARCH;")
expect_sec "vsearch on the secondary: stale" "snapshot cache stale on v_[a-z0-9_]*: run vload" "$SEARCH;"
expect_sec "vknn on the secondary has no stale check: it answers from the old snapshot" "^rows: 5$" "SELECT 'rows: ' || COUNT(*) FROM ($KNN) s;"
expect_sec "load_all on the secondary applies the patch" "loaded on all nodes of subcluster $THERE (2 of the chain" "CALL vvector.load_all('$IX');"
S2=$(run_sec "search on the secondary" "$SEARCH;")
same "the same results again after the patch" "$P2" "$S2"
expect_sec "vinfo on the secondary: the new snapshot on every node, patched on the first" "^ok$" "
SELECT CASE WHEN COUNT(*) > 0 AND COUNT(*) = SUM(CASE WHEN loaded AND snapshot_id = m.active_snapshot AND base_snapshot = SPLIT_PART(m.snapshot_chain, ',', 1)::INT THEN 1 ELSE 0 END)
       THEN 'ok' ELSE 'not on every node' END
FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) i
CROSS JOIN (SELECT active_snapshot, snapshot_chain FROM vvector.manifest WHERE index_name = '$IX') m;"

echo "== a refresh from the secondary, the primary is behind"
expect_sec "10 more rows and a refresh from the secondary" "incremental from snapshot" "$NEW
CALL vvector.refresh_index('$IX');"
expect "vsearch on the primary: stale" "snapshot cache stale on v_[a-z0-9_]*: run vload" "$SEARCH;"
expect "load_all on the primary applies the chain" "loaded on all nodes of subcluster $HERE (3 of the chain" "CALL vvector.load_all('$IX');"
P3=$(run_sql "search on the primary" "$SEARCH;")
S3=$(run_sec "search on the secondary" "$SEARCH;")
same "the same results on both subclusters after a refresh from the secondary" "$P3" "$S3"
expect "the exact search agrees with the built-in on the primary" "^same$" "
SELECT CASE WHEN a.ids = b.ids THEN 'same' ELSE 'differ: ' || a.ids || ' vs ' || b.ids END FROM
  (SELECT LISTAGG(id::VARCHAR) WITHIN GROUP (ORDER BY rank) AS ids FROM ($SEARCH) s) a,
  (SELECT LISTAGG(id::VARCHAR) WITHIN GROUP (ORDER BY d, id) AS ids FROM
     (SELECT id, VECTOR_L2(vec, ARRAY$Q::ARRAY[FLOAT]) AS d FROM $SCHEMA.t ORDER BY 2, 1 LIMIT 5) t) b;"

echo "== cleanup"
expect "unregister and drop" "^left: 0$" "
CALL vvector.unregister_index('$IX');
DROP SCHEMA $SCHEMA CASCADE;
SELECT 'left: ' || COUNT(*) FROM vvector.manifest WHERE index_name = '$IX';"
finish_tests test_subcluster
