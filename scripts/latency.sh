#!/usr/bin/env bash
# Where the time of a single search goes: runs each statement shape many times in one vsql session
# and reports the median and 99th percentile, at the client (vsql \timing) and at the server
# (v_monitor.query_requests, found by a LABEL hint unique to this run).
#
#   scripts/latency.sh --index=NAME --schema=SCHEMA [--runs=N] [--shapes=a,b,...] [--first_call] [--echo_only]
#
#   --index    a registered index with a version column (for the delta shapes), for example the
#              SIFT1M index of scripts/benchmark.sh; its query table <SCHEMA>.<data>_query supplies
#              the query vector (--queries=TABLE to name another table with columns qid, qvec)
#   --runs     repetitions per shape (default 200); the first 5 are warm-up and not counted
#   --first_call  the cost of the first search of a session instead: every run opens a new vsql
#              session and sends the statement twice; reports the median of the first and of the
#              second (client ms). Fenced, every session starts a new fenced process, which maps the
#              snapshot anew. Runs = --runs / 10, at least 10.
#
# Shapes (each isolates one more component):
#   select1    SELECT 1: client round trip, parse, trivial plan
#   vversion   a transform function with no input: function setup, fenced or not
#   tiny       vsearch, query parameter, 16-vector index: setup, parameters and cache, no data
#   snap       vsearch, query parameter, FROM <index>_snap: the fastest statement on the index
#   dual       the same FROM dual (no view, no stale check)
#   literal    the query as an ARRAY literal row instead of the query parameter
#   delta0     query parameter FROM <index>_delta with freshness exact, the delta holds the sentinel
#   delta1000  the same with 1000 journal rows (written into the index's journal before and deleted
#              physically after, also when the script is stopped; a refresh in between sees them)
#   threads1, threads2, threads4   the snap shape with threads=1, 2, 4: the engine's share
#   vknn       vknn with the query parameter FROM dual: no OVER(), no view, no stale check
#   vknnrow    vknn on the query row of the query table (qid = 0), beside its qid
#   filter100, filter10k, filter100k   the snap shape with an allow-list of 100, 10,000 or 100,000
#              ids (rows of SCHEMA.vvlat_allow, UNSEGMENTED ALL NODES; filtered search)
#   filter100seg, filter10kseg, filter100kseg   the same ids from a segmented table (vvlat_allow_seg):
#              on a cluster the rows are gathered from every node
#   range10    the snap shape with k 16384 and the radius of the query's 10th neighbour (range search)
# Creates index vvlat_tiny and tables vvlat_allow, vvlat_allow_seg in SCHEMA (kept for later runs).
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

INDEX= SCHEMA= RUNS=200 SHAPES=select1,vversion,tiny,snap,dual,literal,delta0,delta1000,threads1,threads2,threads4 QUERIES= ECHO_ONLY=no FIRST=no
for arg in "$@"; do
    case "$arg" in
        --index=*)   INDEX="${arg#*=}" ;;
        --schema=*)  SCHEMA="${arg#*=}" ;;
        --runs=*)    RUNS="${arg#*=}" ;;
        --shapes=*)  SHAPES="${arg#*=}" ;;
        --queries=*) QUERIES="${arg#*=}" ;;
        --first_call) FIRST=yes ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "latency.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$INDEX" ] && [ -n "$SCHEMA" ] || { echo "latency.sh: --index and --schema are required" >&2; exit 2; }
case "$INDEX$SCHEMA$RUNS${QUERIES//./}" in *[!A-Za-z0-9_]*) echo "latency.sh: names must be letters, digits or underscores" >&2; exit 2 ;; esac

TAG="vvlat$(date +%s)"
SRC=$(vsql -X -A -t -c "SELECT source_table FROM vvector.manifest WHERE index_name = '$INDEX'" 2>/dev/null || true)
[ -n "$SRC" ] || [ "$ECHO_ONLY" = yes ] || { echo "latency.sh: index $INDEX is not registered" >&2; exit 2; }
[ -n "$QUERIES" ] || QUERIES="$SCHEMA.$(sed -E 's/.*\.//; s/_base$//' <<< "$SRC")_query"

setup_tiny() {
    vsql -X -q -v ON_ERROR_STOP=1 <<SQL > /dev/null
DROP TABLE IF EXISTS $SCHEMA.vvlat_tiny CASCADE;
CREATE TABLE $SCHEMA.vvlat_tiny AS SELECT qid AS id, qvec AS vec FROM $QUERIES WHERE qid < 16;
SQL
    vsql -X -q -c "CALL vvector.unregister_index('vvlat_tiny');" > /dev/null 2>&1 || true
    vsql -X -q -v ON_ERROR_STOP=1 -c "CALL vvector.register_index('vvlat_tiny', '$SCHEMA.vvlat_tiny', 'id', 'vec', NULL, NULL, 'l2', NULL, 'flat');" > /dev/null
    vsql -X -q -v ON_ERROR_STOP=1 -c "CALL vvector.refresh_index('vvlat_tiny');" > /dev/null
}

# Every 10th id of the index (100,000 of SIFT1M's 1,000,000); the filter shapes take subsets.
# vvlat_allow is UNSEGMENTED ALL NODES (read on the initiator); vvlat_allow_seg holds the same rows
# segmented by id, as a filter on a large table would be.
setup_allow() {
    vsql -X -q -v ON_ERROR_STOP=1 <<SQL > /dev/null
DROP TABLE IF EXISTS $SCHEMA.vvlat_allow;
DROP TABLE IF EXISTS $SCHEMA.vvlat_allow_seg;
CREATE TABLE $SCHEMA.vvlat_allow (id INT NOT NULL, size INT NOT NULL) ORDER BY size, id UNSEGMENTED ALL NODES;
CREATE TABLE $SCHEMA.vvlat_allow_seg (id INT NOT NULL, size INT NOT NULL) ORDER BY size, id SEGMENTED BY HASH(id) ALL NODES;
INSERT INTO $SCHEMA.vvlat_allow
SELECT id, CASE WHEN n % 1000 = 0 THEN 100 WHEN n % 10 = 0 THEN 10000 ELSE 100000 END AS size
FROM (SELECT id, ROW_NUMBER() OVER(ORDER BY id) - 1 AS n FROM (SELECT DISTINCT id FROM $SRC) d) i WHERE n < 100000;
INSERT INTO $SCHEMA.vvlat_allow_seg SELECT id, size FROM $SCHEMA.vvlat_allow;
COMMIT;
SELECT ANALYZE_STATISTICS('$SCHEMA.vvlat_allow');
SELECT ANALYZE_STATISTICS('$SCHEMA.vvlat_allow_seg');
SQL
}

Q=$(vsql -X -A -t -c "SELECT TO_JSON(qvec) FROM $QUERIES WHERE qid = 0" 2>/dev/null || echo "[0]")
QROW="SELECT 0 AS qid, (ARRAY$Q)::ARRAY[FLOAT] AS qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, NULL::BOOLEAN AS del, NULL::INT AS ver, NULL::INT AS snapshot_id"
V="qid, qvec, id, vec, del, ver, snapshot_id"
statement() {   # SHAPE -> one SQL statement
    case "$1" in
        select1)   echo "SELECT /*+LABEL(${TAG}_$1)*/ 1;" ;;
        vversion)  echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vversion() OVER();" ;;
        tiny)      echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='vvlat_tiny', query='$Q', k=1) OVER() FROM $SCHEMA.vvlat_tiny_snap;" ;;
        snap)      echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10) OVER() FROM $SCHEMA.${INDEX}_snap;" ;;
        dual)      echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch(NULL::INT, NULL::ARRAY[FLOAT], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT, NULL::INT USING PARAMETERS index_name='$INDEX', query='$Q', k=10) OVER() FROM dual;" ;;
        literal)   echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', k=10) OVER() FROM ($QROW) q;" ;;
        delta0|delta1000)
                   echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10, freshness='exact') OVER() FROM $SCHEMA.${INDEX}_delta;" ;;
        vknn)      echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='$INDEX', query='$Q', k=10) FROM dual;" ;;
        vknnrow)   echo "SELECT /*+LABEL(${TAG}_$1)*/ q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='$INDEX', k=10) FROM $QUERIES q WHERE q.qid = 0;" ;;
        filter100|filter10k|filter100k|filter100seg|filter10kseg|filter100kseg)
                   local size=${1#filter} table=vvlat_allow
                   [ "${size%seg}" != "$size" ] && { size=${size%seg}; table=vvlat_allow_seg; }
                   size=${size/k/000}
                   echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10) OVER() FROM (SELECT * FROM $SCHEMA.${INDEX}_snap UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM $SCHEMA.$table WHERE size <= $size) x;" ;;
        range10)   echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=16384, radius=$RADIUS) OVER() FROM $SCHEMA.${INDEX}_snap;" ;;
        threads*)  echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10, threads=${1#threads}) OVER() FROM $SCHEMA.${INDEX}_snap;" ;;
    esac
}

RADIUS=1
if [[ ",$SHAPES," == *,range10,* ]] && [ "$ECHO_ONLY" = no ]; then
    RADIUS=$(vsql -X -A -t -c "SELECT score FROM (SELECT vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10, exact=true) OVER()
                                FROM $SCHEMA.${INDEX}_snap) r WHERE rank = 10")
fi
if [ "$ECHO_ONLY" = yes ]; then
    for s in ${SHAPES//,/ }; do statement "$s" | cut -c1-300; done
    exit 0
fi

[[ ",$SHAPES," == *,tiny,* ]] && setup_tiny
[[ ",$SHAPES," == *,filter* ]] && setup_allow
FENCING=$(vsql -X -A -t -c "SELECT CASE WHEN MIN(is_fenced::INT) = 1 THEN 'fenced' ELSE 'not fenced' END FROM v_catalog.user_functions WHERE schema_name = 'vvector' AND function_name = 'vsearch'")
if [ "$FIRST" = yes ]; then
    n=$((RUNS / 10)); [ "$n" -lt 10 ] && n=10
    printf "%-10s %10s %10s   %s\n" shape first_p50 second_p50 "(client ms, $n new sessions, vsearch $FENCING)"
    for s in ${SHAPES//,/ }; do
        stmt=$(statement "$s")
        for ((i = 0; i < n; i++)); do
            { echo '\timing on'; echo "$stmt"; echo "$stmt"; } | vsql -X -q -A -t 2>&1 |
                sed -n 's/.*All rows formatted: \([0-9.]*\) ms.*/\1/p' | paste -sd' ' -
        done | awk '{f[NR]=$1; g[NR]=$2} END {n=asort(f); asort(g); printf "%.2f %.2f\n", f[int((n+1)/2)], g[int((n+1)/2)]}' |
            { read -r a b; printf "%-10s %10s %10s\n" "$s" "$a" "$b"; }
    done
    exit 0
fi
printf "%-10s %10s %10s %10s %10s   %s\n" shape client_p50 client_p99 server_p50 server_p99 "(ms, $RUNS runs, vsearch $FENCING)"
for s in ${SHAPES//,/ }; do
    if [ "$s" = delta1000 ]; then
        # Writes 1000 rows into the index's journal and deletes them physically afterwards, also when
        # the script is stopped (the trap). A refresh in between would take them in; the next refresh
        # after the DELETE then verifies the journal and builds in full.
        JT=$(vsql -X -A -t -c "SELECT source_table FROM vvector.manifest WHERE index_name = '$INDEX'")
        trap 'vsql -X -q -c "DELETE FROM $JT WHERE id >= 2000000000; COMMIT;" > /dev/null' EXIT
        vsql -X -q -v ON_ERROR_STOP=1 -c "INSERT INTO $JT (id, vec) SELECT 2000000000 + qid, qvec FROM $QUERIES WHERE qid < 1000; COMMIT;" > /dev/null
    fi
    stmt=$(statement "$s")
    client=$( { echo '\timing on'; for ((i = 0; i < RUNS + 5; i++)); do echo "$stmt"; done; } | vsql -X -q -A -t 2>&1 |
              sed -n 's/.*All rows formatted: \([0-9.]*\) ms.*/\1/p' | tail -n "$RUNS" | sort -n |
              awk '{a[NR]=$1} END {if (NR == 0) {print "error error"; exit} printf "%.2f %.2f", a[int((NR+1)/2)], a[int(NR*0.99+0.5) < 1 ? 1 : int(NR*0.99+0.5)]}')
    server=$(vsql -X -A -t -F ' ' -c "SELECT APPROXIMATE_PERCENTILE(ms USING PARAMETERS percentile=0.5)::NUMERIC(10,2),
                                             APPROXIMATE_PERCENTILE(ms USING PARAMETERS percentile=0.99)::NUMERIC(10,2)
                                      FROM (SELECT EXTRACT(EPOCH FROM end_timestamp - start_timestamp) * 1000 AS ms,
                                                   ROW_NUMBER() OVER(ORDER BY start_timestamp) AS n
                                            FROM v_monitor.query_requests WHERE request_label = '${TAG}_$s' AND success) r WHERE n > 5")
    if [ "$s" = delta1000 ]; then vsql -X -q -c "DELETE FROM $JT WHERE id >= 2000000000; COMMIT;" > /dev/null; trap - EXIT; fi
    printf "%-10s %10s %10s %10s %10s\n" "$s" $client $server
done
