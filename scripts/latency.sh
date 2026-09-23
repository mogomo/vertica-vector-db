#!/usr/bin/env bash
# Where the time of a single search goes: runs each statement shape many times in one vsql session
# and reports the median and 99th percentile, at the client (vsql \timing) and at the server
# (v_monitor.query_requests, found by a LABEL hint unique to this run).
#
#   scripts/latency.sh --index=NAME --schema=SCHEMA [--runs=N] [--shapes=a,b,...] [--echo_only]
#
#   --index    a registered index with a version column (for the delta shapes), for example the
#              SIFT1M index of scripts/benchmark.sh; its query table <SCHEMA>.<data>_query supplies
#              the query vector (--queries=TABLE to name another table with columns qid, qvec)
#   --runs     repetitions per shape (default 200); the first 5 are warm-up and not counted
#
# Shapes (each isolates one more component):
#   select1    SELECT 1: client round trip, parse, trivial plan
#   vversion   a transform function with no input: function setup, fenced or not
#   tiny       vsearch, query parameter, 16-vector index: setup, parameters and cache, no data
#   snap       vsearch, query parameter, FROM <index>_snap: the fastest statement on the index
#   dual       the same FROM dual (no view, no stale check)
#   literal    the query as an ARRAY literal row instead of the query parameter
#   delta0     query parameter FROM <index>_delta with freshness exact, the delta holds the sentinel
#   delta1000  the same with 1000 journal rows (inserted before, deleted after)
#   threads1, threads2, threads4   the snap shape with threads=1, 2, 4: the engine's share
# Creates index vvlat_tiny in SCHEMA (kept for later runs).
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

INDEX= SCHEMA= RUNS=200 SHAPES=select1,vversion,tiny,snap,dual,literal,delta0,delta1000,threads1,threads2,threads4 QUERIES= ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --index=*)   INDEX="${arg#*=}" ;;
        --schema=*)  SCHEMA="${arg#*=}" ;;
        --runs=*)    RUNS="${arg#*=}" ;;
        --shapes=*)  SHAPES="${arg#*=}" ;;
        --queries=*) QUERIES="${arg#*=}" ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "latency.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$INDEX" ] && [ -n "$SCHEMA" ] || { echo "latency.sh: --index and --schema are required" >&2; exit 2; }
case "$INDEX$SCHEMA$RUNS${QUERIES//./}" in *[!A-Za-z0-9_]*) echo "latency.sh: names must be letters, digits or underscores" >&2; exit 2 ;; esac

TAG="vvlat$(date +%s)"
SRC=$(vsql -X -A -t -c "SELECT source_table FROM vvector.manifest WHERE index_name = '$INDEX'" 2>/dev/null || true)
[ -n "$QUERIES" ] || QUERIES="$SCHEMA.$(sed -E 's/.*\.//; s/_base$//' <<< "$SRC")_query"

setup_tiny() {
    vsql -X -q -v ON_ERROR_STOP=1 <<SQL > /dev/null
DROP TABLE IF EXISTS $SCHEMA.vvlat_tiny CASCADE;
CREATE TABLE $SCHEMA.vvlat_tiny AS SELECT qid AS id, qvec AS vec FROM $QUERIES WHERE qid < 16;
SQL
    vsql -X -q -c "CALL vvector.unregister_index('vvlat_tiny');" > /dev/null 2>&1 || true
    vsql -X -q -v ON_ERROR_STOP=1 -c "CALL vvector.register_index('vvlat_tiny', '$SCHEMA.vvlat_tiny', 'id', 'vec', NULL, NULL, 'l2', NULL);" > /dev/null
    vsql -X -q -v ON_ERROR_STOP=1 -c "CALL vvector.refresh_index('vvlat_tiny');" > /dev/null
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
        threads*)  echo "SELECT /*+LABEL(${TAG}_$1)*/ vvector.vsearch($V USING PARAMETERS index_name='$INDEX', query='$Q', k=10, threads=${1#threads}) OVER() FROM $SCHEMA.${INDEX}_snap;" ;;
    esac
}

if [ "$ECHO_ONLY" = yes ]; then
    for s in ${SHAPES//,/ }; do statement "$s" | cut -c1-300; done
    exit 0
fi

[[ ",$SHAPES," == *,tiny,* ]] && setup_tiny
printf "%-10s %10s %10s %10s %10s   %s\n" shape client_p50 client_p99 server_p50 server_p99 "(ms, $RUNS runs, vsearch $(vsql -X -A -t -c "SELECT CASE WHEN MIN(is_fenced::INT) = 1 THEN 'fenced' ELSE 'not fenced' END FROM v_catalog.user_functions WHERE schema_name = 'vvector' AND function_name = 'vsearch'"))"
for s in ${SHAPES//,/ }; do
    if [ "$s" = delta1000 ]; then
        JT=$(vsql -X -A -t -c "SELECT source_table FROM vvector.manifest WHERE index_name = '$INDEX'")
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
    [ "$s" = delta1000 ] && vsql -X -q -c "DELETE FROM $JT WHERE id >= 2000000000; COMMIT;" > /dev/null
    printf "%-10s %10s %10s %10s %10s\n" "$s" $client $server
done
