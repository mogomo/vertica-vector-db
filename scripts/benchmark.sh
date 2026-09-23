#!/usr/bin/env bash
# Benchmark on SIFT1M (1,000,000 vectors of 128 dimensions, 10,000 queries, ground truth):
# the engine alone, and the SQL statements in every deploy mode.
#
#   scripts/benchmark.sh --data_dir=DIR [--schema=VVBENCH] [--modes=yes,no,mixed] [--runs=200]
#                        [--batch=1000] [--skip_load] [--echo_only]
#
#   --data_dir   where sift_base.fvecs, sift_query.fvecs and sift_groundtruth.ivecs are
#   --modes      deploy modes to measure (FENCED=yes|no|mixed); deploys fenced again at the end
#   --runs       repetitions of each single-query statement (scripts/latency.sh)
#   --skip_load  keep the tables loaded by an earlier run
#
# Measures: engine (make bench: memory bandwidth, flat search 1 query and batches, build);
# refresh of the index (build, load); per mode: the SQL full scan of one query (ORDER BY
# VECTOR_L2 LIMIT 10), the single-query shapes of scripts/latency.sh, a batch of queries in one
# statement, and recall@10 of vsearch against the ground truth. Prints the results; also saved to
# build/benchmark-<date>.txt.
# Needs make, make tools and a deployed library. Creates schema VVBENCH (tables sift_*) and
# index sift. Takes 10 to 20 minutes on 8 cores.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR= SCHEMA=VVBENCH MODES=yes,no,mixed RUNS=200 BATCH=1000 SKIP_LOAD=no ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --data_dir=*) DATA_DIR="${arg#*=}" ;;
        --schema=*)   SCHEMA="${arg#*=}" ;;
        --modes=*)    MODES="${arg#*=}" ;;
        --runs=*)     RUNS="${arg#*=}" ;;
        --batch=*)    BATCH="${arg#*=}" ;;
        --skip_load)  SKIP_LOAD=yes ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "benchmark.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$DATA_DIR" ] || { echo "benchmark.sh: --data_dir is required" >&2; exit 2; }
case "$SCHEMA$RUNS$BATCH" in *[!A-Za-z0-9_]*) echo "benchmark.sh: bad --schema, --runs or --batch" >&2; exit 2 ;; esac

IX=sift
OUT="build/benchmark-$(date +%Y%m%d-%H%M%S).txt"
if [ "$ECHO_ONLY" = yes ]; then
    echo "scripts/load_dataset.sh --dataset=sift --dir=$DATA_DIR --schema=$SCHEMA"
    echo "CALL vvector.register_index('$IX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL); CALL vvector.refresh_index('$IX');"
    echo "make bench DATA_DIR=$DATA_DIR"
    for m in ${MODES//,/ }; do echo "scripts/deploy.sh --fenced=$m; full scan; scripts/latency.sh --index=$IX --schema=$SCHEMA --runs=$RUNS; batch of $BATCH; recall@10"; done
    exit 0
fi
mkdir -p build
exec > >(tee "$OUT") 2>&1

sql() { vsql -X -A -t -q -v ON_ERROR_STOP=1 -c "$1"; }
label_ms() { sql "SELECT MAX(request_duration_ms) FROM v_monitor.query_requests WHERE request_label = '$1'"; }

echo "vvector benchmark, $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "machine: $(uname -m), $(nproc) cores, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo) memory; $(sql "SELECT version()")"
echo "library: $(sql "SELECT library_version || ', format ' || format_version || ', ' || build_flags FROM (SELECT vvector.vversion() OVER()) v")"

if [ "$SKIP_LOAD" = no ] || [ "$(sql "SELECT COUNT(*) FROM v_catalog.tables WHERE LOWER(table_schema) = LOWER('$SCHEMA') AND table_name = 'sift_base'")" = 0 ]; then
    echo; echo "== load"
    scripts/load_dataset.sh --dataset=sift --dir="$DATA_DIR" --schema="$SCHEMA" 2>&1 | grep -v NOTICE
fi

echo; echo "== engine alone (no Vertica): make bench"
make -s bench DATA_DIR="$DATA_DIR" 2>&1 | grep -vE '^(g\+\+|==)'

TAG="vvbench$(date +%s)"
echo; echo "== refresh of the index (FENCED=yes)"
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
sql "CALL vvector.unregister_index('$IX');" > /dev/null 2>&1 || true
sql "CALL vvector.register_index('$IX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL);" > /dev/null 2>&1
start=$(date +%s%N)
sql "CALL vvector.refresh_index('$IX');" > /dev/null 2>&1
printf "%-58s %12.3f s\n" "refresh_index, 1M x 128 (all steps)" "$(awk -v ns=$(( $(date +%s%N) - start )) 'BEGIN {print ns / 1e9}')"
printf "%-58s %12d ms\n" "  vbuild statement (consolidation, build, store chunks)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_build' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12d ms\n" "  vload statement (write and verify the cache file)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_load' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12s MB\n" "  snapshot size" "$(sql "SELECT index_bytes // 1048576 FROM vvector.manifest WHERE index_name = '$IX'")"

Q0="SELECT qvec FROM $SCHEMA.sift_query WHERE qid = 0"
V="qid, qvec, id, vec, del, ver, snapshot_id"
for m in ${MODES//,/ }; do
    echo; echo "== FENCED=$m"
    scripts/deploy.sh --fenced="$m" > /dev/null 2>&1
    if [ "$m" = "${MODES%%,*}" ]; then
        for i in 1 2 3; do sql "SELECT /*+LABEL(${TAG}_scan)*/ b.id, VECTOR_L2(b.vec, q.qvec) AS d FROM $SCHEMA.sift_base b CROSS JOIN ($Q0) q ORDER BY d LIMIT 10" > /dev/null; done
        printf "%-58s %12s ms\n" "SQL full scan, 1 query (ORDER BY VECTOR_L2 LIMIT 10), median of 3" \
            "$(sql "SELECT APPROXIMATE_PERCENTILE(request_duration_ms USING PARAMETERS percentile=0.5)::INT FROM v_monitor.query_requests WHERE request_label = '${TAG}_scan'")"
    fi
    scripts/latency.sh --index="$IX" --schema="$SCHEMA" --runs="$RUNS" --shapes=select1,vversion,snap,delta0,threads1 2>&1 | grep -v NOTICE
    for i in 1 2 3; do
        sql "SELECT /*+LABEL(${TAG}_batch_$m)*/ COUNT(*) FROM (SELECT vvector.vsearch($V USING PARAMETERS index_name='$IX', k=10) OVER()
             FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.sift_query WHERE qid < $BATCH) x) r" > /dev/null
    done
    ms=$(sql "SELECT APPROXIMATE_PERCENTILE(request_duration_ms USING PARAMETERS percentile=0.5)::INT FROM v_monitor.query_requests WHERE request_label = '${TAG}_batch_$m'")
    printf "%-58s %12s ms  (%s queries/s)\n" "vsearch, $BATCH queries in one statement, median of 3" "$ms" "$(( BATCH * 1000 / (ms > 0 ? ms : 1) ))"
done

echo; echo "== recall@10 of vsearch (exact) against the ground truth, $BATCH queries"
sql "SELECT 'recall@10: ' || (COUNT(*) / ($BATCH * 10.0))::NUMERIC(6,4) FROM
     (SELECT vvector.vsearch($V USING PARAMETERS index_name='$IX', k=10) OVER()
      FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.sift_query WHERE qid < $BATCH) x) r
     JOIN $SCHEMA.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10"
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
echo; echo "saved to $OUT"
