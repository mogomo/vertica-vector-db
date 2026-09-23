#!/usr/bin/env bash
# Benchmark on SIFT1M (1,000,000 vectors of 128 dimensions, 10,000 queries, ground truth):
# the engine alone, and the SQL statements in every deploy mode.
#
#   scripts/benchmark.sh --data_dir=DIR [--schema=VVBENCH] [--modes=yes,no,mixed] [--runs=200]
#                        [--batch=1000] [--skip_load] [--hnswlib=DIR] [--echo_only]
#
#   --data_dir   where sift_base.fvecs, sift_query.fvecs and sift_groundtruth.ivecs are
#   --modes      deploy modes to measure (FENCED=yes|no|mixed); deploys fenced again at the end
#   --runs       repetitions of each single-query statement (scripts/latency.sh)
#   --skip_load  keep the tables loaded by an earlier run
#
#   --hnswlib    a clone of github.com/nmslib/hnswlib: the engine benchmark also runs hnswlib
#
# Measures: engine (make bench: memory bandwidth, flat search 1 query and batches, build; HNSW
# build and search per ef_search, and hnswlib the same way when --hnswlib is given); refresh of the
# flat index sift and the HNSW index sift_hnsw (build, load); per mode: the SQL full scan of one
# query (ORDER BY VECTOR_L2 LIMIT 10), the single-query shapes of scripts/latency.sh on both
# indexes (vsearch and vknn), batches of queries in one statement, and recall@10 against the
# ground truth (flat, and HNSW at every precision level). Prints the results; also saved to
# build/benchmark-<date>.txt.
# Needs make, make tools and a deployed library. Creates schema VVBENCH (tables sift_*) and
# indexes sift and sift_hnsw. Takes 30 to 40 minutes on 8 cores.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR= SCHEMA=VVBENCH MODES=yes,no,mixed RUNS=200 BATCH=1000 SKIP_LOAD=no HNSWLIB= ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --data_dir=*) DATA_DIR="${arg#*=}" ;;
        --schema=*)   SCHEMA="${arg#*=}" ;;
        --modes=*)    MODES="${arg#*=}" ;;
        --runs=*)     RUNS="${arg#*=}" ;;
        --batch=*)    BATCH="${arg#*=}" ;;
        --skip_load)  SKIP_LOAD=yes ;;
        --hnswlib=*)  HNSWLIB="${arg#*=}" ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "benchmark.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$DATA_DIR" ] || { echo "benchmark.sh: --data_dir is required" >&2; exit 2; }
case "$SCHEMA$RUNS$BATCH" in *[!A-Za-z0-9_]*) echo "benchmark.sh: bad --schema, --runs or --batch" >&2; exit 2 ;; esac

IX=sift HX=sift_hnsw
OUT="build/benchmark-$(date +%Y%m%d-%H%M%S).txt"
if [ "$ECHO_ONLY" = yes ]; then
    echo "scripts/load_dataset.sh --dataset=sift --dir=$DATA_DIR --schema=$SCHEMA"
    echo "CALL vvector.register_index('$IX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'flat'); CALL vvector.refresh_index('$IX');"
    echo "CALL vvector.register_index('$HX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw'); CALL vvector.refresh_index('$HX');"
    echo "make bench DATA_DIR=$DATA_DIR${HNSWLIB:+ HNSWLIB_DIR=$HNSWLIB}"
    for m in ${MODES//,/ }; do echo "scripts/deploy.sh --fenced=$m; full scan; scripts/latency.sh --index=$IX and --index=$HX --schema=$SCHEMA --runs=$RUNS; batches of $BATCH; recall@10"; done
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
make -s bench DATA_DIR="$DATA_DIR" ${HNSWLIB:+HNSWLIB_DIR="$HNSWLIB"} 2>&1 | grep -vE '^(g\+\+|c\+\+|==)'

TAG="vvbench$(date +%s)"
echo; echo "== refresh of the index (FENCED=yes)"
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
sql "CALL vvector.unregister_index('$IX');" > /dev/null 2>&1 || true
sql "CALL vvector.register_index('$IX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'flat');" > /dev/null 2>&1
start=$(date +%s%N)
sql "CALL vvector.refresh_index('$IX');" > /dev/null 2>&1
printf "%-58s %12.3f s\n" "refresh_index, 1M x 128 (all steps)" "$(awk -v ns=$(( $(date +%s%N) - start )) 'BEGIN {print ns / 1e9}')"
printf "%-58s %12d ms\n" "  vbuild statement (consolidation, build, store chunks)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_build' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12d ms\n" "  vload statement (write and verify the cache file)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_load' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12s MB\n" "  snapshot size" "$(sql "SELECT index_bytes // 1048576 FROM vvector.manifest WHERE index_name = '$IX'")"

echo; echo "== refresh of the HNSW index (m 16, ef_construction 200, FENCED=yes)"
sql "CALL vvector.unregister_index('$HX');" > /dev/null 2>&1 || true
sql "CALL vvector.register_index('$HX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw');" > /dev/null 2>&1
start=$(date +%s%N)
sql "CALL vvector.refresh_index('$HX');" > /dev/null 2>&1
printf "%-58s %12.3f s\n" "refresh_index, 1M x 128, HNSW (all steps)" "$(awk -v ns=$(( $(date +%s%N) - start )) 'BEGIN {print ns / 1e9}')"
printf "%-58s %12d ms\n" "  vbuild statement (consolidation, graph build, chunks)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_build' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12d ms\n" "  vload statement (write and verify the cache file)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_load' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12s MB\n" "  snapshot size (graph included)" "$(sql "SELECT index_bytes // 1048576 FROM vvector.manifest WHERE index_name = '$HX'")"

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
    echo "flat index $IX:"
    scripts/latency.sh --index="$IX" --schema="$SCHEMA" --runs="$RUNS" --shapes=select1,vversion,snap,delta0,threads1 2>&1 | grep -v NOTICE
    echo "HNSW index $HX (precision fast):"
    scripts/latency.sh --index="$HX" --schema="$SCHEMA" --runs="$RUNS" --shapes=snap,dual,delta0,vknn,vknnrow 2>&1 | grep -v NOTICE
    batch() {   # LABEL WHAT SQL
        for i in 1 2 3; do sql "SELECT /*+LABEL(${TAG}_$1_$m)*/ COUNT(*) FROM ($3) r" > /dev/null; done
        local ms
        ms=$(sql "SELECT APPROXIMATE_PERCENTILE(request_duration_ms USING PARAMETERS percentile=0.5)::INT FROM v_monitor.query_requests WHERE request_label = '${TAG}_$1_$m'")
        printf "%-58s %12s ms  (%s queries/s)\n" "$2" "$ms" "$(( BATCH * 1000 / (ms > 0 ? ms : 1) ))"
    }
    QS="SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.sift_query WHERE qid < $BATCH"
    batch flat "vsearch flat, $BATCH queries in one statement, median of 3" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$IX', k=10) OVER() FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL $QS) x"
    batch hnsw "vsearch HNSW (fast), $BATCH queries in one statement" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$HX', k=10) OVER() FROM (SELECT * FROM $SCHEMA.${HX}_snap UNION ALL $QS) x"
    batch hnswb "vsearch HNSW (balanced), $BATCH queries in one statement" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$HX', k=10, precision='balanced') OVER() FROM (SELECT * FROM $SCHEMA.${HX}_snap UNION ALL $QS) x"
    batch vknn "vknn HNSW (fast), $BATCH query rows in one statement" \
        "SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='$HX', k=10) FROM $SCHEMA.sift_query q WHERE q.qid < $BATCH"
done

echo; echo "== recall@10 against the ground truth, $BATCH queries"
recall() {   # WHAT INDEX PARAMS
    printf "%-58s %12s\n" "$1" "$(sql "SELECT (COUNT(*) / ($BATCH * 10.0))::NUMERIC(6,4) FROM
     (SELECT vvector.vsearch($V USING PARAMETERS index_name='$2', k=10$3) OVER()
      FROM (SELECT * FROM $SCHEMA.${2}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.sift_query WHERE qid < $BATCH) x) r
     JOIN $SCHEMA.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10")"
}
recall "flat index (exact)" "$IX" ""
for p in fast balanced best exact; do recall "HNSW index, precision $p" "$HX" ", precision='$p'"; done
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
echo; echo "saved to $OUT"
