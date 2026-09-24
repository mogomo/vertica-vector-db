#!/usr/bin/env bash
# Benchmark on SIFT1M (1,000,000 vectors of 128 dimensions, 10,000 queries, ground truth):
# the engine alone, and the SQL statements in every deploy mode.
#
#   scripts/benchmark.sh --data_dir=DIR [--schema=VVBENCH] [--modes=yes,no,mixed] [--runs=200]
#                        [--batch=1000] [--skip_load] [--hnswlib=DIR] [--parts=LIST] [--echo_only]
#
#   --data_dir   where sift_base.fvecs, sift_query.fvecs and sift_groundtruth.ivecs are
#   --modes      deploy modes to measure (FENCED=yes|no|mixed); deploys fenced again at the end
#   --runs       repetitions of each single-query statement (scripts/latency.sh)
#   --skip_load  keep the tables loaded by an earlier run
#   --parts      what to run, default all: engine,refresh,incremental,search,recall
#
#   --hnswlib    a clone of github.com/nmslib/hnswlib: the engine benchmark also runs hnswlib
#
# Measures: engine (make bench: memory bandwidth, flat search 1 query and batches, build; HNSW
# build and search per ef_search, the same with sq8 codes, and hnswlib the same way when --hnswlib is
# given); refresh of the flat index sift, the HNSW index sift_hnsw and the HNSW index with sq8 codes
# sift_sq8 (build, load); incremental refresh against the size
# of the change (a flat and an HNSW index on a copy of the first 900,000 vectors, changes of 0 to
# 50,000 adds plus half as many deletes, then a full build for comparison); per mode: the SQL full scan of one
# query (ORDER BY VECTOR_L2 LIMIT 10), the single-query shapes of scripts/latency.sh on the three
# indexes (vsearch and vknn), batches of queries in one statement, and recall@10 against the
# ground truth (flat, and HNSW with and without sq8 at every precision level). Prints the results;
# also saved to build/benchmark-<date>.txt.
# Needs make, make tools and a deployed library. Creates schema VVBENCH (tables sift_*) and
# indexes sift, sift_hnsw and sift_sq8 (the incremental part makes and drops sift_inc, sift_inc_f and
# sift_inc_h). Takes 40 to 50 minutes on 8 cores.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR= SCHEMA=VVBENCH MODES=yes,no,mixed RUNS=200 BATCH=1000 SKIP_LOAD=no HNSWLIB= ECHO_ONLY=no PARTS=engine,refresh,incremental,search,recall
for arg in "$@"; do
    case "$arg" in
        --data_dir=*) DATA_DIR="${arg#*=}" ;;
        --schema=*)   SCHEMA="${arg#*=}" ;;
        --modes=*)    MODES="${arg#*=}" ;;
        --runs=*)     RUNS="${arg#*=}" ;;
        --batch=*)    BATCH="${arg#*=}" ;;
        --skip_load)  SKIP_LOAD=yes ;;
        --hnswlib=*)  HNSWLIB="${arg#*=}" ;;
        --parts=*)    PARTS="${arg#*=}" ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "benchmark.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$DATA_DIR" ] || { echo "benchmark.sh: --data_dir is required" >&2; exit 2; }
case "$SCHEMA$RUNS$BATCH" in *[!A-Za-z0-9_]*) echo "benchmark.sh: bad --schema, --runs or --batch" >&2; exit 2 ;; esac

IX=sift HX=sift_hnsw QX=sift_sq8
part() { [[ ",$PARTS," == *",$1,"* ]]; }
OUT="build/benchmark-$(date +%Y%m%d-%H%M%S).txt"
if [ "$ECHO_ONLY" = yes ]; then
    echo "scripts/load_dataset.sh --dataset=sift --dir=$DATA_DIR --schema=$SCHEMA"
    echo "CALL vvector.register_index('$IX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'flat'); CALL vvector.refresh_index('$IX');"
    echo "CALL vvector.register_index('$HX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw'); CALL vvector.refresh_index('$HX');"
    echo "CALL vvector.register_index('$QX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw'); CALL vvector.set_index_options('$QX', NULL, NULL, NULL, 'sq8', ...); CALL vvector.refresh_index('$QX');"
    echo "make bench DATA_DIR=$DATA_DIR${HNSWLIB:+ HNSWLIB_DIR=$HNSWLIB}"
    echo "incremental: $SCHEMA.sift_inc = first 900000 rows of sift_base; indexes sift_inc_f (flat), sift_inc_h (hnsw); changes of 0, 100, 1000, 10000, 50000 adds and half as many deletes; refresh_index after each; refresh_index(..., 'full')"
    for m in ${MODES//,/ }; do echo "scripts/deploy.sh --fenced=$m; full scan; scripts/latency.sh --index=$IX, --index=$HX and --index=$QX --schema=$SCHEMA --runs=$RUNS; batches of $BATCH; recall@10"; done
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

if part engine; then
echo; echo "== engine alone (no Vertica): make bench"
make -s bench DATA_DIR="$DATA_DIR" ${HNSWLIB:+HNSWLIB_DIR="$HNSWLIB"} 2>&1 | grep -vE '^(g\+\+|c\+\+|==)'
fi

TAG="vvbench$(date +%s)"
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
if part refresh; then
echo; echo "== refresh of the index (FENCED=yes)"
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

echo; echo "== refresh of the HNSW index with sq8 codes (quantization sq8, FENCED=yes)"
sql "CALL vvector.unregister_index('$QX');" > /dev/null 2>&1 || true
sql "CALL vvector.register_index('$QX', '$SCHEMA.sift_base', 'id', 'vec', 'del', 'ts', 'l2', NULL, 'hnsw');
     CALL vvector.set_index_options('$QX', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);" > /dev/null 2>&1
start=$(date +%s%N)
sql "CALL vvector.refresh_index('$QX');" > /dev/null 2>&1
printf "%-58s %12.3f s\n" "refresh_index, 1M x 128, HNSW with sq8 (all steps)" "$(awk -v ns=$(( $(date +%s%N) - start )) 'BEGIN {print ns / 1e9}')"
printf "%-58s %12d ms\n" "  vbuild statement (consolidation, graph, codes, chunks)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_build' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12d ms\n" "  vload statement (write and verify the cache file)" "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_load' ORDER BY start_timestamp DESC LIMIT 1")"
printf "%-58s %12s MB\n" "  snapshot size (graph and codes included)" "$(sql "SELECT index_bytes // 1048576 FROM vvector.manifest WHERE index_name = '$QX'")"
fi

if part incremental; then
echo; echo "== incremental refresh against the size of the change (FENCED=yes)"
sql "CALL vvector.unregister_index('sift_inc_f');" > /dev/null 2>&1 || true
sql "CALL vvector.unregister_index('sift_inc_h');" > /dev/null 2>&1 || true
sql "DROP TABLE IF EXISTS $SCHEMA.sift_inc CASCADE;
CREATE TABLE $SCHEMA.sift_inc (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                               ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.sift_inc (id, vec, ts) SELECT id, vec, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM $SCHEMA.sift_base WHERE id < 900000;
COMMIT;" > /dev/null
refresh_timed() {   # INDEX [MODE] -> one line: total, journal statement (verify = recompute the digest up to the boundary,
                    # digest = carry it forward from the new rows, or take it for a full build), vbuild, vload (none when nothing changed), what was built
    local start total loads
    loads=$(sql "SELECT COUNT(*) FROM v_monitor.query_requests WHERE request_label = 'vvector_load'")
    start=$(date +%s%N)
    sql "CALL vvector.refresh_index('$1'${2:+, '$2'});" > /dev/null 2>&1
    total=$(( ($(date +%s%N) - start) / 1000000 ))
    printf "  %-12s %9d ms  %-16s ms  vbuild %7s ms  vload %6s ms  %s\n" "$1" "$total" \
        "$(sql "SELECT SUBSTR(request_label, 9) || ' ' || request_duration_ms FROM v_monitor.query_requests WHERE request_label IN ('vvector_digest', 'vvector_verify') ORDER BY start_timestamp DESC LIMIT 1")" \
        "$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = 'vvector_build' ORDER BY start_timestamp DESC LIMIT 1")" \
        "$(sql "SELECT CASE WHEN COUNT(*) > $loads THEN MAX(CASE WHEN n = 1 THEN request_duration_ms END)::VARCHAR ELSE '-' END
                FROM (SELECT request_duration_ms, ROW_NUMBER() OVER(ORDER BY start_timestamp DESC) AS n
                      FROM v_monitor.query_requests WHERE request_label = 'vvector_load') l")" \
        "$(sql "SELECT REGEXP_REPLACE(refresh_note, ', [0-9.]+ seconds$', '') FROM vvector.manifest WHERE index_name = '$1'" | cut -c1-120)"
}
sql "CALL vvector.register_index('sift_inc_f', '$SCHEMA.sift_inc', 'id', 'vec', 'del', 'ts', 'l2', 0, 'flat');" > /dev/null 2>&1
sql "CALL vvector.register_index('sift_inc_h', '$SCHEMA.sift_inc', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');" > /dev/null 2>&1
echo "first build, 900,000 vectors:"
refresh_timed sift_inc_f
refresh_timed sift_inc_h
next=900000 gone=0
for adds in 0 100 1000 10000 50000; do
    dels=$((adds / 2))
    echo "change of $adds adds and $dels deletes:"
    if [ "$adds" -gt 0 ]; then
        sql "INSERT INTO $SCHEMA.sift_inc (id, vec) SELECT id, vec FROM $SCHEMA.sift_base WHERE id >= $next AND id < $((next + adds));
             INSERT INTO $SCHEMA.sift_inc (id, del) SELECT id, TRUE FROM $SCHEMA.sift_base WHERE id >= $gone AND id < $((gone + dels)); COMMIT;" > /dev/null
        next=$((next + adds)) gone=$((gone + dels))
    fi
    refresh_timed sift_inc_f
    refresh_timed sift_inc_h
    if [ "$adds" -eq 0 ]; then
        echo "no change again, verify_every 0 (the digest carried forward, never recomputed):"
        for x in sift_inc_f sift_inc_h; do
            sql "CALL vvector.set_index_options('$x', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0);" > /dev/null 2>&1
            refresh_timed $x
            sql "CALL vvector.set_index_options('$x', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);" > /dev/null 2>&1
        done
    fi
done
echo "full build of the same vectors, for comparison:"
refresh_timed sift_inc_f full
refresh_timed sift_inc_h full
sql "CALL vvector.unregister_index('sift_inc_f');" > /dev/null 2>&1 || true
sql "CALL vvector.unregister_index('sift_inc_h');" > /dev/null 2>&1 || true
sql "DROP TABLE IF EXISTS $SCHEMA.sift_inc CASCADE;" > /dev/null
fi

Q0="SELECT qvec FROM $SCHEMA.sift_query WHERE qid = 0"
V="qid, qvec, id, vec, del, ver, snapshot_id"
part search && for m in ${MODES//,/ }; do
    echo; echo "== FENCED=$m"
    scripts/deploy.sh --fenced="$m" > /dev/null 2>&1
    if [ "$m" = "${MODES%%,*}" ]; then
        for i in 1 2 3; do sql "SELECT /*+LABEL(${TAG}_scan)*/ b.id, VECTOR_L2(b.vec, q.qvec) AS d FROM $SCHEMA.sift_base b CROSS JOIN ($Q0) q ORDER BY d LIMIT 10" > /dev/null; done
        printf "%-58s %12s ms\n" "SQL full scan, 1 query (ORDER BY VECTOR_L2 LIMIT 10), median of 3" \
            "$(sql "SELECT APPROXIMATE_PERCENTILE(request_duration_ms USING PARAMETERS percentile=0.5)::INT FROM v_monitor.query_requests WHERE request_label = '${TAG}_scan'")"
    fi
    echo "flat index $IX:"
    scripts/latency.sh --index="$IX" --schema="$SCHEMA" --runs="$RUNS" --shapes=select1,vversion,snap,delta0,threads1 2>&1 | grep -v NOTICE
    echo "HNSW index $HX (precision balanced, the default):"
    scripts/latency.sh --index="$HX" --schema="$SCHEMA" --runs="$RUNS" --shapes=snap,dual,delta0,vknn,vknnrow 2>&1 | grep -v NOTICE
    echo "HNSW index with sq8 codes $QX (precision balanced: 2 x k rescored):"
    scripts/latency.sh --index="$QX" --schema="$SCHEMA" --runs="$RUNS" --shapes=snap,vknn 2>&1 | grep -v NOTICE
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
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$HX', k=10, precision='fast') OVER() FROM (SELECT * FROM $SCHEMA.${HX}_snap UNION ALL $QS) x"
    batch hnswb "vsearch HNSW (balanced, the default), $BATCH queries in one statement" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$HX', k=10) OVER() FROM (SELECT * FROM $SCHEMA.${HX}_snap UNION ALL $QS) x"
    batch sq8f "vsearch HNSW sq8 (fast: codes only), $BATCH queries" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$QX', k=10, precision='fast') OVER() FROM (SELECT * FROM $SCHEMA.${QX}_snap UNION ALL $QS) x"
    batch sq8b "vsearch HNSW sq8 (balanced: 2 x k rescored), $BATCH queries" \
        "SELECT vvector.vsearch($V USING PARAMETERS index_name='$QX', k=10) OVER() FROM (SELECT * FROM $SCHEMA.${QX}_snap UNION ALL $QS) x"
    batch vknn "vknn HNSW (balanced), $BATCH query rows in one statement" \
        "SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='$HX', k=10) FROM $SCHEMA.sift_query q WHERE q.qid < $BATCH"
done

if part recall; then
echo; echo "== recall@10 against the ground truth, $BATCH queries"
recall() {   # WHAT INDEX PARAMS
    printf "%-58s %12s\n" "$1" "$(sql "SELECT (COUNT(*) / ($BATCH * 10.0))::NUMERIC(6,4) FROM
     (SELECT vvector.vsearch($V USING PARAMETERS index_name='$2', k=10$3) OVER()
      FROM (SELECT * FROM $SCHEMA.${2}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.sift_query WHERE qid < $BATCH) x) r
     JOIN $SCHEMA.sift_gt g ON g.qid = r.qid AND g.id = r.id AND g.rank <= 10")"
}
recall "flat index (exact)" "$IX" ""
for p in fast balanced best exact; do recall "HNSW index, precision $p" "$HX" ", precision='$p'"; done
for p in fast balanced best; do recall "HNSW index with sq8, precision $p" "$QX" ", precision='$p'"; done
fi
scripts/deploy.sh --fenced=yes > /dev/null 2>&1
echo; echo "saved to $OUT"
