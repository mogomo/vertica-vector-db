#!/usr/bin/env bash
# Scale test: one data set, three indexes (flat, HNSW, HNSW with sq8), measured end to end.
#
#   scripts/scale.sh --dataset=NAME [--schema=VVSCALE] [--dir=DIR [--rows=N] [--gt=FILE] | --generate=DIMS --rows=N]
#                    [--streams=N] [--indexes=flat,hnsw,sq8] [--metric=l2] [--parts=...] [--queries=1000]
#                    [--runs=200] [--modes=yes,mixed] [--echo_only]
#
#   --dataset   the tables <schema>.<NAME>_base, _query and _gt of scripts/load_dataset.sh
#   --dir, --rows, --gt, --generate, --streams
#               load the data first with scripts/load_dataset.sh (same meaning there); without --dir
#               and --generate the tables must exist
#   --indexes   which indexes to build, named <NAME>_flat, <NAME>_hnsw, <NAME>_sq8 (HNSW m 16,
#               ef_construction 200; sq8 = HNSW with int8 codes)
#   --metric    l2 (default), cosine, dot or l1
#   --parts     any of build,recall,latency,batch,incremental (default all):
#               build        register, vvector.sizing, full refresh_index of each index (vbuild and
#                            vload statement times), manifest, vinfo on every node
#               recall       recall@10 of every precision level over --queries queries: against the
#                            ground truth when <NAME>_gt has rows, else against precision exact
#               latency      scripts/latency.sh shapes snap, dual, delta0, vknn per index and mode
#               batch        --queries queries in one statement per index and precision, per mode
#               incremental  up to 1000 adds (the first query vectors as new ids) and 500 deletes, one
#                            refresh_index per index, then one refresh with nothing changed
#   --modes     deploy modes of the latency and batch parts (scripts/deploy.sh --fenced=...); the
#               library is deployed fenced again at the end
#
# Prints the results, also saved to build/scale-<NAME>-<date>.txt. Not part of make test: a 100M-row
# run takes hours. Check the memory first: a full build holds the whole snapshot (vvector.sizing).
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/.."

NAME= SCHEMA=VVSCALE DIR= ROWS= GT= GEN= STREAMS= INDEXES=flat,hnsw,sq8 METRIC=l2
PARTS=build,recall,latency,batch,incremental QUERIES=1000 RUNS=200 MODES=yes,mixed ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --dataset=*)  NAME="${arg#*=}" ;;
        --schema=*)   SCHEMA="${arg#*=}" ;;
        --dir=*)      DIR="${arg#*=}" ;;
        --rows=*)     ROWS="${arg#*=}" ;;
        --gt=*)       GT="${arg#*=}" ;;
        --generate=*) GEN="${arg#*=}" ;;
        --streams=*)  STREAMS="${arg#*=}" ;;
        --indexes=*)  INDEXES="${arg#*=}" ;;
        --metric=*)   METRIC="${arg#*=}" ;;
        --parts=*)    PARTS="${arg#*=}" ;;
        --queries=*)  QUERIES="${arg#*=}" ;;
        --runs=*)     RUNS="${arg#*=}" ;;
        --modes=*)    MODES="${arg#*=}" ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "scale.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
die() { echo "scale.sh: $*" >&2; exit 2; }
[ -n "$NAME" ] || die "--dataset is required"
case "$NAME$SCHEMA" in *[!A-Za-z0-9_]*) die "--dataset and --schema must be letters, digits or underscores" ;; esac
case "$QUERIES$RUNS" in *[!0-9]*) die "--queries and --runs must be numbers" ;; esac
case "$METRIC" in l2|cosine|dot|l1) ;; *) die "--metric must be l2, cosine, dot or l1" ;; esac
for x in ${INDEXES//,/ }; do case "$x" in flat|hnsw|sq8) ;; *) die "--indexes: flat, hnsw, sq8" ;; esac; done
for x in ${PARTS//,/ }; do case "$x" in build|recall|latency|batch|incremental) ;; *) die "--parts: build, recall, latency, batch, incremental" ;; esac; done
for x in ${MODES//,/ }; do case "$x" in yes|no|mixed) ;; *) die "--modes: yes, no, mixed" ;; esac; done
[ "$QUERIES" -ge 1 ] || die "--queries must be at least 1"

T="$SCHEMA.$NAME" V="qid, qvec, id, vec, del, ver, snapshot_id" TAG="vvscale$(date +%s)"
has() { case ",$1," in *",$2,"*) return 0 ;; esac; return 1; }
ix() { echo "${NAME}_$1"; }
LOAD=()
if [ -n "$GEN" ]; then LOAD=(--dataset="$NAME" --schema="$SCHEMA" --generate="$GEN" --rows="$ROWS" --queries="$QUERIES")
elif [ -n "$DIR" ]; then LOAD=(--dataset="$NAME" --schema="$SCHEMA" --dir="$DIR" ${ROWS:+--rows=$ROWS} ${GT:+--gt=$GT})
fi
[ ${#LOAD[@]} -eq 0 ] || [ -z "$STREAMS" ] || LOAD+=(--streams="$STREAMS")

if [ "$ECHO_ONLY" = yes ]; then
    [ ${#LOAD[@]} -eq 0 ] || scripts/load_dataset.sh "${LOAD[@]}" --echo_only
    for x in ${INDEXES//,/ }; do
        kind=hnsw; [ "$x" = flat ] && kind=flat
        echo "CALL vvector.register_index('$(ix $x)', '${T}_base', 'id', 'vec', 'del', 'ts', '$METRIC', 0, '$kind');"
        [ "$x" = sq8 ] && echo "CALL vvector.set_index_options('$(ix $x)', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
        has "$PARTS" build && echo "CALL vvector.refresh_index('$(ix $x)', 'full');"
    done
    has "$PARTS" latency && echo "scripts/latency.sh --index=<each index> --schema=$SCHEMA --runs=$RUNS --shapes=snap,dual,delta0,vknn   (modes $MODES)"
    has "$PARTS" incremental && echo "INSERT 1000 adds and 500 deletes into ${T}_base; CALL vvector.refresh_index(<each index>);"
    exit 0
fi

mkdir -p build
OUT="build/scale-$NAME-$(date -u +%Y%m%d-%H%M%S).txt"
exec > >(tee "$OUT") 2>&1

sql() { vsql -X -A -t -q -v ON_ERROR_STOP=1 -c "$1"; }
now() { date -u +%H:%M:%S; }
secs() { awk -v ns=$(( $(date +%s%N) - $1 )) 'BEGIN {printf "%.2f", ns / 1e9}'; }
last() {   # LABEL [SINCE]: the duration of the latest statement with that label (since SINCE), or -
    local ms
    ms=$(sql "SELECT request_duration_ms FROM v_monitor.query_requests WHERE request_label = '$1'${2:+ AND start_timestamp >= '$2'} ORDER BY start_timestamp DESC LIMIT 1")
    echo "${ms:--}"
}
mem() { echo "$(now) free memory per node (GB, without the page cache): $(sql "SELECT (total_memory_free_bytes / 1073741824.0)::NUMERIC(6,1) FROM v_monitor.host_resources ORDER BY host_name" | tr '\n' ' ')"; }
refresh_timed() {   # INDEX [MODE]
    local t0 out since
    since=$(sql "SELECT CLOCK_TIMESTAMP()")
    t0=$(date +%s%N)
    if ! out=$(sql "CALL vvector.refresh_index('$1'${2:+, '$2'});" 2>&1); then
        echo "$(now) refresh_index $1${2:+ $2} FAILED:"; tail -3 <<< "$out"; return 1
    fi
    echo "$(now) refresh_index $1${2:+ $2}: $(secs "$t0") s (vbuild statement $(last vvector_build "$since") ms, vload $(last vvector_load "$since") ms)"
    echo "    $(sql "SELECT refresh_note FROM vvector.manifest WHERE index_name = '$1'" | cut -c1-240)"
}

echo "== scale test $NAME, schema $SCHEMA, $(now) UTC, $(sql "SELECT COUNT(*) FROM nodes WHERE node_state = 'UP'") nodes, $(sql "SELECT version()")"
if [ ${#LOAD[@]} -gt 0 ]; then
    echo "== load"
    t0=$(date +%s%N)
    scripts/load_dataset.sh "${LOAD[@]}" 2>&1 | grep -v NOTICE || { echo "load failed"; exit 1; }
    echo "$(now) loaded in $(secs "$t0") s"
fi
ROWS_IN=$(sql "SELECT COUNT(*) FROM ${T}_base") || { echo "scale.sh: ${T}_base not found"; exit 1; }
DIMS=$(sql "SELECT MAX(ARRAY_LENGTH(vec)) FROM ${T}_base WHERE ARRAY_LENGTH(vec) IS NOT NULL")
NQ=$(sql "SELECT COUNT(*) FROM ${T}_query WHERE qid < $QUERIES")
NGT=$(sql "SELECT COUNT(*) FROM ${T}_gt" 2>/dev/null || echo 0)
echo "journal ${T}_base: $ROWS_IN rows of $DIMS dimensions; $NQ queries; ground truth rows $NGT; metric $METRIC"
mem

if has "$PARTS" build; then
    echo; echo "== build (deploy fenced)"
    scripts/deploy.sh --fenced=yes > /dev/null 2>&1 || { echo "deploy failed"; exit 1; }
    for x in ${INDEXES//,/ }; do
        kind=hnsw quant=none
        [ "$x" = flat ] && kind=flat
        [ "$x" = sq8 ] && quant=sq8
        echo "-- $(ix $x): vvector.sizing($ROWS_IN, $DIMS, '$kind', '$quant')"
        sql "CALL vvector.sizing($ROWS_IN, $DIMS, '$kind', '$quant');" 2>&1 | sed -n 's/^.*NOTICE [0-9]*: *//p' | sed 's/^/    /'
        sql "CALL vvector.unregister_index('$(ix $x)');" > /dev/null 2>&1 || true
        sql "CALL vvector.register_index('$(ix $x)', '${T}_base', 'id', 'vec', 'del', 'ts', '$METRIC', 0, '$kind');" > /dev/null 2>&1 || { echo "register_index $(ix $x) failed"; exit 1; }
        [ "$quant" = sq8 ] && { sql "CALL vvector.set_index_options('$(ix $x)', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);" > /dev/null 2>&1 || { echo "set_index_options $(ix $x) failed"; exit 1; }; }
        refresh_timed "$(ix $x)" full || exit 1
        mem
    done
    echo "-- manifest"
    sql "SELECT index_name, vector_count, dims, index_type, quantization, graph_bytes // 1048576 AS graph_mb,
                index_bytes // 1048576 AS index_mb, build_seconds FROM vvector.manifest
         WHERE index_name LIKE '${NAME}\\_%' ORDER BY 1" | column -t -s'|'
    echo "-- vinfo on every node"
    sql "SELECT node_name, index_name, snapshot_id, vector_count, index_type, quantization, loaded
         FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) i
         WHERE index_name LIKE '${NAME}\\_%' ORDER BY 2, 1" | column -t -s'|'
fi

QS="SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM ${T}_query WHERE qid < $QUERIES"
levels() { [ "$1" = flat ] && echo "exact" || echo "fast balanced best exact"; }

if has "$PARTS" recall; then
    echo; echo "== recall@10, $NQ queries"
    if [ "$NGT" -gt 0 ]; then
        REF="SELECT qid, id FROM ${T}_gt WHERE rank <= 10"
        echo "reference: the ground truth"
    else
        ref_ix=$(ix "${INDEXES%%,*}")
        sql "DROP TABLE IF EXISTS ${T}_ref; CREATE TABLE ${T}_ref AS SELECT vvector.vsearch($V USING PARAMETERS index_name='$ref_ix', k=10, precision='exact') OVER() FROM (SELECT * FROM $SCHEMA.${ref_ix}_snap UNION ALL $QS) x;" > /dev/null 2>&1 || { echo "reference search failed"; exit 1; }
        REF="SELECT qid, id FROM ${T}_ref"
        echo "reference: precision exact on $ref_ix (table ${T}_ref)"
    fi
    for x in ${INDEXES//,/ }; do
        for p in $(levels $x); do
            t0=$(date +%s%N)
            r=$(sql "SELECT (COUNT(*) / ($NQ * 10.0))::NUMERIC(6,4) FROM
                     (SELECT vvector.vsearch($V USING PARAMETERS index_name='$(ix $x)', k=10, precision='$p') OVER()
                      FROM (SELECT * FROM $SCHEMA.$(ix $x)_snap UNION ALL $QS) x) r
                     JOIN ($REF) g ON g.qid = r.qid AND g.id = r.id")
            printf "%-34s %-9s %8s   (statement %s s)\n" "$(ix $x)" "$p" "$r" "$(secs "$t0")"
        done
    done
fi

if has "$PARTS" latency || has "$PARTS" batch; then
    for m in ${MODES//,/ }; do
        echo; echo "== FENCED=$m"
        scripts/deploy.sh --fenced="$m" > /dev/null 2>&1 || { echo "deploy $m failed"; continue; }
        if has "$PARTS" latency; then
            for x in ${INDEXES//,/ }; do
                echo "-- single query, $(ix $x) (precision balanced; client and server ms)"
                scripts/latency.sh --index="$(ix $x)" --schema="$SCHEMA" --queries="${T}_query" --runs="$RUNS" --shapes=snap,dual,delta0,vknn 2>&1 | grep -v NOTICE
            done
        fi
        if has "$PARTS" batch; then
            echo "-- $NQ queries in one statement (server ms, median of 3)"
            for x in ${INDEXES//,/ }; do
                for p in $(levels $x); do
                    [ "$p" = exact ] && [ "$x" != flat ] && continue
                    label="${TAG}_${x}_${p}_$m"
                    for i in 1 2 3; do
                        sql "SELECT /*+LABEL($label)*/ COUNT(*) FROM (SELECT vvector.vsearch($V USING PARAMETERS index_name='$(ix $x)', k=10, precision='$p') OVER()
                             FROM (SELECT * FROM $SCHEMA.$(ix $x)_snap UNION ALL $QS) x) r" > /dev/null || break
                    done
                    ms=$(sql "SELECT MEDIAN(request_duration_ms) OVER() FROM v_monitor.query_requests WHERE request_label = '$label' LIMIT 1")
                    ms=${ms%.*}
                    printf "%-34s %-9s %9s ms  (%s queries/s)\n" "$(ix $x)" "$p" "$ms" "$(( NQ * 1000 / (ms > 0 ? ms : 1) ))"
                done
            done
        fi
    done
    scripts/deploy.sh --fenced=yes > /dev/null 2>&1
fi

if has "$PARTS" incremental; then
    echo; echo "== incremental refresh: $(sql "SELECT COUNT(*) FROM ${T}_query WHERE qid < 1000") adds (query vectors as new ids) and 500 deletes"
    # New ids above the largest; every run deletes the next 500 ids of the loaded range.
    top=$(sql "SELECT MAX(id) FROM ${T}_base") gone=$(sql "SELECT COUNT(*) FROM ${T}_base WHERE del")
    sql "INSERT INTO ${T}_base (id, vec) SELECT $top + 1 + qid, qvec FROM ${T}_query WHERE qid < 1000;
         INSERT INTO ${T}_base (id, del) SELECT id, TRUE FROM ${T}_base WHERE id >= $gone AND id < $gone + 500 AND NOT del; COMMIT;" > /dev/null || exit 1
    for x in ${INDEXES//,/ }; do refresh_timed "$(ix $x)"; done
    echo "-- nothing changed"
    for x in ${INDEXES//,/ }; do refresh_timed "$(ix $x)"; done
    mem
fi
echo; echo "== done $(now) UTC; report: $OUT"
