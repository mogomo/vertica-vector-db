#!/usr/bin/env bash
# Load a benchmark data set of vector files (TEXMEX format: .fvecs, .ivecs, .bvecs) into Vertica,
# or generate one.
#
#   scripts/load_dataset.sh --dataset=NAME --dir=DIR [--schema=VVBENCH] [--rows=N] [--gt=FILE]
#                           [--streams=N] [--echo_only]
#   scripts/load_dataset.sh --dataset=NAME --generate=DIMS --rows=N [--queries=1000] [--seed=1]
#                           [--schema=VVBENCH] [--streams=N] [--echo_only]
#
#   --dataset  file name prefix: DIR/NAME_base.fvecs (or .bvecs), DIR/NAME_query.fvecs (or .bvecs)
#              and DIR/NAME_groundtruth.ivecs (SIFT1M: --dataset=sift --dir=<where sift.tar.gz was unpacked>)
#   --rows     load only the first N base vectors (the ground truth file then does not apply and is
#              skipped, unless --gt names one for these N vectors, for example idx_100M.ivecs of BIGANN)
#   --gt       the ground truth file (.ivecs) to load instead of DIR/NAME_groundtruth.ivecs
#   --streams  parallel COPY streams for the base vectors (default: 1 per 2 cores, 1 to 16); one
#              fvecs process and one vsql session each, every stream loads its own range of rows
#   --generate generated vectors of DIMS elements instead of files (build/tools/fvecs gen: a mixture
#              of 1000 Gaussian clusters, seeded, the same rows for every --streams); --queries query
#              vectors from the same clusters; no ground truth. For memory, speed and scale tests:
#              the recall of generated data says little about real embeddings.
#
# Creates, replacing earlier ones:
#   <schema>.<NAME>_base  (id INT, vec ARRAY[FLOAT], del BOOLEAN, ts TIMESTAMPTZ): a journal table,
#                          ids = position in the file from 0, partitioned by the date of ts
#   <schema>.<NAME>_query (qid INT, qvec ARRAY[FLOAT]): qid = position from 0
#   <schema>.<NAME>_gt    (qid INT, rank INT, id INT): the true nearest neighbours, rank from 1
# Needs build/tools/fvecs (make tools). Run it where the files are.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

NAME= DIR= SCHEMA=VVBENCH ROWS= GT= STREAMS= GEN= QUERIES=1000 SEED=1 ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --dataset=*)  NAME="${arg#*=}" ;;
        --dir=*)      DIR="${arg#*=}" ;;
        --schema=*)   SCHEMA="${arg#*=}" ;;
        --rows=*)     ROWS="${arg#*=}" ;;
        --gt=*)       GT="${arg#*=}" ;;
        --streams=*)  STREAMS="${arg#*=}" ;;
        --generate=*) GEN="${arg#*=}" ;;
        --queries=*)  QUERIES="${arg#*=}" ;;
        --seed=*)     SEED="${arg#*=}" ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "load_dataset.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
die() { echo "load_dataset.sh: $*" >&2; exit 2; }
[ -n "$NAME" ] || die "--dataset is required"
case "$NAME$SCHEMA" in *[!A-Za-z0-9_]*) die "--dataset and --schema must be letters, digits or underscores" ;; esac
for v in "$ROWS" "$STREAMS" "$GEN" "$QUERIES" "$SEED"; do
    case "$v" in *[!0-9]*) die "--rows, --streams, --generate, --queries and --seed must be numbers" ;; esac
done
GT_NAMED=${GT:+yes}
if [ -n "$GEN" ]; then
    [ -z "$DIR" ] && [ -z "$GT" ] || die "--generate takes no --dir and no --gt"
    [ -n "$ROWS" ] && [ "$ROWS" -gt 0 ] || die "--generate needs --rows=N"
    [ "$GEN" -ge 1 ] && [ "$GEN" -le 32768 ] || die "--generate: 1 to 32768 dimensions"
else
    [ -n "$DIR" ] || die "--dir (or --generate) is required"
fi
if [ -z "$STREAMS" ]; then
    STREAMS=$(( $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2) / 2 ))
    [ "$STREAMS" -ge 1 ] || STREAMS=1
    [ "$STREAMS" -le 16 ] || STREAMS=16
fi
[ "$STREAMS" -ge 1 ] && [ "$STREAMS" -le 64 ] || die "--streams: 1 to 64"

FVECS=build/tools/fvecs
T="$SCHEMA.${NAME}"
pick() {   # the .fvecs file, else the .bvecs file of DIR/NAME_PART
    if [ -e "$DIR/${NAME}_$1.fvecs" ] || [ ! -e "$DIR/${NAME}_$1.bvecs" ]; then echo "$DIR/${NAME}_$1.fvecs"; else echo "$DIR/${NAME}_$1.bvecs"; fi
}
if [ -z "$GEN" ]; then
    BASE=$(pick base) QUERY=$(pick query)
    if [ -z "$GT" ] && [ -z "$ROWS" ]; then GT="$DIR/${NAME}_groundtruth.ivecs"; fi
fi

CREATE="CREATE SCHEMA IF NOT EXISTS $SCHEMA;
DROP TABLE IF EXISTS ${T}_base CASCADE; DROP TABLE IF EXISTS ${T}_query CASCADE; DROP TABLE IF EXISTS ${T}_gt CASCADE;
CREATE TABLE ${T}_base (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                        ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
CREATE TABLE ${T}_query (qid INT NOT NULL, qvec ARRAY[FLOAT]) ORDER BY qid UNSEGMENTED ALL NODES;
CREATE TABLE ${T}_gt (qid INT NOT NULL, rank INT NOT NULL, id INT NOT NULL) ORDER BY qid, rank UNSEGMENTED ALL NODES;"
COPY_BASE="COPY ${T}_base (id, vec) FROM STDIN DELIMITER '|' ABORT ON ERROR DIRECT"
COPY_QUERY="COPY ${T}_query (qid, qvec) FROM STDIN DELIMITER '|' ABORT ON ERROR DIRECT"
COPY_GT="COPY ${T}_gt (qid, rank, id) FROM STDIN DELIMITER '|' ABORT ON ERROR DIRECT"

# The number of base vectors: --rows, else the file size over the record size (4 + dims x value bytes).
count_vectors() {
    local f=$1 d vb=4 size
    case "$f" in *.bvecs) vb=1 ;; esac
    d=$(od -An -t d4 -N4 "$f" | tr -d ' ')
    size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
    echo $(( size / (4 + d * vb) ))
}
source_rows() {   # FIRST LIMIT: the text rows of one stream
    if [ -n "$GEN" ]; then "$FVECS" gen --dims="$GEN" --seed="$SEED" --first="$1" --limit="$2"
    else "$FVECS" rows "$BASE" --first="$1" --limit="$2"; fi
}
query_rows() {
    if [ -n "$GEN" ]; then "$FVECS" gen --dims="$GEN" --seed="$SEED" --limit="$QUERIES" --queries
    else "$FVECS" rows "$QUERY"; fi
}

if [ "$ECHO_ONLY" = yes ]; then
    echo "vsql -X -c \"$CREATE\""
    echo "# $STREAMS streams, each:"
    if [ -n "$GEN" ]; then
        echo "$FVECS gen --dims=$GEN --seed=$SEED --first=FIRST --limit=COUNT | vsql -X -c \"$COPY_BASE\""
        echo "$FVECS gen --dims=$GEN --seed=$SEED --limit=$QUERIES --queries | vsql -X -c \"$COPY_QUERY\""
    else
        echo "$FVECS rows $BASE --first=FIRST --limit=COUNT | vsql -X -c \"$COPY_BASE\""
        echo "$FVECS rows $QUERY | vsql -X -c \"$COPY_QUERY\""
        [ -n "$GT" ] && echo "$FVECS gt $GT --k=100 | vsql -X -c \"$COPY_GT\""
    fi
    exit 0
fi

[ -x "$FVECS" ] || { echo "load_dataset.sh: $FVECS not found: run make tools" >&2; exit 1; }
if [ -z "$GEN" ]; then
    for f in "$BASE" "$QUERY"; do [ -r "$f" ] || { echo "load_dataset.sh: $f not found" >&2; exit 1; }; done
    total=$(count_vectors "$BASE")
    [ -z "$ROWS" ] || [ "$ROWS" -le "$total" ] || { echo "load_dataset.sh: --rows=$ROWS but $BASE holds $total vectors" >&2; exit 1; }
    [ -n "$ROWS" ] || ROWS=$total
    if [ -n "$GT" ] && [ ! -r "$GT" ]; then
        # The default ground truth file is optional; a named one is not.
        [ -z "$GT_NAMED" ] || { echo "load_dataset.sh: $GT not found" >&2; exit 1; }
        GT=
    fi
fi

vsql -X -q -v ON_ERROR_STOP=1 -c "$CREATE" > /dev/null
start=$(date +%s)
per=$(( (ROWS + STREAMS - 1) / STREAMS )) pids=() first=0
while [ "$first" -lt "$ROWS" ]; do
    n=$(( ROWS - first < per ? ROWS - first : per ))
    ( set -o pipefail; source_rows "$first" "$n" | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_BASE" ) &
    pids+=($!)
    first=$(( first + n ))
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
[ "$failed" -eq 0 ] || { echo "load_dataset.sh: a COPY stream of the base vectors failed" >&2; exit 1; }
query_rows | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_QUERY"
if [ -n "$GT" ]; then
    "$FVECS" gt "$GT" --k=100 | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_GT"
fi
vsql -X -A -t -c "SELECT '${NAME}: ' || (SELECT COUNT(*) FROM ${T}_base) || ' base vectors of ' ||
                          (SELECT MAX(ARRAY_LENGTH(vec)) FROM ${T}_base) || ' dimensions, ' ||
                          (SELECT COUNT(*) FROM ${T}_query) || ' queries, ' ||
                          (SELECT COUNT(*) FROM ${T}_gt) || ' ground truth rows, loaded in $(( $(date +%s) - start )) s ($STREAMS streams)'"
