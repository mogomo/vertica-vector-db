#!/usr/bin/env bash
# Load a benchmark data set of vector files (TEXMEX format: .fvecs, .ivecs, .bvecs) into Vertica.
#
#   scripts/load_dataset.sh --dataset=NAME --dir=DIR [--schema=VVBENCH] [--rows=N] [--echo_only]
#
#   --dataset  file name prefix: DIR/NAME_base.fvecs, DIR/NAME_query.fvecs and
#              DIR/NAME_groundtruth.ivecs (SIFT1M: --dataset=sift --dir=<where sift.tar.gz was unpacked>)
#   --rows     load only the first N base vectors (the ground truth then does not apply and is skipped)
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

NAME= DIR= SCHEMA=VVBENCH ROWS= ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --dataset=*) NAME="${arg#*=}" ;;
        --dir=*)     DIR="${arg#*=}" ;;
        --schema=*)  SCHEMA="${arg#*=}" ;;
        --rows=*)    ROWS="${arg#*=}" ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "load_dataset.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$NAME" ] && [ -n "$DIR" ] || { echo "load_dataset.sh: --dataset and --dir are required" >&2; exit 2; }
case "$NAME$SCHEMA" in *[!A-Za-z0-9_]*) echo "load_dataset.sh: --dataset and --schema must be letters, digits or underscores" >&2; exit 2 ;; esac
case "$ROWS" in ''|[0-9]*) ;; *) echo "load_dataset.sh: --rows must be a number" >&2; exit 2 ;; esac

FVECS=build/tools/fvecs
BASE="$DIR/${NAME}_base.fvecs" QUERY="$DIR/${NAME}_query.fvecs" GT="$DIR/${NAME}_groundtruth.ivecs"
T="$SCHEMA.${NAME}"
LIMIT=${ROWS:+--limit=$ROWS}

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

if [ "$ECHO_ONLY" = yes ]; then
    echo "vsql -X -c \"$CREATE\""
    echo "$FVECS rows $BASE $LIMIT | vsql -X -c \"$COPY_BASE\""
    echo "$FVECS rows $QUERY | vsql -X -c \"$COPY_QUERY\""
    [ -z "$ROWS" ] && echo "$FVECS gt $GT --k=100 | vsql -X -c \"$COPY_GT\""
    exit 0
fi

[ -x "$FVECS" ] || { echo "load_dataset.sh: $FVECS not found: run make tools" >&2; exit 1; }
for f in "$BASE" "$QUERY"; do [ -r "$f" ] || { echo "load_dataset.sh: $f not found" >&2; exit 1; }; done

vsql -X -q -v ON_ERROR_STOP=1 -c "$CREATE" > /dev/null
start=$(date +%s)
"$FVECS" rows "$BASE" $LIMIT | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_BASE"
"$FVECS" rows "$QUERY" | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_QUERY"
if [ -z "$ROWS" ] && [ -r "$GT" ]; then
    "$FVECS" gt "$GT" --k=100 | vsql -X -q -v ON_ERROR_STOP=1 -c "$COPY_GT"
fi
vsql -X -A -t -c "SELECT '${NAME}: ' || (SELECT COUNT(*) FROM ${T}_base) || ' base vectors of ' ||
                          (SELECT MAX(ARRAY_LENGTH(vec)) FROM ${T}_base) || ' dimensions, ' ||
                          (SELECT COUNT(*) FROM ${T}_query) || ' queries, ' ||
                          (SELECT COUNT(*) FROM ${T}_gt) || ' ground truth rows, loaded in $(( $(date +%s) - start )) s'"
