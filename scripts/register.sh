#!/usr/bin/env bash
# Register a table of vectors as a vector index.
#
#   scripts/register.sh --index=NAME --table=SCHEMA.TABLE --id=COL --vec=COL --metric=l2|cosine|dot|l1
#                       [--op=COL] [--ver=COL] [--margin=N] [--index_type=hnsw|flat] [--quantization=none|sq8]
#                       [--echo_only]
#
#   --vec     vector column: ARRAY[FLOAT] (recommended), ARRAY[INT] or ARRAY[NUMERIC]
#   --op      delete flag column: BOOLEAN (true = deleted) or INT (+1 / -1)
#   --ver     version column: TIMESTAMPTZ DEFAULT CLOCK_TIMESTAMP() (recommended), TIMESTAMP or INT
#   --margin  overlap of the changes: seconds for a timestamp version (default 60), units for an INT version
#   --index_type  hnsw (default: approximate, fast) or flat (every query scans every vector)
#   --quantization  none (default) or sq8: one byte per element besides the floats; searches rank by
#               the bytes and rescore the best candidates (set_index_options, before the first refresh)
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

INDEX= TABLE= ID= VEC= METRIC= OP=NULL VER=NULL MARGIN=NULL KIND=hnsw QUANT=none ECHO_ONLY=no
SQ="'"
q() { printf "'%s'" "${1//$SQ/$SQ$SQ}"; }     # an SQL string literal
for arg in "$@"; do
    case "$arg" in
        --index=*)   INDEX="${arg#*=}" ;;
        --table=*)   TABLE="${arg#*=}" ;;
        --id=*)      ID="${arg#*=}" ;;
        --vec=*)     VEC="${arg#*=}" ;;
        --metric=*)  METRIC="${arg#*=}" ;;
        --op=*)      OP=$(q "${arg#*=}") ;;
        --ver=*)     VER=$(q "${arg#*=}") ;;
        --margin=*)  MARGIN="${arg#*=}" ;;
        --index_type=*) KIND="${arg#*=}" ;;
        --quantization=*) QUANT="${arg#*=}" ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "register.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
if [ -z "$INDEX" ] || [ -z "$TABLE" ] || [ -z "$ID" ] || [ -z "$VEC" ] || [ -z "$METRIC" ]; then
    echo "register.sh: --index, --table, --id, --vec and --metric are required" >&2; exit 2
fi
[[ "$MARGIN" =~ ^(NULL|[0-9]+)$ ]] || { echo "register.sh: --margin must be a whole number" >&2; exit 2; }
case "$QUANT" in none|sq8) ;; *) echo "register.sh: --quantization must be none or sq8" >&2; exit 2 ;; esac

SQL="CALL vvector.register_index($(q "$INDEX"), $(q "$TABLE"), $(q "$ID"), $(q "$VEC"), $OP, $VER, $(q "$METRIC"), $MARGIN, $(q "$KIND"));"
SQL2=
[ "$QUANT" = sq8 ] && SQL2="CALL vvector.set_index_options($(q "$INDEX"), NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
if [ "$ECHO_ONLY" = yes ]; then
    echo "vsql -X -v ON_ERROR_STOP=1 -c \"$SQL\""
    [ -z "$SQL2" ] || echo "vsql -X -v ON_ERROR_STOP=1 -c \"$SQL2\""
    exit 0
fi
# One statement per call: vsql -c prints the NOTICEs of the last statement only.
vsql -X -v ON_ERROR_STOP=1 -c "$SQL"
[ -z "$SQL2" ] || vsql -X -v ON_ERROR_STOP=1 -c "$SQL2"
