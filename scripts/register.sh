#!/usr/bin/env bash
# Register a table of vectors as a vector index.
#
#   scripts/register.sh --index=NAME --table=SCHEMA.TABLE --id=COL --vec=COL --metric=l2|cosine|dot
#                       [--op=COL] [--ver=COL] [--margin=N] [--echo_only]
#
#   --vec     vector column: ARRAY[FLOAT] (recommended), ARRAY[INT] or ARRAY[NUMERIC]
#   --op      delete flag column: BOOLEAN (true = deleted) or INT (+1 / -1)
#   --ver     version column: TIMESTAMPTZ DEFAULT CLOCK_TIMESTAMP() (recommended), TIMESTAMP or INT
#   --margin  overlap of the changes: seconds for a timestamp version (default 60), units for an INT version
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

INDEX= TABLE= ID= VEC= METRIC= OP=NULL VER=NULL MARGIN=NULL ECHO_ONLY=no
q() { printf "'%s'" "$1"; }
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
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "register.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
if [ -z "$INDEX" ] || [ -z "$TABLE" ] || [ -z "$ID" ] || [ -z "$VEC" ] || [ -z "$METRIC" ]; then
    echo "register.sh: --index, --table, --id, --vec and --metric are required" >&2; exit 2
fi
case "$MARGIN" in NULL|[0-9]*) ;; *) echo "register.sh: --margin must be a number" >&2; exit 2 ;; esac

SQL="CALL vvector.register_index($(q "$INDEX"), $(q "$TABLE"), $(q "$ID"), $(q "$VEC"), $OP, $VER, $(q "$METRIC"), $MARGIN);"
if [ "$ECHO_ONLY" = yes ]; then echo "vsql -X -c \"$SQL\""; exit 0; fi
vsql -X -v ON_ERROR_STOP=1 -c "$SQL"
