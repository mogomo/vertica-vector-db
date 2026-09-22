#!/usr/bin/env bash
# Install or remove the vvector library and functions.
#
#   scripts/deploy.sh [--fenced=yes|no] [--undeploy] [--echo_only]
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD,
# VSQL_DATABASE from the environment. Nothing is stored here.
# Run it on a Vertica node: CREATE LIBRARY reads the .so from the
# initiator node's file system.
set -euo pipefail

cd "$(dirname "$0")/.."

FENCED=yes
UNDEPLOY=no
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --fenced=yes|--fenced=no) FENCED="${arg#--fenced=}" ;;
        --undeploy)  UNDEPLOY=yes ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "deploy.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

LIBFILE="$PWD/build/libvvector.so"

if [ "$FENCED" = yes ]; then FENCED_SQL="FENCED"; else FENCED_SQL="NOT FENCED"; fi

if [ "$UNDEPLOY" = yes ]; then
    CMD=(vsql -X -v ON_ERROR_STOP=1 -f sql/uninstall.sql)
else
    CMD=(vsql -X -v ON_ERROR_STOP=1 -v "libfile='$LIBFILE'" -v "fenced=$FENCED_SQL" -f sql/install.sql)
fi

if [ "$ECHO_ONLY" = yes ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

if [ "$UNDEPLOY" = no ] && [ ! -f "$LIBFILE" ]; then
    echo "deploy.sh: $LIBFILE not found. Run make first." >&2
    exit 1
fi

"${CMD[@]}"

if [ "$UNDEPLOY" = no ]; then
    vsql -X -c "SELECT vvector.vversion() OVER();"
    vsql -X -c "SELECT function_name, is_fenced FROM v_catalog.user_functions
                WHERE schema_name = 'vvector'
                  AND function_name IN ('vversion','vbuild','vload','vnode','vinfo','vsearch')
                ORDER BY 1;"
fi
