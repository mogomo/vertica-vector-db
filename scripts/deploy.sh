#!/usr/bin/env bash
# Install or remove the vvector library and functions.
#
#   scripts/deploy.sh [--fenced=yes|no|mixed] [--search=role|public] [--undeploy] [--echo_only]
#
#   --fenced=yes    every function runs in a separate fenced process (default; a crash cannot
#                   take down the node)
#   --fenced=no     every function runs inside the Vertica process (fastest; a crash in the
#                   library is a crash of the node)
#   --fenced=mixed  vbuild, vload, vconfig, vnode fenced; vsearch, vknn, vinfo, vversion and the
#                   vector functions not fenced
#   --search=role   the search functions (schema vvector) for the role vvector_search (default)
#   --search=public the search functions for every user, as before milestone M6
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD,
# VSQL_DATABASE from the environment. Nothing is stored here.
# Run it on a Vertica node: CREATE LIBRARY reads the .so from the
# initiator node's file system.
set -euo pipefail

cd "$(dirname "$0")/.."

FENCED=yes
SEARCH=role
UNDEPLOY=no
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --fenced=yes|--fenced=no|--fenced=mixed) FENCED="${arg#--fenced=}" ;;
        --search=role|--search=public) SEARCH="${arg#--search=}" ;;
        --undeploy)  UNDEPLOY=yes ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "deploy.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

LIBFILE="$PWD/build/libvvector.so"

case "$FENCED" in
    yes)   BUILD_SQL="FENCED";     SEARCH_SQL="FENCED" ;;
    no)    BUILD_SQL="NOT FENCED"; SEARCH_SQL="NOT FENCED" ;;
    mixed) BUILD_SQL="FENCED";     SEARCH_SQL="NOT FENCED" ;;
esac
GRANTEES="vvector_search"
[ "$SEARCH" = public ] && GRANTEES="vvector_search, PUBLIC"

if [ "$UNDEPLOY" = yes ]; then
    CMD=(vsql -X -v ON_ERROR_STOP=1 -f sql/uninstall.sql)
else
    CMD=(vsql -X -v ON_ERROR_STOP=1 -v "libfile='$LIBFILE'" -v "fenced_build=$BUILD_SQL" -v "fenced_search=$SEARCH_SQL" -v "search_grantees=$GRANTEES" -f sql/install.sql)
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
    vsql -X -c "SELECT schema_name || '.' || function_name AS function_name, is_fenced FROM v_catalog.user_functions
                WHERE schema_name IN ('vvector', 'vvector_admin')
                  AND function_name IN ('vversion','vbuild','vload','vconfig','vnode','vinfo','vsearch','vknn','vector_add','vector_avg')
                ORDER BY 1;"
fi
