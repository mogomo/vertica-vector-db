#!/usr/bin/env bash
# Runs every integration test in every deploy mode: fenced first, then unfenced, then mixed,
# and deploys fenced again at the end.
#
#   tests/sql/run_all.sh [--modes=yes,no,mixed] [--echo_only]
#
# Each mode: scripts/deploy.sh --fenced=MODE, then tests/sql/test_*.sh. Stops at the first failing mode.
# Run it on a database node after make. The tests drop and recreate their own schemas
# (VVTEST, VVSEARCH) and indexes (vvtest, vvfresh, vs_*).
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

MODES=yes,no,mixed
ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --modes=*)   MODES="${arg#--modes=}" ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "run_all.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

status=0
for mode in ${MODES//,/ }; do
    echo "######## FENCED=$mode"
    if [ "$ECHO_ONLY" = yes ]; then
        scripts/deploy.sh --fenced="$mode" --echo_only
        for t in tests/sql/test_*.sh; do echo "$t"; done
        continue
    fi
    scripts/deploy.sh --fenced="$mode" > /dev/null 2>&1 || { echo "deploy --fenced=$mode failed"; status=1; break; }
    for t in tests/sql/test_*.sh; do
        start=$(date +%s)
        if out=$("$t" 2>&1); then
            echo "$(tail -1 <<< "$out")  ($(( $(date +%s) - start )) s)"
        else
            grep -E '^FAIL|wanted|got' <<< "$out" | head -30
            echo "$(tail -1 <<< "$out")"
            status=1
        fi
    done
    [ "$status" -eq 0 ] || break
done
[ "$ECHO_ONLY" = yes ] || scripts/deploy.sh --fenced=yes > /dev/null 2>&1
[ "$status" -eq 0 ] && echo "all integration tests passed in: $MODES" || echo "integration tests FAILED"
exit $status
