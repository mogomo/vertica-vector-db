#!/usr/bin/env bash
# Runs the integration tests in the deploy modes, and deploys fenced again at the end.
#
#   tests/sql/run_all.sh [--modes=no,yes] [--complete] [--echo_only]
#
# Default: every test unfenced (the first mode), then a short set fenced (test_snapshot, test_search:
# build, load, vinfo and search in the default mode, with its memory limit). The modes change where
# the functions run, not what they compute. Unfenced is the stricter one for state kept per process
# (one process serves every session: node cache, mappings, the 200 ms ACTIVE check); fenced gets a
# new process per session. mixed runs the build fenced and the search unfenced: covered by the two.
# Run new code fenced first (a fault there stops a side process, unfenced it stops the node).
# --complete: every test in yes, no and mixed (before a milestone commit).
# Each mode: scripts/deploy.sh --fenced=MODE, then the tests. Stops at the first failing mode.
# Run it on a database node after make. The tests drop and recreate their own schemas
# (VVTEST, VVSEARCH) and indexes (vvtest, vvfresh, vs_*).
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

MODES=no,yes
COMPLETE=no
SHORT="test_snapshot.sh test_search.sh"
ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --modes=*)   MODES="${arg#--modes=}" ;;
        --complete)  COMPLETE=yes ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "run_all.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ "$COMPLETE" = yes ] && case "$*" in *--modes=*) ;; *) MODES=yes,no,mixed ;; esac

status=0
for mode in ${MODES//,/ }; do
    echo "######## FENCED=$mode"
    if [ "$ECHO_ONLY" = yes ]; then
        scripts/deploy.sh --fenced="$mode" --echo_only
        if [ "$mode" = "${MODES%%,*}" ] || [ "$COMPLETE" = yes ]; then ls tests/sql/test_*.sh; else for t in $SHORT; do echo "tests/sql/$t"; done; fi
        continue
    fi
    scripts/deploy.sh --fenced="$mode" > /dev/null 2>&1 || { echo "deploy --fenced=$mode failed"; status=1; break; }
    first=yes; [ "$mode" = "${MODES%%,*}" ] || first=no
    tests=(tests/sql/test_*.sh)
    if [ "$first" = no ] && [ "$COMPLETE" = no ]; then tests=(); for t in $SHORT; do tests+=("tests/sql/$t"); done; fi
    for t in "${tests[@]}"; do
        start=$(date +%s)
        # The scheduled refresh waits for the clock: once per run, not per mode.
        extra=(); [ "$first" = yes ] && [ "$t" = tests/sql/test_freshness.sh ] && extra=(--schedule_fires)
        if out=$("$t" "${extra[@]}" 2>&1); then
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
[ "$status" -eq 0 ] && echo "all integration tests passed in: $MODES ($([ "$COMPLETE" = yes ] && echo complete || echo "all tests in ${MODES%%,*}, the short set in the others"))" || echo "integration tests FAILED"
exit $status
