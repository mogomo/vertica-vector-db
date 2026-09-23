#!/usr/bin/env bash
# Refresh the snapshot of a vector index and load it on every node, or only repair the node caches.
#
#   scripts/refresh.sh --index=NAME [--mode=auto|incremental|full] [--load_only] [--schedule='CRON'] [--status] [--echo_only]
#
#   --mode        refresh_index with this mode (default: the index's refresh_mode)
#   --load_only   vvector.load_all: load the active snapshot again on every node
#   --schedule    vvector.schedule_refresh with a cron expression, for example '0 * * * *'
#   --status      vvector.status: pending changes, open writers, warnings
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -euo pipefail

INDEX= MODE=refresh CRON= RMODE= ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --index=*)    INDEX="${arg#*=}" ;;
        --mode=*)     RMODE="${arg#*=}" ;;
        --load_only)  MODE=load ;;
        --schedule=*) MODE=schedule; CRON="${arg#*=}" ;;
        --status)     MODE=status ;;
        --echo_only)  ECHO_ONLY=yes ;;
        -h|--help)    sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "refresh.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$INDEX" ] || { echo "refresh.sh: --index is required" >&2; exit 2; }
case "$INDEX" in *[!A-Za-z0-9_]*) echo "refresh.sh: --index must be letters, digits or underscores" >&2; exit 2 ;; esac
case "$RMODE" in ''|auto|incremental|full) ;; *) echo "refresh.sh: --mode must be auto, incremental or full" >&2; exit 2 ;; esac

case "$MODE" in
    refresh)  SQL="CALL vvector.refresh_index('$INDEX'${RMODE:+, '$RMODE'});" ;;
    load)     SQL="CALL vvector.load_all('$INDEX');" ;;
    schedule) SQL="CALL vvector.schedule_refresh('$INDEX', '$CRON');" ;;
    status)   SQL="CALL vvector.status('$INDEX');" ;;
esac
if [ "$ECHO_ONLY" = yes ]; then echo "vsql -X -c \"$SQL\""; exit 0; fi
vsql -X -v ON_ERROR_STOP=1 -c "$SQL"
