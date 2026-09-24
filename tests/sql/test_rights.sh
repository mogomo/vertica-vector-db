#!/usr/bin/env bash
# Integration test of the rights model, with a database user that has only the rights the README
# lists for an index administrator.
#
#   tests/sql/test_rights.sh [--schema=NAME] [--cache_dir=DIR] [--echo_only]
#
# Creates a user vvrights_<random suffix> with a random password, the role vvector_admin as its
# default role, USAGE and CREATE on the test schema and SELECT on the journal. As that user:
# register_index, refresh_index (first build, incremental, full), status, searches (vsearch over the
# delta view, vsearch FROM dual, vknn) and unregister_index (vvector_admin holds vvector_search).
# With its roles switched off (SET ROLE NONE) the same user cannot search (vsearch, vknn, vinfo need
# the role vvector_search since milestone M6), cannot read the manifest, cannot call vbuild, vload,
# vconfig or vnode (schema vvector_admin), read vvector.snapshot or run refresh_index; sizing stays
# open. With the role vvector_search alone it can search, but not build, load or read the manifest.
# The user and the schema are dropped at the end, also when the test fails (EXIT trap).
#
# Needs a superuser connection (CREATE USER); skipped with a message otherwise.
# Test data: schema VVRIGHTS (or --schema=NAME), dropped and recreated; index vr_rights.
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

SCHEMA=VVRIGHTS
CACHE_DIR=/tmp/vvector
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "test_rights.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh

IX=vr_rights
DIMS=8
VEC=$(random_vector $DIMS)
SUFFIX=$(od -An -N4 -tx4 /dev/urandom | tr -d ' \n')
TEST_USER="vvrights_$SUFFIX"
TEST_PASSWORD="Vv1$(od -An -N12 -tx4 /dev/urandom | tr -d ' \n')"

if [ "$ECHO_ONLY" = no ] && [ "$(vsql -X -A -t -q -c "SELECT is_super_user FROM v_catalog.users WHERE user_name = CURRENT_USER();")" != t ]; then
    echo "test_rights: SKIPPED: the connection is not a superuser (the test creates and drops a database user)"
    exit 0
fi

cleanup() {
    [ "$ECHO_ONLY" = yes ] && return
    printf '%s\n' "CALL vvector.unregister_index('$IX');" "DROP SCHEMA IF EXISTS $SCHEMA CASCADE;" \
                  "DROP USER IF EXISTS $TEST_USER CASCADE;" | vsql -X -A -t -q > /dev/null 2>&1
}
trap cleanup EXIT

# expect_as NAME PATTERN SQL: expect, run as the test user.
expect_as() {
    local had_user=${VSQL_USER+x} old_user=${VSQL_USER-} had_pw=${VSQL_PASSWORD+x} old_pw=${VSQL_PASSWORD-}
    export VSQL_USER="$TEST_USER" VSQL_PASSWORD="$TEST_PASSWORD"
    expect "as the user: $1" "$2" "$3"
    if [ -n "$had_user" ]; then export VSQL_USER="$old_user"; else unset VSQL_USER; fi
    if [ -n "$had_pw" ]; then export VSQL_PASSWORD="$old_pw"; else unset VSQL_PASSWORD; fi
}
NO_ROLE="SET ROLE NONE;"
QUERY="[$(printf '0.1, %.0s' $(seq 1 $((DIMS - 1))))0.1]"

echo "== test data and the user"
cleanup
expect "journal of 2000 vectors, user with vvector_admin" "^journal 2000, user 1$" "
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.journal (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                              ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.journal (id, vec, ts) SELECT id, $VEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers 2000)) g;
COMMIT;
CREATE USER $TEST_USER IDENTIFIED BY '$TEST_PASSWORD';
GRANT vvector_admin TO $TEST_USER;
ALTER USER $TEST_USER DEFAULT ROLE vvector_admin;
GRANT USAGE, CREATE ON SCHEMA $SCHEMA TO $TEST_USER;
GRANT SELECT ON $SCHEMA.journal TO $TEST_USER;
SELECT 'journal ' || (SELECT COUNT(*) FROM $SCHEMA.journal) || ', user ' || (SELECT COUNT(*) FROM v_catalog.users WHERE user_name = '$TEST_USER');"

echo "== the index administrator: vvector_admin, USAGE and CREATE on the schema, SELECT on the table"
expect_as "register_index" "index $IX registered" "
CALL vvector.register_index('$IX', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');"
expect_as "refresh_index: the first build" "index $IX refreshed: snapshot [0-9]*, full build (first build), 2000 vectors" "
CALL vvector.refresh_index('$IX');"
run_sql "journal changes" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 5000 + id, $VEC FROM ($(row_numbers 100)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id, TRUE FROM ($(row_numbers 50)) g;
COMMIT;" > /dev/null
expect_as "refresh_index: incremental" "index $IX refreshed: snapshot [0-9]*, incremental from snapshot [0-9]* (100 vectors appended, 50 tombstoned)" "
CALL vvector.refresh_index('$IX');"
expect_as "refresh_index(name, 'full')" "index $IX refreshed: snapshot [0-9]*, full build (mode full), 2050 vectors" "
CALL vvector.refresh_index('$IX', 'full');"
expect_as "status" "index $IX: hnsw index, 2050 live vectors" "CALL vvector.status('$IX');"
expect_as "vsearch over the delta view" "^rows 10$" "
SELECT 'rows ' || COUNT(*) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='$IX', query='$QUERY', k=10, freshness='exact') OVER() FROM $SCHEMA.${IX}_delta) s;"

DENIED="ermission denied\|does not exist, or permission is denied"
SEARCH_DUAL="SELECT 'rows ' || COUNT(*) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='$IX', query='$QUERY', k=10) OVER()
    FROM (SELECT NULL::INT AS qid, NULL::ARRAY[FLOAT] AS qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec,
                 NULL::BOOLEAN AS del, NULL::INT AS ver, NULL::INT AS snapshot_id) d) s;"
KNN_DUAL="SELECT 'rows ' || COUNT(*) FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='$IX', query='$QUERY', k=5) FROM dual) s;"
VINFO="SELECT 'vinfo rows ' || COUNT(*) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) i;"

echo "== the same user with its roles switched off: no search, no build or load"
expect_as "vsearch FROM dual is refused without the role vvector_search" "$DENIED" "$NO_ROLE
$SEARCH_DUAL"
expect_as "vknn is refused" "$DENIED" "$NO_ROLE
$KNN_DUAL"
expect_as "vinfo is refused" "$DENIED" "$NO_ROLE
$VINFO"
expect_as "the manifest cannot be read" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM vvector.manifest;"
expect_as "sizing stays open to everyone" "vvector.sizing: 1000000 vectors of 128 dimensions" "$NO_ROLE
CALL vvector.sizing(1000000, 128, 'hnsw', 'none');"
expect_as "vbuild is refused" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM (SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='$IX', base_snapshot=1) OVER() FROM $SCHEMA.journal) b;"
expect_as "vload is refused" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM (SELECT vvector_admin.vload(0, 'x'::LONG VARBINARY USING PARAMETERS index_name='$IX', snapshot_id=1) OVER(PARTITION NODES)) l;"
expect_as "vconfig is refused" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM (SELECT vvector_admin.vconfig(k USING PARAMETERS index_name='$IX', options='') OVER(PARTITION NODES) FROM vvector.probe) c;"
expect_as "vnode is refused" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n;"
expect_as "vvector.snapshot cannot be read" "$DENIED" "$NO_ROLE
SELECT COUNT(*) FROM vvector.snapshot;"
expect_as "refresh_index is refused" "$DENIED" "$NO_ROLE
CALL vvector.refresh_index('$IX');"

echo "== the role vvector_search alone: search yes, build, load and the manifest no"
run_sql "grant vvector_search" "GRANT vvector_search TO $TEST_USER;" > /dev/null
SEARCH_ROLE="SET ROLE vvector_search;"
expect_as "vsearch FROM dual" "^rows 10$" "$SEARCH_ROLE
$SEARCH_DUAL"
expect_as "vknn FROM dual" "^rows 5$" "$SEARCH_ROLE
$KNN_DUAL"
expect_as "vinfo" "^vinfo rows [1-9]" "$SEARCH_ROLE
$VINFO"
expect_as "vbuild is refused" "$DENIED" "$SEARCH_ROLE
SELECT COUNT(*) FROM (SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='$IX', base_snapshot=1) OVER() FROM $SCHEMA.journal) b;"
expect_as "the manifest cannot be read" "$DENIED" "$SEARCH_ROLE
SELECT COUNT(*) FROM vvector.manifest;"
expect_as "refresh_index is refused" "$DENIED" "$SEARCH_ROLE
CALL vvector.refresh_index('$IX');"

echo "== back with the role"
expect_as "schedule_refresh needs a superuser (Vertica: triggers)" "Super User\|[Pp]ermission denied\|superuser" "
CALL vvector.schedule_refresh('$IX', '0 3 * * *');"
expect_as "unregister_index" "index $IX unregistered" "CALL vvector.unregister_index('$IX');"
expect "the views are gone" "^views 0$" "
SELECT 'views ' || COUNT(*) FROM v_catalog.views WHERE LOWER(table_schema) = LOWER('$SCHEMA');"

finish_tests test_rights
