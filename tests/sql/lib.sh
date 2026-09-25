# Shared by the tests in tests/sql. Sourced, not run.
# Expects ECHO_ONLY (yes|no) and, optionally, PRE (SQL run first in every session).

FAILED=0
PRE="${PRE:-}"

# NODES_UP: FROM clause of the UP nodes a statement of this session runs on. OVER(PARTITION NODES)
# covers the session's subcluster in Eon (docs/design.md "Eon subclusters") and every node in
# Enterprise, where subcluster_name is NULL (the NULL-safe <=> makes both cases one expression).
NODES_UP="v_catalog.nodes WHERE node_state = 'UP' AND subcluster_name <=> (SELECT subcluster_name FROM v_catalog.nodes WHERE node_name = local_node_name())"

# run_sql NAME SQL: prints the vsql output. With --echo_only prints the SQL instead.
run_sql() {
    if [ "$ECHO_ONLY" = yes ]; then echo "-- $1"; [ -n "$PRE" ] && echo "$PRE"; echo "$2"; return 0; fi
    printf '%s\n%s\n' "$PRE" "$2" | vsql -X -A -t -q 2>&1
}

# expect NAME PATTERN SQL: the output must contain PATTERN (grep, basic regular expression).
expect() {
    local out
    out=$(run_sql "$1" "$3")
    if [ "$ECHO_ONLY" = yes ]; then echo "$out"; return 0; fi
    if echo "$out" | grep -q -- "$2"; then
        echo "PASS  $1"
    else
        echo "FAIL  $1"; echo "      wanted: $2"; echo "$out" | sed 's/^/      got: /' | head -6
        FAILED=$((FAILED + 1))
    fi
}

# random_vector DIMS: SQL for an ARRAY[FLOAT] of DIMS random elements in [-1, 1).
random_vector() {
    local i s="ARRAY["
    for ((i = 1; i <= $1; i++)); do
        [ "$i" -gt 1 ] && s+=", "
        s+="RANDOM() * 2 - 1"
    done
    echo "$s]"
}

# row_numbers N: SQL for a table (id) with the numbers 1 to N (N up to 7.4 billion).
row_numbers() {
    echo "SELECT id FROM (SELECT ROW_NUMBER() OVER() AS id FROM
            (SELECT 1 FROM (SELECT '2000-01-01 00:00:00'::TIMESTAMP AS t UNION ALL SELECT '2000-01-01 23:59:59'::TIMESTAMP) b
             TIMESERIES ts AS '1 second' OVER (ORDER BY t)) x
            CROSS JOIN (SELECT 1 FROM (SELECT '2000-01-01 00:00:00'::TIMESTAMP AS t UNION ALL SELECT '2000-01-01 23:59:59'::TIMESTAMP) b
             TIMESERIES ts AS '1 second' OVER (ORDER BY t)) y
            LIMIT $1) g"
}

# wait_cache_check: a warm query trusts the ACTIVE and OPTIONS files of the node cache for 200 ms
# (src/engine/cache.h, ACTIVE_CHECK_MS). Tests that change those files by hand, or change index
# defaults, wait that long before the next query, as unfenced queries in the same process would.
wait_cache_check() { [ "$ECHO_ONLY" = yes ] || sleep 0.3; }

# finish_tests NAME: summary line and exit code.
finish_tests() {
    [ "$ECHO_ONLY" = yes ] && exit 0
    if [ "$FAILED" -ne 0 ]; then echo "$1: $FAILED FAILED"; exit 1; fi
    echo "$1: OK"
}
