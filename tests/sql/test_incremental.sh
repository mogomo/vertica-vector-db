#!/usr/bin/env bash
# Integration test of incremental refresh and of the journal overlay on flat and HNSW indexes.
#
#   tests/sql/test_incremental.sh [--rows=N] [--schema=NAME] [--cache_dir=DIR] [--sift=SCHEMA] [--echo_only]
#
# Four indexes on one journal of N random vectors of 16 elements (default 20000): flat and HNSW,
# each with l2 and cosine. Checked:
# - journaled changes without a refresh (1000 adds, 1000 deletes, 500 replacements, 200 rows that
#   repeat the live vector, 50 deletes of ids that never existed, a delete and a re-add of one id):
#   freshness exact with precision exact equals the full scan of the live rows, on flat and on HNSW;
#   on HNSW the recall at the default precision is at least 0.9;
# - the incremental refresh that follows: the manifest, vinfo and the live rows agree, it appended
#   and tombstoned exactly what changed, the delta view is the sentinel only, precision exact equals
#   the full scan, HNSW recall at the default precision is at least 0.95;
# - a refresh without changes (or with rows that change nothing) keeps the snapshot;
# - every full-build trigger: mode full, refresh_mode full, changed build options, tombstone_ratio,
#   rebuild_every, a physical DELETE, a physical UPDATE of an old row's vector and of its version
#   (same row count, other digest; a search then finds the new vector), verify_every 0 (not noticed),
#   2 (noticed at the second refresh), an index without a digest (taken without a rebuild), a node cache without the
#   active snapshot, a static index; refresh_mode incremental ignores the ratio and status warns; a
#   bad mode is refused;
# - one refresh at a time: a second refresh of an index that is being refreshed stops with an error
#   (a real second refresh while the first waits for a lock, and a mark set by hand); status shows
#   the running refresh; a mark whose session is gone is ignored at once; the mark of a scheduled
#   refresh is ignored after 6 hours, or 4 times the last build time when that is longer; a refresh
#   that fails removes its mark;
# - rows after the boundary (an INT version column, margin 10): they are served by the delta, not built
#   into the snapshot; a first refresh with no row up to the boundary stops with a message that says
#   so; a physical DELETE of such a row before the next refresh never reaches the snapshot; a
#   physical UPDATE of one is built with its new vector; neither needs a full build;
# - a refresh whose build estimate exceeds half of the free memory builds in a file (build_in).
# With --sift=SCHEMA (sift_base and sift_query of scripts/load_dataset.sh in that schema) also the
# acceptance test of milestone M3: an HNSW index on 900,000 SIFT1M vectors, then 100 refreshes of
# 1000 adds and 500 deletes each, tombstone_ratio 0.03 (a full build must fire on the way); at the
# end the index holds the 950,000 live vectors and recall@10 at the default precision is at least
# 0.95 against the exact search. The time of every refresh is printed. Takes about 20 minutes and
# 2 GB of disk for the journal copy.
#
# Test data: schema VVINC (or --schema=NAME), dropped and recreated. Indexes vif_l2, vif_cos (flat),
# vih_l2, vih_cos (hnsw), vi_static, vi_tiny, vi_empty, vi_int, vi_sift.
#
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=20000
DIMS=16
SCHEMA=VVINC
CACHE_DIR=/tmp/vvector
SIFT=
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --cache_dir=*) CACHE_DIR="${arg#--cache_dir=}" ;;
        --sift=*)      SIFT="${arg#--sift=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "test_incremental.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PRE="ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '$CACHE_DIR';"
. tests/sql/lib.sh
IX_PREFIX=vif_
. tests/sql/search_lib.sh

INDEXES="vif_l2 vif_cos vih_l2 vih_cos"
unregister_all() {
    run_sql "unregister" "$(for i in $INDEXES vi_static vi_tiny vi_empty vi_int vi_sift; do echo "CALL vvector.unregister_index('$i');"; done)" > /dev/null
}
value() {   # SQL -> its single value
    if [ "$ECHO_ONLY" = yes ]; then echo 1; return; fi
    printf '%s\n%s\n' "$PRE" "$1" | vsql -X -A -t -q
}
# built_equals_live NAME INDEX: manifest, vinfo (live vectors, every node on the active snapshot) and the
# live rows of the journal agree.
built_equals_live() {
    expect "$1" "^built = live: [1-9][0-9]*$" "
SELECT CASE WHEN m.vector_count = l.n AND i.n = l.n AND i.nodes = u.up THEN 'built = live: ' || l.n
            ELSE 'built ' || m.vector_count || ' (manifest), ' || i.n || ' (vinfo, ' || i.nodes || ' nodes), live ' || l.n END
FROM (SELECT vector_count, active_snapshot FROM vvector.manifest WHERE index_name = '$2') m
CROSS JOIN (SELECT MAX(vector_count) AS n, COUNT(DISTINCT node_name) AS nodes, MAX(snapshot_id) AS sid
            FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$2') OVER(PARTITION NODES) FROM vvector.probe) v WHERE loaded) i
CROSS JOIN (SELECT COUNT(*) AS n FROM ($LIVE) x) l
CROSS JOIN (SELECT COUNT(*) AS up FROM nodes WHERE node_state = 'UP') u
WHERE i.sid = m.active_snapshot;"
}
refresh() {   # NAME INDEX PATTERN [MODE]
    local mode=""
    [ -n "${4:-}" ] && mode=", '$4'"
    expect "$1" "$3" "CALL vvector.refresh_index('$2'$mode);"
}
set_opts() {  # INDEX ARGUMENTS 2 TO 13 OF set_index_options
    run_sql "set_index_options $1" "CALL vvector.set_index_options('$1', $2);" > /dev/null
}
# change_round: a few adds, deletes and replacements of ids that are live now, for the trigger tests.
ROUND=0
change_round() {
    ROUND=$((ROUND + 1))
    run_sql "change round $ROUND" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 700000000 + $ROUND * 1000 + id, $VEC FROM ($(row_numbers 100)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT 5000 + $ROUND * 300 + id, TRUE FROM ($(row_numbers 300)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 14000 + $ROUND * 100 + id, $VEC FROM ($(row_numbers 100)) g;
COMMIT;" > /dev/null
}
FULL="refreshed: snapshot [0-9]*, full build"
INCR="refreshed: snapshot [0-9]*, incremental from snapshot [0-9]*"

echo "== test data: journal of $ROWS vectors; flat and HNSW indexes, l2 and cosine"
unregister_all
expect "journal and queries" "^journal $ROWS, queries 20$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.journal (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                              ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.journal (id, vec, ts) SELECT id, $VEC, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers "$ROWS")) g;
CREATE TABLE $SCHEMA.queries (qid INT, qvec ARRAY[FLOAT]);
INSERT INTO $SCHEMA.queries SELECT id, $VEC FROM ($(row_numbers 20)) g;
COMMIT;
SELECT 'journal ' || (SELECT COUNT(*) FROM $SCHEMA.journal) || ', queries ' || (SELECT COUNT(*) FROM $SCHEMA.queries);"
for ix in $INDEXES; do
    m=${ix#vi?_}; metric=$m; [ "$m" = cos ] && metric=cosine
    type=flat; [ "${ix:2:1}" = h ] && type=hnsw
    expect "register $ix ($type, $metric, margin 0) and the first refresh: a full build" "index $ix $FULL (first build)" "
CALL vvector.register_index('$ix', '$SCHEMA.journal', 'id', 'vec', 'del', 'ts', '$metric', 0, '$type');
CALL vvector.refresh_index('$ix');"
done

echo "== journaled changes, no refresh: the overlay on flat and on HNSW"
expect "journal the changes" "^delta rows: 2751$" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT 900000000 + id, $VEC FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, del) SELECT id, TRUE FROM ($(row_numbers 1000)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT 1000 + id, $VEC FROM ($(row_numbers 500)) g;
INSERT INTO $SCHEMA.journal (id, vec) SELECT id, vec FROM $SCHEMA.journal WHERE id BETWEEN 3001 AND 3200;
INSERT INTO $SCHEMA.journal (id, del) SELECT 800000000 + id, TRUE FROM ($(row_numbers 50)) g;
COMMIT;
INSERT INTO $SCHEMA.journal (id, vec) VALUES (5, $VEC);
COMMIT;
SELECT 'delta rows: ' || COUNT(id) FROM $SCHEMA.vih_l2_delta;"
for p in vif_ vih_; do
    IX_PREFIX=$p
    for m in l2 cos; do
        compare "$p$m: freshness exact, precision exact equals the full scan of the live rows" "$m" queries 10 exact ", precision='exact'"
    done
done
IX_PREFIX=vih_
for m in l2 cos; do recall "vih_$m: freshness exact at the default precision: recall@10 >= 0.9" "$m" queries 10 ", freshness='exact'" 0.9; done

echo "== incremental refresh"
for ix in $INDEXES; do
    refresh "$ix: incremental, appended and tombstoned exactly what changed" "$ix" \
        "index $ix $INCR (1501 vectors appended, 1500 tombstoned), $((ROWS + 1)) vectors of 16 dimensions, 1500 tombstones"
    built_equals_live "$ix: the snapshot holds exactly the live vectors" "$ix"
done
DELTAS=$(sep=""; for ix in $INDEXES; do printf '%sSELECT id FROM %s.%s_delta' "$sep" "$SCHEMA" "$ix"; sep=" UNION ALL "; done)
expect "the delta views hold only the sentinel" "^rows 4, journal rows 0$" "
SELECT 'rows ' || COUNT(*) || ', journal rows ' || COUNT(id) FROM ($DELTAS) d;"
expect "vinfo: the base snapshot and the tombstones of the new snapshot" "^base = previous: t, tombstones 1500$" "
SELECT 'base = previous: ' || (i.base_snapshot = m.base_snapshot AND m.base_snapshot > 0 AND m.base_snapshot < m.active_snapshot)::VARCHAR
       || ', tombstones ' || i.tombstones
FROM (SELECT MAX(base_snapshot) AS base_snapshot, MAX(tombstones) AS tombstones
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='vih_l2') OVER(PARTITION NODES) FROM vvector.probe) v) i
CROSS JOIN (SELECT base_snapshot, active_snapshot FROM vvector.manifest WHERE index_name = 'vih_l2') m;"
for p in vif_ vih_; do
    IX_PREFIX=$p
    for m in l2 cos; do compare "$p$m after the refresh: precision exact equals the full scan" "$m" queries 10 exact ", precision='exact'"; done
done
IX_PREFIX=vih_
for m in l2 cos; do recall "vih_$m after the refresh: recall@10 at the default precision >= 0.95" "$m" queries 10 "" 0.95; done
expect "status reports the tombstones" "$((ROWS + 1)) live vectors, 1500 tombstones (tombstone_ratio 0.2), refresh_mode auto, 1 incremental refreshes" "
CALL vvector.status('vih_l2');"
expect "status reports the last refresh" "last refresh: refreshed: snapshot [0-9]*, incremental" "CALL vvector.status('vih_l2');"

echo "== incremental transfer: the changed bytes only, a chain in vvector.snapshot, a cache repair by replay"
for ix in vif_l2 vih_l2; do
    # expect greps with a basic regex (no alternation): one check per index.
    expect "$ix: the refresh sent a patch (the changed bytes) over the previous snapshot; the table holds the whole copy and the patch" "^$ix patch, small: t, chain of 2, table: 2 snapshots, patch rows on the base: t$" "
SELECT m.index_name || ' ' || m.transfer || ', small: ' || (m.sent_bytes < m.index_bytes / CASE WHEN m.index_type = 'flat' THEN 4 ELSE 1 END)::VARCHAR
       || ', chain of ' || (LENGTH(m.snapshot_chain) - LENGTH(REPLACE(m.snapshot_chain, ',', '')) + 1)
       || ', table: ' || t.n || ' snapshots'
       || ', patch rows on the base: ' || (a.b = m.base_snapshot AND a.nulls = 0)::VARCHAR
FROM vvector.manifest m
JOIN (SELECT index_name, COUNT(DISTINCT snapshot_id) AS n FROM vvector.snapshot GROUP BY index_name) t ON t.index_name = m.index_name
JOIN (SELECT index_name, snapshot_id, MAX(base_snapshot) AS b, SUM(CASE WHEN base_snapshot IS NULL THEN 1 ELSE 0 END) AS nulls
      FROM vvector.snapshot GROUP BY index_name, snapshot_id) a ON a.index_name = m.index_name AND a.snapshot_id = m.active_snapshot
WHERE m.index_name = '$ix';"
done
expect "the refresh note says what was sent" "sent [0-9]* MB of [0-9]* MB (a patch on snapshot [0-9]* in every node cache; the table holds a chain of 2 snapshots" "
SELECT refresh_note FROM vvector.manifest WHERE index_name = 'vih_l2';"
# A patch's runs are small: load_on_nodes sizes its passes by their bytes, so a patch loads in one
# vload statement (session 18: counted as 8 MB pieces, a 3 MB patch took 13 passes). Counted from
# the request log: the vload statements that name the active snapshot of vih_l2 (the patch above).
expect "a patch loads in one pass (one vload statement for the active snapshot)" "^vload statements: 1$" "
SELECT 'vload statements: ' || COUNT(*)
FROM v_monitor.query_requests q JOIN vvector.manifest m ON REGEXP_LIKE(q.request, 'index_name=''vih_l2''.*snapshot_id=' || m.active_snapshot || '[^0-9]')
WHERE m.index_name = 'vih_l2' AND q.request_label = 'vvector_load';"
expect "status prints the chain and the room to grow" "snapshot pieces in vvector.snapshot: a chain of 2 snapshots from the whole copy [0-9]*, [0-9]* MB of patches after it; the layout has room for [1-9][0-9]* more vectors" "
CALL vvector.status('vih_l2');"
expect "vinfo reports the capacity and the file size" "^capacity > count: t, file_bytes > 0: t$" "
SELECT 'capacity > count: ' || (MIN(capacity) > MAX(vector_count + tombstones))::VARCHAR || ', file_bytes > 0: ' || (MIN(file_bytes) > 0)::VARCHAR
FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='vih_l2') OVER(PARTITION NODES) FROM vvector.probe) i;"
[ "$ECHO_ONLY" = yes ] || rm -rf "$CACHE_DIR/vih_l2"
wait_cache_check
expect "load_all replays the chain (the whole copy, then the patch) into a cache directory that was removed" "loaded on all nodes (2 of the chain" "
CALL vvector.load_all('vih_l2');"
built_equals_live "vih_l2 after the replay" vih_l2
IX_PREFIX=vih_
compare "vih_l2 after the replay: precision exact equals the full scan" l2 queries 10 exact ", precision='exact'"

echo "== refreshes that change nothing"
SID=$(value "SELECT active_snapshot FROM vvector.manifest WHERE index_name = 'vih_l2';")
refresh "no journal rows since the refresh: the snapshot is kept" vih_l2 "index vih_l2 refreshed: snapshot $SID kept, no vector changed"
expect "rows that repeat the live vectors and deletes of absent ids: the snapshot is kept" "index vih_l2 refreshed: snapshot $SID kept" "
INSERT INTO $SCHEMA.journal (id, vec) SELECT id, vec FROM $SCHEMA.journal WHERE id BETWEEN 4001 AND 4100;
INSERT INTO $SCHEMA.journal (id, del) SELECT 800000000 + id, TRUE FROM ($(row_numbers 10)) g;
COMMIT;
CALL vvector.refresh_index('vih_l2');"
expect "the boundary moved: the delta view holds only the sentinel" "^journal rows 0$" "
SELECT 'journal rows ' || COUNT(id) FROM $SCHEMA.vih_l2_delta;"
built_equals_live "vih_l2 still holds exactly the live vectors" vih_l2

echo "== full-build triggers"
refresh "refresh_index(name, 'full')" vih_l2 "index vih_l2 $FULL (mode full)" full
expect "a full build sends the whole snapshot and the table keeps only it" "^whole, chain of 1, table: 1 snapshots, whole rows: t$" "
SELECT m.transfer || ', chain of ' || (LENGTH(m.snapshot_chain) - LENGTH(REPLACE(m.snapshot_chain, ',', '')) + 1)
       || ', table: ' || t.n || ' snapshots, whole rows: ' || (t.patched = 0)::VARCHAR
FROM vvector.manifest m
JOIN (SELECT index_name, COUNT(DISTINCT snapshot_id) AS n, SUM(CASE WHEN base_snapshot IS NOT NULL THEN 1 ELSE 0 END) AS patched
      FROM vvector.snapshot GROUP BY index_name) t ON t.index_name = m.index_name
WHERE m.index_name = 'vih_l2';"
change_round
refresh "refresh_index(name, 'incremental')" vih_l2 "index vih_l2 $INCR" incremental
expect "the chain grows by the patch" "^patch, chain of 2$" "
SELECT transfer || ', chain of ' || (LENGTH(snapshot_chain) - LENGTH(REPLACE(snapshot_chain, ',', '')) + 1) FROM vvector.manifest WHERE index_name = 'vih_l2';"
expect "a bad mode is refused" "mode must be auto, incremental or full" "CALL vvector.refresh_index('vih_l2', 'sometimes');"
set_opts vih_l2 "NULL, 12, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "changed build options (m 12)" vih_l2 "index vih_l2 $FULL (build options changed from hnsw l2 none m=16 ef_construction=200 to hnsw l2 none m=12 ef_construction=200)"
built_equals_live "vih_l2 after the options change" vih_l2
set_opts vih_l2 "NULL, NULL, NULL, NULL, NULL, NULL, 2, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "rebuild_every 2: first incremental refresh" vih_l2 "index vih_l2 $INCR"
change_round
refresh "rebuild_every 2: second incremental refresh" vih_l2 "index vih_l2 $INCR"
change_round
refresh "rebuild_every 2: then a full build" vih_l2 "index vih_l2 $FULL (2 incremental refreshes since the last full build (rebuild_every 2))"
set_opts vih_l2 "NULL, NULL, NULL, NULL, NULL, 0.01, 0, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "tombstone_ratio 0.01: incremental while below" vih_l2 "index vih_l2 $INCR"
change_round
refresh "tombstone_ratio 0.01: a full build once above" vih_l2 "index vih_l2 $FULL ([0-9]* tombstones in [0-9]* positions, more than tombstone_ratio 0.01)"
set_opts vih_l2 "NULL, NULL, NULL, NULL, 'incremental', NULL, NULL, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "refresh_mode incremental" vih_l2 "index vih_l2 $INCR"
change_round
refresh "refresh_mode incremental ignores tombstone_ratio" vih_l2 "index vih_l2 $INCR"
expect "status warns about the tombstones" "refresh_mode incremental never rebuilds by itself" "CALL vvector.status('vih_l2');"
built_equals_live "vih_l2 after the incremental refreshes" vih_l2
set_opts vih_l2 "NULL, NULL, NULL, NULL, 'full', 0.2, NULL, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "refresh_mode full" vih_l2 "index vih_l2 $FULL (mode full)"
set_opts vih_l2 "NULL, NULL, NULL, NULL, 'auto', NULL, NULL, NULL, NULL, NULL, NULL, NULL"
change_round
refresh "refresh_mode auto again" vih_l2 "index vih_l2 $INCR"
expect "a physical DELETE of an old row: a full build that drops it" "index vif_l2 $FULL (the journal has [0-9]* rows up to the previous boundary, the last refresh counted [0-9]*" "
DELETE FROM $SCHEMA.journal WHERE id = 3500; COMMIT;
CALL vvector.refresh_index('vif_l2');"
built_equals_live "vif_l2 no longer holds the deleted row" vif_l2
FIVES="ARRAY[$(printf '5.0%.0s, ' $(seq 1 $((DIMS - 1))))5.0]"
expect "a physical UPDATE of an old row's vector: same count, other digest, a full build" \
    "index vif_l2 $FULL (the journal rows up to the previous boundary changed since the last refresh: same count ([0-9]*), other digest" "
UPDATE $SCHEMA.journal SET vec = $FIVES WHERE id = 3600; COMMIT;
CALL vvector.refresh_index('vif_l2');"
built_equals_live "vif_l2 after the UPDATE" vif_l2
expect "a search finds the updated vector" "^3600|0|1$" "
SELECT id, score, rank FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='vif_l2', query='[$(printf '5, %.0s' $(seq 1 $((DIMS - 1))))5]', k=1) OVER() FROM $SCHEMA.vif_l2_snap) s;"
expect "a physical UPDATE of an old row's version (still before the boundary): a full build" \
    "index vif_l2 $FULL (the journal rows up to the previous boundary changed since the last refresh: same count" "
UPDATE $SCHEMA.journal SET ts = ts - INTERVAL '1 hour' WHERE id = 3700; COMMIT;
CALL vvector.refresh_index('vif_l2');"
built_equals_live "vif_l2 after the version UPDATE" vif_l2
refresh "no change after the UPDATEs: the next refresh keeps the snapshot" vif_l2 "index vif_l2 refreshed: snapshot [0-9]* kept"
SEVENS="ARRAY[$(printf '7.0%.0s, ' $(seq 1 $((DIMS - 1))))7.0]"
expect "verify_every 0: a physical UPDATE is not noticed" "index vif_l2 refreshed: snapshot [0-9]* kept, .*; journal not verified (verify_every 0)" "
CALL vvector.set_index_options('vif_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0);
UPDATE $SCHEMA.journal SET vec = $SEVENS WHERE id = 3800; COMMIT;
CALL vvector.refresh_index('vif_l2');"
expect "status shows verify_every 0 and what the DBA must do" "journal digest verified never (verify_every 0): after a physical UPDATE" "CALL vvector.status('vif_l2');"
expect "verify_every 2, the second refresh since the last verification: verified, the UPDATE found" \
    "index vif_l2 $FULL (the journal rows up to the previous boundary changed since the last refresh: same count .*; journal verified in [0-9.]* seconds" "
CALL vvector.set_index_options('vif_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 2);
CALL vvector.refresh_index('vif_l2');"
refresh "verify_every 2, the first refresh after it: not verified" vif_l2 "kept, .*; journal not verified (verify_every 2, last verified 1 refreshes ago)"
refresh "verify_every 2, the second: verified" vif_l2 "kept, .*; journal verified in [0-9.]* seconds"
expect "an index without a digest gets one by a scan, without a rebuild" "index vif_l2 refreshed: snapshot [0-9]* kept, .*; journal digest taken for the first time in [0-9.]* seconds (row count verified)" "
CALL vvector.set_index_options('vif_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);
UPDATE vvector.manifest SET boundary_digest = NULL WHERE index_name = 'vif_l2'; COMMIT;
CALL vvector.refresh_index('vif_l2');"
refresh "... and the next refresh verifies against it" vif_l2 "kept, .*; journal verified in [0-9.]* seconds"
expect "a bad verify_every is refused" "verify_every must be 0 (never), 1 (every refresh) or more" "
CALL vvector.set_index_options('vif_l2', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, -1);"
refresh "the HNSW index on the same journal: a full build" vih_l2 "index vih_l2 $FULL (the journal "
expect "an exact search of the HNSW index finds the updated vector" "^3600|0|1$" "
SELECT id, score, rank FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
    USING PARAMETERS index_name='vih_l2', query='[$(printf '5, %.0s' $(seq 1 $((DIMS - 1))))5]', k=1, precision='exact') OVER() FROM $SCHEMA.vih_l2_snap) s;"
SID=$(value "SELECT active_snapshot FROM vvector.manifest WHERE index_name = 'vif_cos';")
[ "$ECHO_ONLY" = yes ] || rm -f "$CACHE_DIR/vif_cos/$SID.vv"
wait_cache_check
expect "a node cache without the active snapshot: a full build" "index vif_cos $FULL (the active snapshot $SID is in the cache of [0-9]* of [0-9]* nodes)" "
CALL vvector.refresh_index('vif_cos');"
built_equals_live "vif_cos after the full build" vif_cos
expect "a static index is always built in full" "index vi_static $FULL (a static index" "
CREATE TABLE $SCHEMA.static AS SELECT id, vec FROM $SCHEMA.journal WHERE id BETWEEN 10001 AND 10100 AND NOT del;
CALL vvector.register_index('vi_static', '$SCHEMA.static', 'id', 'vec', NULL, NULL, 'l2', NULL, 'flat');
CALL vvector.refresh_index('vi_static');
CALL vvector.refresh_index('vi_static');"
# A tiny index builds and loads within the 200 ms that a process trusts its cached ACTIVE file: the
# counts read with vinfo after the load must still be those of the new snapshot.
expect "a tiny index: an incremental refresh that takes milliseconds" "index vi_tiny $INCR (1 vectors appended, 1 tombstoned), 5 vectors of 3 dimensions, 1 tombstones" "
CREATE TABLE $SCHEMA.tiny (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE, ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP());
INSERT INTO $SCHEMA.tiny (id, vec, ts) SELECT id, ARRAY[id::FLOAT, 1.0, 0.0], CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM ($(row_numbers 5)) g;
COMMIT;
CALL vvector.register_index('vi_tiny', '$SCHEMA.tiny', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');
CALL vvector.refresh_index('vi_tiny');
INSERT INTO $SCHEMA.tiny (id, vec) VALUES (6, ARRAY[6.0, 1.0, 0.0]);
INSERT INTO $SCHEMA.tiny (id, del) VALUES (2, TRUE);
COMMIT;
CALL vvector.refresh_index('vi_tiny');"
expect "the tiny index: manifest counts and status" "^vector_count 5, tombstones 1, note: refreshed" "
SELECT 'vector_count ' || vector_count || ', tombstones ' || tombstones || ', note: ' || LEFT(refresh_note, 9) FROM vvector.manifest WHERE index_name = 'vi_tiny';"
expect "status of the tiny index" "5 live vectors, 1 tombstones" "CALL vvector.status('vi_tiny');"

echo "== a build larger than half of the free memory is made in a file"
change_round
expect "the refresh builds in a file (index_bytes set by hand to 4 PB) and says so" "index vih_l2 $INCR .*; built in a file: the build needs about" "
UPDATE vvector.manifest SET index_bytes = 4000000000000000 WHERE index_name = 'vih_l2'; COMMIT;
CALL vvector.refresh_index('vih_l2');"
built_equals_live "vih_l2 after the build in a file" vih_l2

echo "== rows after the boundary: served by the delta, never built before they are below it"
# INT versions: the boundary is the highest version minus the margin (10), so a new row with a
# higher version moves it without waiting.
q_int() {  # QUERY_VECTOR FRESHNESS: the nearest id through the view of that freshness
    local view=vi_int_snap; [ "$2" = exact ] && view=vi_int_delta
    echo "SELECT id FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='vi_int',
          query='$1', k=1, freshness='$2', precision='exact') OVER() FROM $SCHEMA.$view) s;"
}
expect "a first refresh with no row up to the boundary: refused with the reason" \
    "index vi_int: no live vector up to the delta boundary -5 .*; 5 rows of $SCHEMA.ints are newer: queries with freshness='exact' find them" "
CREATE TABLE $SCHEMA.ints (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE, v INT NOT NULL);
INSERT INTO $SCHEMA.ints (id, vec, v) SELECT id, ARRAY[id::FLOAT, 1.0, 0.0], id FROM ($(row_numbers 5)) g;
COMMIT;
CALL vvector.register_index('vi_int', '$SCHEMA.ints', 'id', 'vec', 'del', 'v', 'l2', 10, 'flat');
CALL vvector.refresh_index('vi_int');"
expect "rows below the boundary are built, the rows inside the margin are not" "index vi_int $FULL (first build), 5 vectors of 3 dimensions" "
INSERT INTO $SCHEMA.ints (id, vec, v) VALUES (6, ARRAY[6.0, 1.0, 0.0], 20);
INSERT INTO $SCHEMA.ints (id, vec, v) VALUES (7, ARRAY[7.0, 1.0, 0.0], 21);
INSERT INTO $SCHEMA.ints (id, vec, v) VALUES (8, ARRAY[8.0, 1.0, 0.0], 22);
COMMIT;
CALL vvector.refresh_index('vi_int');"
expect "a row inside the margin: not in the snapshot" "^5$" "$(q_int '[7, 1, 0]' snapshot)"
expect "... but found through the delta" "^7$" "$(q_int '[7, 1, 0]' exact)"
expect "a physical DELETE of that row, then a refresh: incremental, the row was never built" \
    "index vi_int $INCR (2 vectors appended, 0 tombstoned), 7 vectors of 3 dimensions.*; journal verified" "
DELETE FROM $SCHEMA.ints WHERE id = 7; COMMIT;
INSERT INTO $SCHEMA.ints (id, vec, v) VALUES (9, ARRAY[9.0, 1.0, 0.0], 40); COMMIT;
CALL vvector.refresh_index('vi_int');"
expect "the deleted row is in no search" "^6$" "$(q_int '[7, 1, 0]' exact)"
expect "a physical UPDATE of a row inside the margin, then a refresh: built with the new vector" \
    "index vi_int $INCR (1 vectors appended, 0 tombstoned), 8 vectors of 3 dimensions.*; journal verified" "
UPDATE $SCHEMA.ints SET vec = ARRAY[50.0, 1.0, 0.0] WHERE id = 9; COMMIT;
INSERT INTO $SCHEMA.ints (id, vec, v) VALUES (10, ARRAY[10.0, 1.0, 0.0], 60); COMMIT;
CALL vvector.refresh_index('vi_int');"
expect "the snapshot holds the updated vector" "^9$" "$(q_int '[50, 1, 0]' snapshot)"

echo "== one refresh at a time"
# A real second refresh: the first holds its mark while it waits for a lock on vvector.snapshot (the
# INSERT of its chunks), which another session keeps for a few seconds.
if [ "$ECHO_ONLY" = no ]; then
    change_round
    printf '%s\n' "LOCK TABLE vvector.snapshot IN EXCLUSIVE MODE;" "SELECT SLEEP(6);" "COMMIT;" | vsql -X -A -t -q > /dev/null 2>&1 &
    locker=$!
    sleep 1
    printf '%s\n%s\n' "$PRE" "CALL vvector.refresh_index('vif_cos');" | vsql -X -A -t -q > "${TMPDIR:-/tmp}/vvinc_first.$$" 2>&1 &
    first=$!
    for _ in $(seq 1 40); do
        [ "$(value "SELECT COUNT(refresh_started_at) FROM vvector.manifest WHERE index_name = 'vif_cos';")" = 1 ] && break
        sleep 0.1
    done
fi
expect "a second refresh while the first runs: refused" "index vif_cos is being refreshed since .* (by .*, session .*). Two refreshes of one index cannot run at the same time" "
CALL vvector.refresh_index('vif_cos');"
expect "status shows the running refresh" "index vif_cos: a refresh is running since" "CALL vvector.status('vif_cos');"
if [ "$ECHO_ONLY" = no ]; then
    wait "$locker"; wait "$first"
    if grep -q "index vif_cos $INCR" "${TMPDIR:-/tmp}/vvinc_first.$$"; then
        echo "PASS  the first refresh finished"
    else
        echo "FAIL  the first refresh finished"; sed 's/^/      got: /' "${TMPDIR:-/tmp}/vvinc_first.$$" | head -6
        FAILED=$((FAILED + 1))
    fi
    rm -f "${TMPDIR:-/tmp}/vvinc_first.$$"
fi
expect "the mark is gone after the refresh" "^mark: 0$" "
SELECT 'mark: ' || COUNT(refresh_started_at) FROM vvector.manifest WHERE index_name = 'vif_cos';"
expect "a mark of the caller's own session (left by a refresh of this session that failed) is ignored" "index vif_cos refreshed" "
UPDATE vvector.manifest SET refresh_started_at = CLOCK_TIMESTAMP(),
       refresh_started_by = CURRENT_USER() || ', session ' || (SELECT session_id FROM v_monitor.current_session)
    WHERE index_name = 'vif_cos'; COMMIT;
CALL vvector.refresh_index('vif_cos');"
expect "a mark whose session is gone: ignored at once (a superuser sees every session)" "index vif_cos refreshed" "
UPDATE vvector.manifest SET refresh_started_at = CLOCK_TIMESTAMP() - INTERVAL '1 minute', refresh_started_by = 'someone, session x'
    WHERE index_name = 'vif_cos'; COMMIT;
CALL vvector.refresh_index('vif_cos');"
# A refresh run by a schedule trigger has a session that v_monitor.sessions never shows: its mark
# is judged by its age only.
expect "a mark of a scheduled refresh: refused, with the statement that removes it" "UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'vif_cos'" "
UPDATE vvector.manifest SET refresh_started_at = CLOCK_TIMESTAMP() - INTERVAL '5 hours', refresh_started_by = 'someone, scheduled, internal session x'
    WHERE index_name = 'vif_cos'; COMMIT;
CALL vvector.refresh_index('vif_cos');"
expect "a mark older than 6 hours is ignored" "index vif_cos refreshed" "
UPDATE vvector.manifest SET refresh_started_at = CLOCK_TIMESTAMP() - INTERVAL '7 hours', refresh_started_by = 'someone, scheduled, internal session x'
    WHERE index_name = 'vif_cos'; COMMIT;
CALL vvector.refresh_index('vif_cos');"
expect "... unless the index builds long: the limit is 4 times the last build time" "ignored as soon as its session is gone .*, else 12 hours after its start" "
UPDATE vvector.manifest SET build_seconds = 10800, refresh_started_at = CLOCK_TIMESTAMP() - INTERVAL '7 hours',
       refresh_started_by = 'someone, scheduled, internal session x' WHERE index_name = 'vif_cos'; COMMIT;
CALL vvector.refresh_index('vif_cos');"
run_sql "remove the mark" "UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'vif_cos'; COMMIT;" > /dev/null
expect "a refresh that fails reports its error" "vvector.refresh_index: index vi_empty: table $SCHEMA.empty has no vectors" "
CREATE TABLE $SCHEMA.empty (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE, ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP());
CALL vvector.register_index('vi_empty', '$SCHEMA.empty', 'id', 'vec', 'del', 'ts', 'l2', 0, 'flat');
CALL vvector.refresh_index('vi_empty');"
expect "... and removes its mark" "^mark: 0$" "
SELECT 'mark: ' || COUNT(refresh_started_at) FROM vvector.manifest WHERE index_name = 'vi_empty';"

if [ -n "$SIFT" ]; then
    echo "== SIFT1M: HNSW on 900,000 vectors, 100 refreshes of 1000 adds and 500 deletes"
    IX=vi_sift
    expect "journal: the first 900,000 vectors of $SIFT.sift_base" "^journal 900000$" "
CREATE TABLE $SCHEMA.sift (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                           ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP())
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);
INSERT INTO $SCHEMA.sift (id, vec, ts) SELECT id, vec, CLOCK_TIMESTAMP() - INTERVAL '1 day' FROM $SIFT.sift_base WHERE id < 900000;
CREATE TABLE $SCHEMA.sift_rounds (round INT, seconds FLOAT, client_ms INT, note VARCHAR(1000));
COMMIT;
SELECT 'journal ' || COUNT(*) FROM $SCHEMA.sift;"
    expect "register (HNSW, l2, margin 0, tombstone_ratio 0.03) and the full build" "index $IX $FULL (first build)" "
CALL vvector.register_index('$IX', '$SCHEMA.sift', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');
CALL vvector.set_index_options('$IX', NULL, NULL, NULL, NULL, NULL, 0.03, NULL, NULL, NULL, NULL, NULL, NULL);
CALL vvector.refresh_index('$IX');"
    errors=0
    for ((r = 1; r <= 100; r++)); do
        t0=$(date +%s%N)
        out=$(run_sql "round $r" "
INSERT INTO $SCHEMA.sift (id, vec) SELECT id, vec FROM $SIFT.sift_base WHERE id >= $((900000 + (r - 1) * 1000)) AND id < $((900000 + r * 1000));
INSERT INTO $SCHEMA.sift (id, del) SELECT id, TRUE FROM $SIFT.sift_base WHERE id >= $(((r - 1) * 500)) AND id < $((r * 500));
COMMIT;
CALL vvector.refresh_index('$IX');")
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        [ "$ECHO_ONLY" = yes ] && break
        if echo "$out" | grep -q ERROR; then errors=$((errors + 1)); echo "$out" | grep ERROR | head -2; fi
        run_sql "round $r result" "INSERT INTO $SCHEMA.sift_rounds SELECT $r, build_seconds, $ms, refresh_note FROM vvector.manifest WHERE index_name = '$IX'; COMMIT;" > /dev/null
        [ $((r % 10)) -eq 0 ] && echo "      round $r: $(value "SELECT seconds || ' s (' || client_ms || ' ms at the client): ' || note FROM $SCHEMA.sift_rounds WHERE round = $r;")"
    done
    expect "100 rounds without an error" "^errors: 0$" "SELECT 'errors: $errors';"
    LIVE="SELECT id, vec FROM (SELECT id, vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, del DESC) AS rn
          FROM $SCHEMA.sift) j WHERE rn = 1 AND NOT del"
    built_equals_live "$IX holds exactly the live vectors" "$IX"
    expect "the live count is 950,000" "^live 950000$" "SELECT 'live ' || vector_count FROM vvector.manifest WHERE index_name = '$IX';"
    expect "tombstone_ratio fired a full build on the way, the other refreshes were incremental" "^full 1, incremental 99$" "
SELECT 'full ' || SUM(CASE WHEN note LIKE '%full build (%tombstone_ratio 0.03)%' THEN 1 ELSE 0 END)
       || ', incremental ' || SUM(CASE WHEN note LIKE '%incremental from snapshot%' THEN 1 ELSE 0 END) FROM $SCHEMA.sift_rounds;"
    expect "recall@10 at the default precision against the exact search, 1000 queries" "^recall ok" "
DROP TABLE IF EXISTS $SCHEMA.q1000; CREATE TABLE $SCHEMA.q1000 AS SELECT qid, qvec FROM $SIFT.sift_query WHERE qid < 1000;
DROP TABLE IF EXISTS $SCHEMA.got; DROP TABLE IF EXISTS $SCHEMA.ref;
CREATE TABLE $SCHEMA.got AS SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', k=10) OVER()
    FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.q1000) x;
CREATE TABLE $SCHEMA.ref AS SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', k=10, precision='exact') OVER()
    FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $SCHEMA.q1000) x;
SELECT CASE WHEN r >= 0.95 THEN 'recall ok ' ELSE 'recall too low ' END || r::NUMERIC(6,4) FROM
    (SELECT (SELECT COUNT(*) FROM $SCHEMA.got g JOIN $SCHEMA.ref r ON g.qid = r.qid AND g.id = r.id) / (SELECT COUNT(*) FROM $SCHEMA.ref)::FLOAT AS r) x;"
    [ "$ECHO_ONLY" = yes ] || echo "      refresh seconds: $(value "
SELECT 'incremental median ' || MEDIAN(seconds) OVER() || ' (min ' || MIN(seconds) OVER() || ', max ' || MAX(seconds) OVER() || ')'
FROM $SCHEMA.sift_rounds WHERE note LIKE '%incremental%' LIMIT 1;"); full $(value "SELECT MAX(seconds) FROM $SCHEMA.sift_rounds WHERE note LIKE '%full build%';")"
fi

unregister_all
run_sql "cleanup" "DROP SCHEMA $SCHEMA CASCADE;" > /dev/null
finish_tests test_incremental
