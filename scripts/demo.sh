#!/usr/bin/env bash
# A walk through vvector on one table: load vectors into a journal, register it as an HNSW index,
# refresh, search (and compare with the built-in full scan), change vectors without a refresh and
# search again, refresh incrementally, look at every node.
#
#   scripts/demo.sh [--dir=DIR] [--rows=N] [--dims=N] [--schema=VVDEMO] [--keep] [--echo_only]
#
#   --dir     a directory with SIFT1M (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs):
#             the demo loads it and measures the recall against the ground truth. Without --dir it
#             generates --rows vectors of --dims elements (default 100000 x 128) and measures the
#             recall against the exact search.
#   --schema  where the demo's tables go (dropped and made again); index name vvdemo
#   --keep    keep the tables and the index at the end (default: both are removed)
#
# Needs `make` and `make deploy` first; runs `make tools` itself when build/tools/fvecs is missing.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/.."

DIR= ROWS=100000 DIMS=128 SCHEMA=VVDEMO KEEP=no ECHO_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --dir=*)     DIR="${arg#*=}" ;;
        --rows=*)    ROWS="${arg#*=}" ;;
        --dims=*)    DIMS="${arg#*=}" ;;
        --schema=*)  SCHEMA="${arg#*=}" ;;
        --keep)      KEEP=yes ;;
        --echo_only) ECHO_ONLY=yes ;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "demo.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
case "$SCHEMA" in *[!A-Za-z0-9_]*|'') echo "demo.sh: --schema must be letters, digits or underscores" >&2; exit 2 ;; esac
case "$ROWS$DIMS" in *[!0-9]*) echo "demo.sh: --rows and --dims must be numbers" >&2; exit 2 ;; esac

IX=vvdemo
if [ -n "$DIR" ]; then DS=sift; LOAD=(--dataset=sift --dir="$DIR" --schema="$SCHEMA")
else DS=gen; LOAD=(--dataset=gen --generate="$DIMS" --rows="$ROWS" --queries=100 --schema="$SCHEMA"); fi
T="$SCHEMA.${DS}_base" Q="$SCHEMA.${DS}_query"

if [ "$ECHO_ONLY" = yes ]; then
    scripts/load_dataset.sh "${LOAD[@]}" --echo_only
    echo "CALL vvector.register_index('$IX', '$T', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');"
    echo "CALL vvector.refresh_index('$IX');"
    echo "-- searches, recall, INSERT of 1 vector and 1 delete into $T, refresh, vinfo"
    [ "$KEEP" = yes ] || echo "CALL vvector.unregister_index('$IX'); DROP SCHEMA $SCHEMA CASCADE;"
    exit 0
fi

sql() { vsql -X -q -v ON_ERROR_STOP=1 -c "$1"; }
val() { vsql -X -A -t -q -v ON_ERROR_STOP=1 -c "$1"; }
ms() { echo $(( ($(date +%s%N) - $1) / 1000000 )); }
step() { echo; echo "== $*"; }

[ -x build/tools/fvecs ] || make tools > /dev/null || { echo "demo.sh: make tools failed"; exit 1; }
val "SELECT 1 FROM v_catalog.user_functions WHERE schema_name = 'vvector' AND function_name = 'vsearch' LIMIT 1" | grep -q 1 ||
    { echo "demo.sh: vvector is not installed here: run make deploy first"; exit 1; }

step "1. Load $( [ -n "$DIR" ] && echo "SIFT1M from $DIR" || echo "$ROWS generated vectors of $DIMS elements") into the journal $T"
val "CALL vvector.unregister_index('$IX');" > /dev/null 2>&1
t0=$(date +%s%N)
scripts/load_dataset.sh "${LOAD[@]}" 2>&1 | grep -v NOTICE
echo "loaded in $(ms $t0) ms: $(val "SELECT COUNT(*) FROM $T") rows; the table is a journal (id, vec, del, ts): changes are new rows"

step "2. Register it as an HNSW index and build it"
# Margin 0: this demo writes and refreshes in one session on the clock of the database. Keep the
# default (NULL = 60 s) where writers may run on other nodes.
sql "CALL vvector.register_index('$IX', '$T', 'id', 'vec', 'del', 'ts', 'l2', 0, 'hnsw');" 2>&1 | sed -n 's/^.*NOTICE [0-9]*: *//p'
t0=$(date +%s%N)
sql "CALL vvector.refresh_index('$IX');" 2>&1 | sed -n 's/^.*NOTICE [0-9]*: *//p'
echo "refresh_index: $(ms $t0) ms"

step "3. The 5 nearest vectors of one query: vvector against the built-in full scan"
t0=$(date +%s%N)
sql "SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='$IX', k=5) FROM $Q q WHERE q.qid = 0 ORDER BY rank;"
echo "vvector (HNSW, precision balanced): $(ms $t0) ms"
t0=$(date +%s%N)
sql "SELECT t.id, VECTOR_L2(t.vec, q.qvec) AS score FROM $T t CROSS JOIN (SELECT qvec FROM $Q WHERE qid = 0) q ORDER BY score, t.id LIMIT 5;"
echo "built-in VECTOR_L2 over every row: $(ms $t0) ms"

step "4. Recall@10 of 100 queries in one statement"
if [ -n "$DIR" ]; then
    REF="SELECT qid, id FROM $SCHEMA.${DS}_gt WHERE rank <= 10 AND qid < 100"
    what="against the ground truth"
else
    REF="SELECT qid, id FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='$IX', k=10, precision='exact') OVER()
         FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $Q WHERE qid < 100) x) e"
    what="against the exact search (precision exact)"
fi
for p in fast balanced best; do
    t0=$(date +%s%N)
    r=$(val "SELECT (COUNT(r.id) / 1000.0)::NUMERIC(5,3) FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
             USING PARAMETERS index_name='$IX', k=10, precision='$p') OVER()
             FROM (SELECT * FROM $SCHEMA.${IX}_snap UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $Q WHERE qid < 100) x) g
             LEFT JOIN ($REF) r ON r.qid = g.qid AND r.id = g.id")
    echo "precision $p: recall@10 $r $what ($(ms $t0) ms for the statement)"
done

step "5. Change the journal without a refresh: add a copy of query 0 as id 999999999, delete the nearest vector"
top=$(val "SELECT id FROM (SELECT vvector.vknn(q.qvec USING PARAMETERS index_name='$IX', k=1, precision='exact') FROM $Q q WHERE q.qid = 0) s")
sql "INSERT INTO $T (id, vec) SELECT 999999999, qvec FROM $Q WHERE qid = 0;
     INSERT INTO $T (id, del) VALUES ($top, TRUE); COMMIT;"
echo "journal rows since the refresh (the delta view ${IX}_delta): $(val "SELECT COUNT(id) FROM $SCHEMA.${IX}_delta")"
for f in snapshot exact; do
    view=${IX}_snap; [ "$f" = exact ] && view=${IX}_delta
    echo "-- freshness=$f (FROM $view)"
    sql "SELECT id, score, rank FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
         USING PARAMETERS index_name='$IX', k=3, freshness='$f') OVER()
         FROM (SELECT * FROM $SCHEMA.$view UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM $Q WHERE qid = 0) x) s ORDER BY rank;"
done
echo "(snapshot: the index as refreshed, id $top still there; exact: the new id at distance 0, id $top gone)"

step "6. Refresh again: incremental, only the changes are read"
t0=$(date +%s%N)
sql "CALL vvector.refresh_index('$IX');" 2>&1 | sed -n 's/^.*NOTICE [0-9]*: *//p'
echo "refresh_index: $(ms $t0) ms"

step "7. Every node: the snapshot it holds, and how much of it is in memory"
sql "SELECT node_name, snapshot_id, vector_count, index_type, loaded, resident_mb, cache_file
     FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='$IX') OVER(PARTITION NODES) FROM vvector.probe) i ORDER BY 1;"
sql "CALL vvector.status('$IX');" 2>&1 | sed -n 's/^.*NOTICE [0-9]*: *//p' | head -4

if [ "$KEEP" = no ]; then
    step "8. Clean up (--keep keeps it)"
    val "CALL vvector.unregister_index('$IX');" > /dev/null 2>&1
    val "DROP SCHEMA $SCHEMA CASCADE;" > /dev/null && echo "index $IX and schema $SCHEMA removed (the cache files stay on the nodes, see unregister_index)"
fi
echo; echo "demo done"
