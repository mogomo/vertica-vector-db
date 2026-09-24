#!/usr/bin/env bash
# Integration test of the vector functions: vector_add, vector_sub, vector_mul, scalar_vector_mul,
# vector_normalize, vector_l1, vector_l2sq, vector_hamming, vector_jaccard, and the transform
# functions vector_sum and vector_avg.
#
#   tests/sql/test_vector_functions.sh [--rows=N] [--schema=NAME] [--echo_only]
#
# On N pairs of random vectors of 16 elements (default 1000): every element of the array results
# equals the SQL expression on the elements; vector_l1 equals the sum of ABS differences, vector_l2sq
# the square of VECTOR_L2; vector_normalize has VECTOR_MAGNITUDE 1 and COSINE_SIMILARITY 1 with its
# input; vector_sum and vector_avg equal SUM and AVG of every element, over the table and per group.
# Also: literal results (bits for Hamming and Jaccard), NULL arguments, ARRAY[INT] and ARRAY[NUMERIC]
# arguments, arrays of 10,000 elements, and the error messages. Test data: schema VVFUNC (or
# --schema=NAME), dropped at the end.
# Connection: vsql reads VSQL_HOST, VSQL_PORT, VSQL_USER, VSQL_PASSWORD, VSQL_DATABASE from the environment.
set -uo pipefail

cd "$(dirname "$0")/../.."

ROWS=1000
DIMS=16
SCHEMA=VVFUNC
ECHO_ONLY=no

for arg in "$@"; do
    case "$arg" in
        --rows=*)      ROWS="${arg#--rows=}" ;;
        --schema=*)    SCHEMA="${arg#--schema=}" ;;
        --echo_only)   ECHO_ONLY=yes ;;
        -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "test_vector_functions.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

. tests/sql/lib.sh
VEC=$(random_vector "$DIMS")

# every_element EXPR: SQL that is true when EXPR (with the placeholder I for the index) holds for
# every element index 0 .. DIMS - 1.
every_element() {
    local i s=""
    for ((i = 0; i < DIMS; i++)); do s+="${s:+ AND }(${1//I/$i})"; done
    echo "$s"
}
sum_elements() {
    local i s=""
    for ((i = 0; i < DIMS; i++)); do s+="${s:+ + }${1//I/$i}"; done
    echo "$s"
}
close() { echo "ABS(($1) - ($2)) <= 1e-12 * GREATEST(1, ABS($2))"; }

echo "== test data: $ROWS pairs of vectors of $DIMS elements"
expect "pairs" "^pairs $ROWS$" "
DROP SCHEMA IF EXISTS $SCHEMA CASCADE;
CREATE SCHEMA $SCHEMA;
CREATE TABLE $SCHEMA.p (id INT, g INT, s FLOAT, a ARRAY[FLOAT], b ARRAY[FLOAT]);
INSERT INTO $SCHEMA.p SELECT id, id % 7, RANDOM() * 10 - 5, $VEC, $VEC FROM ($(row_numbers "$ROWS")) n;
COMMIT;
SELECT 'pairs ' || COUNT(*) FROM $SCHEMA.p;"

echo "== element results against SQL"
for f in "vector_add:+" "vector_sub:-" "vector_mul:*"; do
    name=${f%%:*} op=${f#*:}
    expect "$name equals a[i] $op b[i] for every element of every row" "^bad 0 of $ROWS$" "
SELECT 'bad ' || SUM(CASE WHEN ARRAY_LENGTH(r) = $DIMS AND $(every_element "$(close "r[I]" "a[I] $op b[I]")") THEN 0 ELSE 1 END) || ' of ' || COUNT(*)
FROM (SELECT a, b, vvector.$name(a, b) AS r FROM $SCHEMA.p) x;"
done
expect "scalar_vector_mul equals s x a[i]" "^bad 0 of $ROWS$" "
SELECT 'bad ' || SUM(CASE WHEN $(every_element "$(close "r[I]" "s * a[I]")") THEN 0 ELSE 1 END) || ' of ' || COUNT(*)
FROM (SELECT s, a, vvector.scalar_vector_mul(s, a) AS r FROM $SCHEMA.p) x;"
expect "vector_normalize: magnitude 1, cosine 1 with the input" "^bad 0 of $ROWS$" "
SELECT 'bad ' || SUM(CASE WHEN ABS(VECTOR_MAGNITUDE(r) - 1) <= 1e-12 AND ABS(COSINE_SIMILARITY(r, a) - 1) <= 1e-12 THEN 0 ELSE 1 END) || ' of ' || COUNT(*)
FROM (SELECT a, vvector.vector_normalize(a) AS r FROM $SCHEMA.p) x;"
expect "vector_l1 equals the sum of ABS(a[i] - b[i])" "^bad 0 of $ROWS$" "
SELECT 'bad ' || SUM(CASE WHEN $(close "vvector.vector_l1(a, b)" "$(sum_elements "ABS(a[I] - b[I])")") THEN 0 ELSE 1 END) || ' of ' || COUNT(*) FROM $SCHEMA.p;"
expect "vector_l2sq equals VECTOR_L2(a, b) squared" "^bad 0 of $ROWS$" "
SELECT 'bad ' || SUM(CASE WHEN ABS(vvector.vector_l2sq(a, b) - POWER(VECTOR_L2(a, b), 2)) <= 1e-9 * GREATEST(1, vvector.vector_l2sq(a, b)) THEN 0 ELSE 1 END) || ' of ' || COUNT(*) FROM $SCHEMA.p;"

echo "== literal results"
expect "vector_add of two literals" '^\[4.5,6.25\]$' "SELECT vvector.vector_add(ARRAY[1.5, 2.0], ARRAY[3.0, 4.25]);"
expect "ARRAY[INT] and ARRAY[NUMERIC] arguments are cast" '^\[4.5,6.0\]$' "SELECT vvector.vector_add(ARRAY[1, 2], ARRAY[3.5, 4]);"
expect "scalar_vector_mul" '^\[-3.0,1.0\]$' "SELECT vvector.scalar_vector_mul(-2, ARRAY[1.5, -0.5]);"
expect "vector_normalize" '^\[0.6,0.8\]$' "SELECT vvector.vector_normalize(ARRAY[3, 4]);"
expect "vector_normalize of a zero vector stays zero" '^\[0.0,0.0\]$' "SELECT vvector.vector_normalize(ARRAY[0.0, 0.0]);"
expect "vector_l1 and vector_l2sq" '^7|25$' "SELECT vvector.vector_l1(ARRAY[0, 0], ARRAY[3, 4])::INT || '|' || vvector.vector_l2sq(ARRAY[0, 0], ARRAY[3, 4])::INT;"
expect "vector_hamming and vector_jaccard on 0/1 elements" '^2|0.5$' "SELECT vvector.vector_hamming(ARRAY[1, 0, 1, 1, 0], ARRAY[1, 1, 0, 1, 0]) || '|' || vvector.vector_jaccard(ARRAY[1, 0, 1, 1, 0], ARRAY[1, 1, 0, 1, 0]);"
expect "vector_hamming on packed 64-bit words" '^68$' "SELECT vvector.vector_hamming(ARRAY[-1, 15], ARRAY[0, 255]);"
expect "vector_jaccard of two vectors without bits is 1" '^1$' "SELECT vvector.vector_jaccard(ARRAY[0, 0], ARRAY[0, 0]);"
expect "empty vectors" '^\[\]|0$' "SELECT TO_JSON(vvector.vector_add(ARRAY[]::ARRAY[FLOAT], ARRAY[]::ARRAY[FLOAT])) || '|' || vvector.vector_l1(ARRAY[]::ARRAY[FLOAT], ARRAY[]::ARRAY[FLOAT])::INT;"
expect "a NULL argument gives NULL" '^t|t|t|t$' "
SELECT (vvector.vector_add(NULL::ARRAY[FLOAT], ARRAY[1.0]) IS NULL)::VARCHAR(1) || '|' || (vvector.scalar_vector_mul(NULL, ARRAY[1.0]) IS NULL)::VARCHAR(1)
       || '|' || (vvector.vector_l1(ARRAY[1.0], NULL) IS NULL)::VARCHAR(1) || '|' || (vvector.vector_hamming(NULL, ARRAY[1]) IS NULL)::VARCHAR(1);"
expect "arrays of 10,000 elements" '^10000|10000|10000|0$' "
SELECT ARRAY_LENGTH(vvector.vector_add(v, v)) || '|' || ARRAY_LENGTH(vvector.vector_normalize(v)) || '|' || ARRAY_LENGTH(vvector.scalar_vector_mul(2, v))
       || '|' || vvector.vector_l1(v, v)::INT
FROM (SELECT STRING_TO_ARRAY('[' || REPEAT('1.5,', 9999) || '2]')::ARRAY[FLOAT, 10000] AS v) x;"

echo "== vector_sum and vector_avg"
expect "vector_sum OVER() equals SUM of every element" "^ok$" "
SELECT CASE WHEN $(every_element "$(close "t.vector_sum[I]" "(SELECT SUM(a[I]) FROM $SCHEMA.p)")") THEN 'ok' ELSE 'bad' END
FROM (SELECT vvector.vector_sum(a) OVER() FROM $SCHEMA.p) t;"
expect "vector_avg OVER(PARTITION BY g) equals AVG of every element per group" "^bad 0 of 7$" "
SELECT 'bad ' || SUM(CASE WHEN $(every_element "$(close "t.vector_avg[I]" "r.e_I")") THEN 0 ELSE 1 END) || ' of ' || COUNT(*)
FROM (SELECT g, vvector.vector_avg(a) OVER(PARTITION BY g) FROM $SCHEMA.p) t
JOIN (SELECT g, $(for ((i = 0; i < DIMS; i++)); do printf '%sAVG(a[%d]) AS e_%d' "$([ $i -gt 0 ] && echo ', ')" $i $i; done) FROM $SCHEMA.p GROUP BY g) r ON r.g = t.g;"
expect "NULL vectors are skipped; only NULL vectors give NULL" '^\[2.0,3.0\]|t$' "
SELECT TO_JSON((SELECT vvector.vector_avg(v) OVER() FROM (SELECT ARRAY[1.0, 2.0] AS v UNION ALL SELECT ARRAY[3.0, 4.0] UNION ALL SELECT NULL) x)) || '|'
       || ((SELECT vvector.vector_sum(v) OVER() FROM (SELECT NULL::ARRAY[FLOAT] AS v) x) IS NULL)::VARCHAR(1);"
expect "vector_avg of 10,000-element arrays" '^10000$' "
SELECT ARRAY_LENGTH(vector_avg) FROM (SELECT vvector.vector_avg(v) OVER() FROM (SELECT STRING_TO_ARRAY('[' || REPEAT('1.5,', 9999) || '2]')::ARRAY[FLOAT, 10000] AS v) x) y;"

echo "== errors"
expect "vectors of different lengths" "vector_add: the vectors have different lengths: 2 and 3 elements" "SELECT vvector.vector_add(ARRAY[1.0, 2.0], ARRAY[1.0, 2.0, 3.0]);"
expect "a NULL element" "vector_l1: element 2 of the second vector is NULL" "SELECT vvector.vector_l1(ARRAY[1.0, 2.0], ARRAY[1.0, NULL]);"
expect "a NULL element of the only vector" "vector_normalize: element 1 of the vector is NULL" "SELECT vvector.vector_normalize(ARRAY[NULL, 2.0]::ARRAY[FLOAT]);"
expect "vector_avg: vectors of different lengths" "vector_avg: the vectors have different lengths: 2 and 1 elements" "
SELECT vvector.vector_avg(v) OVER() FROM (SELECT ARRAY[1.0, 2.0] AS v UNION ALL SELECT ARRAY[3.0]) x;"

run_sql "cleanup" "DROP SCHEMA $SCHEMA CASCADE;" > /dev/null

finish_tests test_vector_functions
