# vertica-vector-db (vvector)

Vector search inside Vertica: find the vectors closest to a given vector,
with one SQL function call.

vvector is an extension for Vertica (a C++ UDx library). It builds an index
of the vectors stored in one of your tables, keeps that index inside
Vertica, and loads it on every node. A search can also see the rows written
after the index was built.

**Status: complete (2026-09-26).** No further work is planned. Issues and
pull requests are welcome. Treat it as a preview: test it on your own
systems before you rely on it.

What it does:

- **Fast approximate search (ANN)** with an HNSW index, the default: the 10
  nearest of 1 million vectors in 1.9 ms, finding 98 of every 100 true
  neighbours.
- **Exact search (kNN)**, the same answer as Vertica's own full scan: with a
  flat index in 5 ms instead of 7.7 s; also with `precision='exact'` on any
  index, or with `vscan` on any table, without an index.
- **Four ways to measure distance:** l2 (straight line), cosine, dot product
  and l1.
- **One query or thousands** in one statement.
- **Always up to date if you want:** a search can include the rows written
  since the last refresh, and a refresh adds only the changes.
- **Filtered search** (only the ids you allow) and **range search**
  (everything closer than a given distance).
- **Optional compression** of the vectors to one byte per number (sq8) for
  faster searches.
- **Vector functions** that Vertica lacks: sum, difference, average,
  normalisation, l1, Hamming and Jaccard distance.

How it was tested:

- Exact results are compared with a full scan using Vertica's built-in
  functions; the accuracy (recall) of the approximate search is measured on
  the SIFT1M benchmark.
- **Performance and the full test suite** ran on three Vertica 26.2 systems:
  a single-node Enterprise-mode VM (aarch64, Rocky Linux 9, 8 cores), a
  4-node Enterprise-mode cluster, and an Eon-mode cluster of 3 nodes plus a
  secondary subcluster of 2 (x86_64, Red Hat Enterprise Linux 8). Every
  measurement in this README ([Performance and results](#performance-and-results)
  and the speeds quoted in the text) comes from these three.
- **Functionality only** for the examples of the [Reference](#reference):
  they ran on a fourth system, a single node on Ubuntu 26.04 (x86_64), and
  the two that need a cluster on the Eon cluster. They show that every
  statement works and what it returns, on small example tables.

New to vector search? Start with [Terms](#terms).

Contents: [Terms](#terms) · [Why](#why) · [Quick start](#quick-start) · [Install](#install) ·
[Prepare a table](#prepare-a-table) · [Reference](#reference) ·
[Search](#search) · [Index types and tuning](#index-types-and-tuning) ·
[Freshness explained](#freshness-explained) · [Vector functions](#vector-functions) ·
[Operations](#operations) · [Performance and results](#performance-and-results) ·
[Restrictions](#restrictions-and-not-supported) · [Files](#files)

## Terms

The words this README uses, in plain language, each with what it gives and
what it costs. The question most readers ask first comes first.

### Exact (kNN) or approximate (ANN)?

A **k-nearest-neighbour search (kNN)** returns the k vectors closest to a
query. Strictly, kNN means the true k closest: the same answer you get by
comparing the query with every vector. An **approximate nearest-neighbour
search (ANN)** returns most of them, much faster, because it reads only a
small part of the vectors.

vvector does both. **With the defaults (an `hnsw` index at
`precision='balanced'`) a search is approximate (ANN).** A search is exact
(kNN) with a `flat` index, with `precision='exact'` or `exact=true` on any
index, and with `vscan`. The function names do not tell you which: `vsearch`
and `vknn` do both, depending on the index and the precision.

The 10 nearest of 1,000,000 vectors of 128 dimensions (SIFT1M), one search
(the test VM, 8 cores, time at the client, search function not fenced; see
[Performance and results](#performance-and-results)):

| Search<br><sub>how the 10 nearest are found</sub> | Exact?<br><sub>always the true 10 nearest, or most of them</sub> | Recall@10<br><sub>share of the true 10 nearest found; 1.0 = all</sub> | One search<br><sub>time for one query, seen by the client</sub> |
|---|---|---:|---:|
| Vertica's built-in full scan: `ORDER BY VECTOR_L2(vec, q) LIMIT 10` | exact | 1.0 | 7.7 s |
| `flat` index (without sq8) | exact (kNN) | 1.0 | 5 ms |
| `hnsw` index, `precision='exact'` or `exact=true` | exact (kNN) | 1.0 | reads every vector, like `flat` |
| `hnsw` index, `precision='fast'` | approximate (ANN) | 0.89 | a little less than balanced |
| `hnsw` index, `precision='balanced'` (the default) | approximate (ANN) | 0.98 | 1.9 ms |
| `hnsw` index, `precision='best'` | approximate (ANN) | 0.999 | a little more than balanced |
| `vscan` (no index), on the 4-node test cluster | exact (kNN) | 1.0 | 0.5 to 0.6 s |

(Recall 1.0 means the same ids as the full scan; against the published
ground truth of SIFT1M an exact search scores 0.999, because some vectors
tie.)

How each grows with the table, from 1M to 100M vectors of 128 dimensions on
the 4-node test cluster: the built-in full scan from 14 to 20 s to 364 s,
`vscan` from 0.5 s to 19 s, an exact search on a `flat` index from 32 ms to
1.25 s: all three read every vector, so the time grows with the table. One
HNSW search went from 6.6 ms to 7.6 ms. But its recall at the same setting
falls with the size: at 100M `balanced` finds 0.90 and `best` 0.98, so a
large index needs a larger `ef_search`.

Which to use: exact when every answer must be the true one (an audit, a
legal or financial rule, a test), for small tables (a `flat` search of
100,000 vectors takes well under a millisecond in the engine), and as the
reference when you measure recall. Approximate when speed and many queries
count, and finding 98 of 100 true neighbours is enough (document retrieval,
recommendations, "more like this").

The word "exact" has two more meanings in vvector, and neither makes a
search exact: `freshness='exact'` decides which rows are searched (also
those written since the last refresh, see [Freshness](#terms-freshness)),
and "exact scores" with sq8 means the returned scores are computed from the
real vectors (see [sq8](#terms-sq8)).

### The terms

<a id="terms-vector"></a>**Vector.** A list of numbers that describes one
thing: a document, a picture, a customer. It is stored in an `ARRAY[FLOAT]`
column (`ARRAY[INT]` and `ARRAY[NUMERIC]` work too). Things that are similar
get vectors that are close to each other. The number of elements is the
*dimensions*; every vector of one index has the same number.

<a id="terms-metric"></a>**Metric and score.** How "close" is measured,
fixed per index: `l2` (the straight-line distance, Vertica's `VECTOR_L2`;
smaller is closer), `cosine` (the angle, `COSINE_SIMILARITY`; larger is
closer), `dot` (`DOT_PRODUCT`; larger is closer) or `l1` (the sum of the
absolute differences; smaller is closer). The *score* of a result is the
value the built-in function of the metric returns. Use the metric your
embedding model was made for.

<a id="terms-full-scan"></a>**Full scan.** Vertica's own way, without
vvector: compute the distance to every row, sort, keep the first k.
- Gives: exact results with nothing installed; it is the reference the
  tests of vvector compare with.
- Costs: the whole table is read and sorted for every query: 7.7 s at 1M
  vectors on the test VM, 364 s at 100M on the 4-node test cluster.

<a id="terms-knn"></a>**Exact search (kNN).** Returns the true k nearest:
the same ids as the full scan (two vectors with the same score: the lower
id first). In vvector: a `flat` index, `precision='exact'` or `exact=true`,
`vscan`, and always the rows written since the last refresh.
- Gives: always the right answer, which you can check against the built-in
  functions.
- Costs: every vector is read for every query, so the time grows in
  proportion to the table. vvector reads fast (all cores, all nodes, SIMD
  instructions: 5 ms for 1M vectors), but it cannot skip the reading: 1.25 s
  for 100M on the 4-node test cluster.

<a id="terms-ann"></a>**Approximate search (ANN).** Finds most of the true
k nearest by reading a small part of the vectors: an HNSW search reads a
few thousand vectors out of a million.
- Gives: milliseconds, almost independent of the size of the table (one
  search 6.6 ms at 1M and 7.6 ms at 100M on the 4-node test cluster).
- Costs: it can miss some of the true neighbours and return the next
  closest instead. How many it misses depends on your data, the size of the
  index and the settings: measure it (recall). It needs an index that takes
  memory and time to build.

<a id="terms-recall"></a>**Recall@k.** The share of the true k nearest that
a search returns, averaged over many queries, measured against an exact
search. A recall@10 of 0.98 means 98 of 100 true neighbours were found;
the 2 missed ones were replaced by vectors that are almost as close. An
exact search has recall 1.

<a id="terms-index"></a>**Index (snapshot).** vvector's copy of the vectors
in a compact binary file. `refresh_index` builds it and stores it in the
Vertica table `vvector.snapshot`; every node keeps it as a cache file and
searches it from memory.
- Gives: searches that do not read the table.
- Costs: memory and disk on every node (`CALL vvector.sizing(...)` estimates
  it), and a refresh to take in new rows (or `freshness='exact'`, below).

<a id="terms-flat"></a>**`flat` index.** The vectors themselves, 4 bytes per
number, searched by reading all of them. Exact.
- Gives: exact answers, the smallest index, the fastest refresh (a full
  refresh of 1M vectors 12 s against 47 s for HNSW), nothing to tune.
- Costs: every search reads every vector: 5 ms for one search at 1M, 1.1 s
  for 1000 queries in one statement (HNSW: 28 ms). Choose it for up to about
  100,000 vectors, when every answer must be exact, or when refreshes must
  be as fast as possible.

<a id="terms-hnsw"></a>**`hnsw` index (the default).** A graph: every vector
is linked to some of its nearest neighbours (HNSW, Malkov and Yashunin
2018). A search starts at an entry point and walks along the links towards
the query. Approximate.
- Gives: 1.9 ms for one search at 1M, 1000 queries in 28 ms, and it stays
  fast as the table grows.
- Costs: approximate (recall@10 0.98 at `balanced` on SIFT1M, 0.90 at 100M);
  the links take memory ((2m + 1) x 4 bytes per vector and more: with the
  default m = 16 about a quarter more than 128 float numbers); a slower
  build (a full refresh of 1M vectors 47 s, of 100M 3.6 hours); a vector
  that no link leads to cannot be found by the walk (counted and reported as
  `unreachable`, 0 in the tests).

<a id="terms-precision"></a>**`precision` and `ef_search`.** How hard an HNSW
search looks. `ef_search` is the length of the list of best candidates the
walk keeps; `precision` is a preset of it: `fast` (2 x k, at least 32;
recall 0.89 on SIFT1M), `balanced` (100; 0.98; the default), `best` (400;
0.999), `exact` (no walk: every vector is read).
- Gives: one setting for the trade between speed and recall, per query,
  per session or per index.
- Costs: more recall costs time, most visibly in large batches (1000
  queries: 14 ms at `fast`, 28 ms at `balanced`). On a `flat` index without
  sq8 it changes nothing: that is always exact.

<a id="terms-sq8"></a>**sq8 (int8 quantisation), rescoring, oversampling.**
Every number is also stored as one byte (a code from 0 to 255). A search
first ranks the candidates by these bytes, then *rescores* the best
k x *oversampling* of them with the real float vectors and returns the k
best with their real scores.
- Gives: faster batches (an HNSW index on SIFT1M: 48,900 to 72,600 queries
  per second in the engine at the same recall 0.983); with
  `memory_mode='compact'` a node keeps only the codes, ids and graph in
  memory.
- Costs: the codes come in addition to the floats (1M x 128 as HNSW: 631 MB
  to 757 MB). The candidates are chosen by the bytes, so a `flat` index with
  sq8 is no longer strictly exact below `precision='exact'` (recall 0.999).
  At `precision='fast'` nothing is rescored and the scores are approximate
  too. From 512 dimensions on it loses more recall; at 100M x 128 it was
  slower than the float index.

<a id="terms-vscan"></a>**`vscan`.** An exact search over any table or
query result, with no index, spread over all nodes.
- Gives: nothing to register, refresh or keep in memory; any SQL filter; it
  always sees the committed rows.
- Costs: every row is read for every statement: 0.5 to 0.6 s at 1M and 19 s
  at 100M on the 4-node test cluster (the built-in full scan: 14 to 20 s and
  364 s).

<a id="terms-journal"></a>**Journal, refresh, delta.** The table is a
journal: adding or changing a vector is an INSERT, deleting is an INSERT
with a delete flag, and the newest row per id wins. `refresh_index` builds
the index from the rows up to a *boundary*; the rows after it are the
*delta*, returned by the view `<index>_delta`.
- Gives: writers are never blocked by a refresh; a refresh after a few
  changes adds only those and sends only the changed bytes to the nodes.
- Costs: the table grows with every change (partition it by day so that old
  rows can be dropped); a refresh takes time (above).

<a id="terms-freshness"></a>**Freshness: `snapshot` or `exact`.** Which rows
a search sees. This is not about exact or approximate neighbours: an HNSW
search with `freshness='exact'` is still approximate over the index.
- `snapshot` (the default) searches the index of the last refresh. Gives:
  the fastest statement. Costs: changes since the refresh are not seen.
- `exact`, over the `_delta` view, also applies the rows written since the
  last refresh; those rows are always searched exactly. Gives: every
  committed change is seen. Costs: the delta view is read (1.1 ms more per
  statement with an empty delta on the test VM), more as the delta grows
  until the next refresh.

<a id="terms-range"></a>**Range search.** All vectors closer than a
`radius`, at most k. On an HNSW index it is approximate like any HNSW
search; add `exact=true` for a complete answer.

<a id="terms-filtered"></a>**Filtered search.** Only the ids you allow (an
allow-list, for example the documents of one customer). A small allow-list
is searched exactly; a large one through the index with the other vectors
skipped, which is approximate on HNSW.

<a id="terms-fenced"></a>**Fenced, unfenced, mixed.** Where the functions
run. Fenced: in a separate process. Gives: a fault cannot stop the node.
Costs: about 6 ms per statement (7.1 ms instead of 1.9 ms for one HNSW
search). Unfenced: inside the Vertica process, the fastest, but a fault
stops the node. `mixed`: build and load fenced, search not fenced (see
[Install](#install)).

### Algorithms: the ones vvector uses and the others you will meet

Vector databases and libraries combine a few families of algorithms: a
*scan* (exact), an *index* that narrows the search (a graph, clusters, trees
or hashes; approximate), and *compression* (quantisation) that makes each
vector cheaper to read. vvector uses a scan, one graph (HNSW) and one
compression (sq8). The others are listed so that you can compare vvector
with the systems that use them; they are **not** in vvector.

| Algorithm<br><sub>the method</sub> | Family<br><sub>how it avoids reading everything</sub> | Exact?<br><sub>always the true nearest, or most of them</sub> | In vvector<br><sub>whether and where vvector uses it</sub> | Where you meet it<br><sub>well-known libraries and databases</sub> |
|---|---|---|---|---|
| Brute-force scan with top-k selection | scan | exact | yes: `flat`, `precision='exact'`, `vscan`, the journal rows | every system (often called "flat" or "exhaustive") |
| HNSW (hierarchical navigable small world) | graph | approximate | yes: `hnsw`, the default | hnswlib, FAISS, USearch, pgvector, Milvus, most vector databases |
| Scalar quantisation (int8, "SQ8") with rescoring | compression | approximate candidates | yes: `quantization='sq8'` | FAISS, Milvus, USearch, most vector databases |
| IVF (inverted file: k-means clusters) | clusters | approximate | no | FAISS, Milvus, pgvector (`ivfflat`) |
| Product quantisation (PQ), often with IVF (IVF-PQ) | compression | approximate | no | FAISS, Milvus |
| Binary quantisation (1 bit per number) | compression | approximate | no (`vector_hamming` compares bit vectors) | several vector databases |
| DiskANN (Vamana graph on SSD) | graph | approximate | no | Microsoft DiskANN, Milvus |
| ScaNN (anisotropic quantisation) | clusters + compression | approximate | no | Google ScaNN |
| Trees: KD-tree, random projection trees (Annoy) | trees | KD-tree exact, Annoy approximate | no | scikit-learn, Spotify's Annoy |
| LSH (locality-sensitive hashing) | hashes | approximate | no | older systems, research |

**In vvector.**

- **Brute-force scan with top-k selection.** Compute the score of every
  vector, keep the k best in a small heap. vvector reads the vectors in
  blocks with SIMD instructions on all cores and nodes, and merges the
  partial lists so that the result does not depend on the number of
  threads. Gives: exact, simple, no build. Costs: time in proportion to the
  number of vectors.
- **HNSW** (Malkov and Yashunin, 2018). Every vector gets links to near
  vectors on the lowest level; a few vectors also sit on higher levels with
  longer links, like motorways above local roads. A search enters at the
  top, moves greedily towards the query, goes down a level, and on the
  lowest level keeps a list of the `ef_search` best candidates. vvector
  builds it the way hnswlib does (m = 16, ef_construction = 200) and
  reaches the same recall and speed (see
  [Performance and results](#performance-and-results)). Gives: the best
  known trade between speed and recall in memory, and inserts without a
  rebuild. Costs: memory for the links, a slow build, recall that falls
  slowly with the size of the index, and deletes that leave holes (vvector
  marks them as tombstones and rebuilds when there are too many).
- **Scalar quantisation (sq8) with rescoring.** Every number becomes one
  byte in one range for the whole index; candidates are ranked by the bytes
  and the best k x oversampling of them rescored with the floats. Gives: a
  quarter of the bytes to read per candidate, integer arithmetic. Costs:
  stored in addition to the floats in vvector; less recall from about 512
  dimensions on unless more candidates are rescored.
- **Cosine by normalisation.** For `cosine` the vectors are scaled to length
  1 at the build; the search then computes a dot product, which is cheaper.

**Not in vvector** (see [Restrictions](#restrictions-and-not-supported)).

- **IVF (inverted file).** k-means divides the vectors into clusters
  (`nlist`); a search looks only at the clusters closest to the query
  (`nprobe`). Gives: little memory beyond the vectors, a fast build. Costs:
  lower recall than HNSW at the same speed; the clusters are trained once and
  fit worse as the data changes, so the index needs retraining.
- **Product quantisation (PQ).** A vector is cut into pieces, and each piece
  is replaced by the number of its nearest of 256 learned pieces (1 byte):
  128 floats (512 bytes) become 16 to 64 bytes. Gives: very large indexes in
  memory. Costs: training, approximate distances, and a lower recall unless
  the candidates are rescored with the full vectors.
- **Binary quantisation.** One bit per number (its sign), compared with the
  Hamming distance: 32 times smaller than floats. Gives: very fast and small.
  Costs: works well only for some embedding models, needs rescoring.
- **DiskANN (Vamana).** A graph like HNSW, but one level, built to be read
  from SSD with compressed vectors in memory. Gives: billions of vectors on
  one machine. Costs: every hop may read the disk; updates are harder.
- **ScaNN.** Clusters plus a quantisation tuned to keep the ranking of
  dot-product scores right. Gives: high throughput for dot-product search.
  Costs: training and many settings.
- **Trees (KD-tree, Annoy).** Split the space by planes, again and again.
  Gives: a KD-tree is exact and fast in few dimensions (up to about 20).
  Costs: in the hundreds of dimensions of embeddings a KD-tree reads almost
  everything; Annoy is approximate and its index cannot be updated, only
  rebuilt.
- **LSH (locality-sensitive hashing).** Random hash functions that give close
  vectors the same hash with high probability. Gives: proven bounds, easy to
  distribute. Costs: many hash tables for a good recall, a lot of memory;
  graphs and IVF mostly replaced it.
- **Beyond single dense vectors:** *sparse vectors* (most elements zero, as
  in keyword-weight models), *several vectors per item* (one per sentence or
  per token, as in ColBERT), *hybrid search* (keywords and vectors ranked
  together, for example by reciprocal rank fusion) and *GPU* search. vvector
  supports none of these; a SQL join or `UNION` of your own can combine a
  vector search with other conditions.

**Vector analytics in vvector.** Besides search, the vector functions do the
arithmetic Vertica lacks: sum, difference, scaling, normalisation, the
average of a group (a *centroid*: the "typical" vector of a customer or a
topic), l1, squared l2, Hamming and Jaccard distance
([Vector functions](#vector-functions)). The recipes in [Search](#search)
use them for recommendation by centroids ("more like these, less like
that"), the best match per group and near-duplicates.

## Why

Vertica 26.2 has no vector type and no vector index. Vectors are stored as
arrays (`ARRAY[FLOAT]`, `ARRAY[INT]` or `ARRAY[NUMERIC]`), and the built-in
functions `COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_L2` and
`VECTOR_MAGNITUDE` compare two arrays. A nearest-neighbour query in SQL is a
full scan of the table:

    SELECT id, VECTOR_L2(vec, ARRAY[0.1, 0.2, 0.3]) AS distance
    FROM app.docs ORDER BY distance LIMIT 10;

On 1,000,000 vectors of 128 dimensions that takes 7.7 seconds on the test
machine. vvector answers the same query in 1.9 ms with an HNSW index (7.1 ms
when the search function is fenced), an approximate search that finds 98 of
100 true neighbours on this data (recall@10 0.98), or with exactly the same
result in 5 ms with a flat index (12 ms fenced). The index is a snapshot of
the vectors that is stored in Vertica, cached on every node, and kept
current: a query can also apply the rows written since the last refresh.

## Quick start

On a Vertica node, with the environment variables of `vsql` set (see
[Install](#install)):

    make && make test && make deploy

    vsql -c "CREATE SCHEMA app;
             CREATE TABLE app.docs (id INT NOT NULL, vec ARRAY[FLOAT], del BOOLEAN NOT NULL DEFAULT FALSE,
                                    ts TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP());
             INSERT INTO app.docs (id, vec) VALUES (1, ARRAY[1.0, 0.0, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (2, ARRAY[0.9, 0.1, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (3, ARRAY[0.0, 1.0, 0.0]);
             INSERT INTO app.docs (id, vec) VALUES (4, ARRAY[0.0, 0.0, 1.0]);
             INSERT INTO app.docs (id, vec) VALUES (5, ARRAY[0.5, 0.5, 0.0]); COMMIT;"
    vsql -c "CALL vvector.register_index('docs', 'app.docs', 'id', 'vec', 'del', 'ts', 'cosine', 0);"
    vsql -c "CALL vvector.refresh_index('docs');"
    vsql -c "SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                    USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
             FROM app.docs_snap;"

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

`register_index` creates an HNSW index unless you ask for `flat` (see
[Index types and tuning](#index-types-and-tuning)). The last argument, the
margin, is 0 here so that the refresh builds the rows written just before it;
that is safe on one node with `CLOCK_TIMESTAMP()` versions. With the default
(NULL = 60 s) a refresh builds the rows older than 60 seconds and serves the
newer ones through the delta (see [Freshness explained](#freshness-explained)). The same query with the
built-in function returns the same ids and scores (vvector computes in 32-bit
floats, so scores agree to about 7 digits):

    SELECT id, COSINE_SIMILARITY(vec, ARRAY[1, 0.2, 0]) AS score FROM app.docs ORDER BY score DESC LIMIT 3;

     id |       score
    ----+-------------------
      2 | 0.996240588195683
      1 |  0.98058067569092
      5 | 0.832050294337844

## Install

### Prerequisites

- Vertica 26.x (tested: 26.2.0-1 single node, 26.2.0-2 Eon with 3 nodes and a secondary subcluster of 2,
  26.2.0-3 Enterprise mode with 4 nodes; the examples of the Reference also on
  26.2.0-3 single node on Ubuntu 26.04, which Vertica does not list as a
  supported platform) with the C++ SDK in `/opt/vertica/sdk`
  (another place: `make SDK_HOME=...`).
- g++ with C++17 (tested: 11.5 on aarch64, 8.5 and 15.2 on x86_64; with 15.2
  the Vertica SDK warns that it was not tested with GCC 14 and later) and GNU make, on a
  Vertica node: `CREATE LIBRARY` reads the .so from the initiator node's file
  system and copies it to the other nodes. No CPU flags are needed: on x86_64
  the distance code is compiled for SSE2, AVX2 and AVX-512 and the best one is
  picked at load time, with bit-identical results on every level.
- Step by step on x86_64 (Red Hat 8, Eon cluster), with the expected output:
  [docs/build-x86.md](docs/build-x86.md).
- A database user that may create a schema, a library, functions and a role
  (dbadmin, or a user with those rights).
- For the tests and the benchmark only: Vertica's packages `VectorOps`
  (`VECTOR_L2`, `COSINE_SIMILARITY`, `DOT_PRODUCT`: the tests compare with
  them) and `approximate` (`APPROXIMATE_PERCENTILE`: the benchmark's medians).
  A new database usually has them; if not, see
  [docs/build-x86.md](docs/build-x86.md), "Problems seen". vvector itself
  needs neither.

### Build, test, deploy

    make                      # build/libvvector.so
    make test                 # engine unit tests, no database needed
    make deploy               # install into the database, fenced (the default)
    make deploy FENCED=no     # every function inside the Vertica process
    make deploy FENCED=mixed  # vbuild, vload, vconfig, vnode fenced; vsearch, vknn, vinfo, vversion and the vector functions not fenced
    make deploy SEARCH=public # searching for every user (default: the role vvector_search)
    make undeploy             # remove the library and its functions; tables and data stay
    tests/sql/run_all.sh      # integration tests: all unfenced, a short set fenced (creates test schemas)

`tests/sql/run_all.sh --complete` runs every test in all three modes. `run_all.sh`
deploys the library again in every mode and ends with
the default deploy (fenced, search for the role only). After a deploy with
`FENCED=mixed` or `SEARCH=public`, run `make deploy` with your settings again
when the tests are done.

What the modes mean:

| Mode | Where the functions run | For | Risk |
|---|---|---|---|
| `yes` (default) | a separate fenced process per session | safety first | none for the node; about 6 ms more per statement |
| `mixed` | build and load fenced, search in the Vertica process | single searches with low latency | a fault in vsearch, vknn, vinfo, vversion or a vector function would stop the node |
| `no` | everything in the Vertica process | the lowest latency and fastest refresh | a fault in any vvector function would stop the node |

The measurements behind this are in [Performance and results](#performance-and-results).

Every script reads the connection from the environment: `VSQL_HOST`,
`VSQL_PORT`, `VSQL_USER`, `VSQL_PASSWORD`, `VSQL_DATABASE`. Every script
accepts `--help`, and every script that changes the database accepts
`--echo_only`: it prints the commands and changes nothing.

### What the install creates

Everything is in schema `vvector`, except the functions that build and load
snapshots, which are in schema `vvector_admin`:

| Object | What it is |
|---|---|
| `vvector.snapshot` | table (segmented by snapshot_id and byte_offset): the index snapshots as pieces of at most 8 MB; a full build stores the whole snapshot, an incremental refresh only the bytes that changed (a patch on the previous snapshot); the table holds the active chain: the last whole copy and the patches after it |
| `vvector.manifest` | table: one row per index with its source, options and state |
| `vvector.probe` | table (8192 rows, segmented): makes node-wise functions run once on every node |
| `vvector.snapshot_seq` | sequence of snapshot ids (never reused) |
| role `vvector_admin` | may build, load and manage indexes; holds `vvector_search` |
| role `vvector_search` | may search (the functions in `vvector`) |
| functions in `vvector` | `vsearch`, `vknn`, `vscan`, `vinfo`, `vversion` (search and information) |
| functions in `vvector_admin` | `vbuild`, `vload`, `vconfig`, `vnode` (build and load; `refresh_index` and `load_all` call them) |
| procedures | `register_index`, `set_index_options`, `set_journal_replica`, `refresh_index`, `load_all` (one index, or all without an argument), `status`, `sizing`, `schedule_refresh`, `unregister_index` |

Rights: searching needs the role `vvector_search`, building and managing
indexes the role `vvector_admin`; see [vvector_search](#vvector_search) and
[vvector_admin](#vvector_admin).

## Prepare a table

vvector indexes a table with an id and a vector column. Changes are written
as new rows: the table is a journal.

    id | vec                | del   | ts
    ---+--------------------+-------+---------------------------
     7 | [0.1, 0.2, ...]    | false | 2026-09-20 10:00:00+00     vector 7 added
     8 | [0.4, 0.1, ...]    | false | 2026-09-20 10:05:00+00     vector 8 added
     7 | [0.3, 0.3, ...]    | false | 2026-09-21 09:00:00+00     vector 7 changed: the newer row wins
     8 |                    | true  | 2026-09-22 11:00:00+00     vector 8 deleted

- **id**: INT (64-bit), unique per vector. Business keys and payload stay in
  your own tables; join them to the results by id. Rows with a NULL id are
  ignored (by the refresh and by the delta view).
- **vec**: `ARRAY[FLOAT]` (recommended), `ARRAY[INT]` or `ARRAY[NUMERIC]`.
  Every vector of one index has the same number of elements. vvector stores
  and computes in 32-bit floats.
- **del** (optional): BOOLEAN, true = deleted; or INT, +1 added and -1 deleted.
- **ts** (optional): the version. For every id the row with the latest
  version wins; with equal versions a delete wins. Use `TIMESTAMPTZ NOT NULL
  DEFAULT CLOCK_TIMESTAMP()`: the database sets the time of the write, which
  is what makes the results exact (see [Freshness](#freshness-explained)).
  TIMESTAMP and increasing INT versions are accepted too. The column must be
  NOT NULL: `register_index` refuses one that is not (a row without a version
  would be in no delta and no refresh).

Recommended table, partitioned by the date of the version so new rows stay in
their own storage and are found quickly:

    CREATE TABLE app.docs (
        id   INT NOT NULL,
        vec  ARRAY[FLOAT],
        del  BOOLEAN NOT NULL DEFAULT FALSE,
        ts   TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP()
    )
    ORDER BY id SEGMENTED BY HASH(id) ALL NODES
    PARTITION BY (ts AT TIME ZONE 'UTC')::DATE
    GROUP BY CALENDAR_HIERARCHY_DAY((ts AT TIME ZONE 'UTC')::DATE, 2, 2);

    INSERT INTO app.docs (id, vec) VALUES (7, ARRAY[0.1, 0.2, 0.3]);   -- add or change
    INSERT INTO app.docs (id, del) VALUES (8, TRUE);                    -- delete

(These two statements show the pattern. The examples of this manual use the
table of the Quick start and add their own rows where they say so; they give
the search results shown whether or not you ran these two; only the listing
of the delta view in [With the changes since the refresh](#with-the-changes-since-the-refresh)
then also shows ids 7 and 8.)

To keep the journal append-only, grant the users that write to it INSERT
only, no UPDATE or DELETE on the table.

A physical `DELETE` or `UPDATE` on the table (or a dropped partition) is
allowed, but queries see it only after the next refresh. Any physical change
to rows up to the boundary of the last refresh is noticed by the next refresh
(with the default `verify_every` 1), which then rebuilds the index in full
(see [refresh_index](#refresh_index)). Rows after the boundary are read by
every refresh anyway. A table without a version column is a **static index**: queries see the
snapshot only, every id must appear once, changes show up at the next refresh,
and every refresh is a full build.

## Reference

Every statement and function of vvector, in the order you use them. Each
entry has the same parts:

1. **What it does**, when you need it, and its limits.
2. **Syntax**: the exact form and every argument or parameter.
3. **Two examples**: a use case, all the statements it needs, and the
   output. Every example was run on Vertica 26.2.0-3 (one node, Ubuntu,
   x86_64); the two marked **Eon** ran on the 5-node Eon test cluster. These
   runs are functionality tests on small example tables: they show how each
   statement works and what it returns. Your snapshot ids, times and memory
   figures will differ. The measurements, from the three test systems named
   at the top, are in [Performance and results](#performance-and-results).

| Group | Entries |
|---|---|
| [Install and rights](#install-and-rights) | [make deploy](#make-deploy) · [vvector_search](#vvector_search) · [vvector_admin](#vvector_admin) |
| [Set up and keep an index](#set-up-and-keep-an-index) | [sizing](#sizing) · [register_index](#register_index) · [The views _snap and _delta](#the-views-_snap-and-_delta) · [refresh_index](#refresh_index) · [set_index_options](#set_index_options) · [set_journal_replica](#set_journal_replica) · [schedule_refresh](#schedule_refresh) · [status](#status) · [load_all](#load_all) · [unregister_index](#unregister_index) |
| [Search](#search-functions) | [vsearch](#vsearch) · [vknn](#vknn) · [vscan](#vscan) · [Session defaults](#session-defaults) |
| [Inspect](#inspect) | [vinfo](#vinfo) · [vversion](#vversion) |
| [Vector functions](#vector-functions) | [vector_add](#vector_add) · [vector_sub](#vector_sub) · [vector_mul](#vector_mul) · [scalar_vector_mul](#scalar_vector_mul) · [vector_normalize](#vector_normalize) · [vector_l1](#vector_l1) · [vector_l2sq](#vector_l2sq) · [vector_hamming](#vector_hamming) · [vector_jaccard](#vector_jaccard) · [vector_sum](#vector_sum) · [vector_avg](#vector_avg) |
| [Build and load by hand](#build-and-load-by-hand) | [vnode](#vnode) · [vconfig](#vconfig) · [vbuild](#vbuild) · [vload](#vload) |

The procedures that are not listed (`register_index_core`, `refresh_index_core`,
`refresh_index_run`, `set_index_options_core`, `load_all_core`, `load_on_nodes`,
`make_views`, `make_views_core`, `push_options`, `apply_replica`,
`apply_replica_core`) are internal: the procedures above call them. Do not
call them yourself.

### Example data

The examples use the tables below, in a schema `ref` of their own. Run this
once; each example then shows every further statement it needs, in the order
of this reference (an example may rely on the ones before it, and says so).

- `ref.articles`: articles as a journal (see [Prepare a table](#prepare-a-table)),
  with 4-number vectors that say how much an article is about databases,
  cooking, travel and sport. It becomes the index `articles`.
- `ref.article_info`: title, language and customer of each article.
- `ref.products`: a product catalogue without versions (vector: price level,
  sport, outdoor). It becomes the index `products`.
- Small tables for the vector functions.

The statements:

    CREATE SCHEMA ref;

    -- the articles, as a journal (index "articles"): vec = [databases, cooking, travel, sport]
    CREATE TABLE ref.articles (
        id   INT NOT NULL,
        vec  ARRAY[FLOAT],
        del  BOOLEAN NOT NULL DEFAULT FALSE,
        ts   TIMESTAMPTZ NOT NULL DEFAULT CLOCK_TIMESTAMP()
    ) ORDER BY id SEGMENTED BY HASH(id) ALL NODES;
    INSERT INTO ref.articles (id, vec) VALUES (1, ARRAY[0.9, 0.1, 0.0, 0.0]);
    INSERT INTO ref.articles (id, vec) VALUES (2, ARRAY[0.8, 0.0, 0.1, 0.1]);
    INSERT INTO ref.articles (id, vec) VALUES (3, ARRAY[0.0, 0.9, 0.1, 0.0]);
    INSERT INTO ref.articles (id, vec) VALUES (4, ARRAY[0.1, 0.8, 0.0, 0.1]);
    INSERT INTO ref.articles (id, vec) VALUES (5, ARRAY[0.0, 0.1, 0.9, 0.0]);
    INSERT INTO ref.articles (id, vec) VALUES (6, ARRAY[0.0, 0.0, 0.6, 0.7]);
    INSERT INTO ref.articles (id, vec) VALUES (7, ARRAY[0.0, 0.1, 0.1, 0.9]);
    INSERT INTO ref.articles (id, vec) VALUES (8, ARRAY[0.0, 0.6, 0.7, 0.0]);

    -- what the articles are (joined to the results by id)
    CREATE TABLE ref.article_info (id INT, title VARCHAR(40), lang CHAR(2), customer VARCHAR(10));
    INSERT INTO ref.article_info VALUES (1, 'Vertica projections explained', 'en', 'acme');
    INSERT INTO ref.article_info VALUES (2, 'Tuning a column store', 'en', 'acme');
    INSERT INTO ref.article_info VALUES (3, 'Bread baking at home', 'en', 'globex');
    INSERT INTO ref.article_info VALUES (4, 'Pasta in ten minutes', 'it', 'globex');
    INSERT INTO ref.article_info VALUES (5, 'A week in Lisbon', 'en', 'acme');
    INSERT INTO ref.article_info VALUES (6, 'Hiking the Alps', 'de', 'initech');
    INSERT INTO ref.article_info VALUES (7, 'Marathon training', 'en', 'initech');
    INSERT INTO ref.article_info VALUES (8, 'Street food in Bangkok', 'en', 'globex');
    INSERT INTO ref.article_info VALUES (9, 'Databases on the road', 'en', 'initech');
    INSERT INTO ref.article_info VALUES (10, 'Cooking for runners', 'en', 'initech');

    -- the live articles: the latest row per id, deletes left out
    CREATE VIEW ref.articles_live AS
    SELECT id, vec FROM (SELECT id, vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, del DESC) AS rn
                         FROM ref.articles) j
    WHERE rn = 1 AND NOT del;

    -- questions to search with
    CREATE TABLE ref.questions (qid INT, question VARCHAR(40), qvec ARRAY[FLOAT]);
    INSERT INTO ref.questions VALUES (100, 'How do I speed up queries?', ARRAY[0.85, 0.0, 0.1, 0.05]);
    INSERT INTO ref.questions VALUES (200, 'What can I cook tonight?', ARRAY[0.05, 0.9, 0.05, 0.0]);
    INSERT INTO ref.questions VALUES (300, 'Where can I travel to run?', ARRAY[0.0, 0.0, 0.6, 0.6]);

    -- a static product catalogue (index "products"): vec = [price level, sport, outdoor]
    CREATE TABLE ref.products (id INT NOT NULL, name VARCHAR(20), vec ARRAY[FLOAT]);
    INSERT INTO ref.products VALUES (10, 'running shoes', ARRAY[0.4, 0.9, 0.6]);
    INSERT INTO ref.products VALUES (11, 'trail shoes',   ARRAY[0.5, 0.8, 0.9]);
    INSERT INTO ref.products VALUES (12, 'rain jacket',   ARRAY[0.6, 0.3, 0.9]);
    INSERT INTO ref.products VALUES (13, 'office chair',  ARRAY[0.7, 0.0, 0.0]);
    INSERT INTO ref.products VALUES (14, 'yoga mat',      ARRAY[0.2, 0.7, 0.1]);
    INSERT INTO ref.products VALUES (15, 'tent',          ARRAY[0.8, 0.2, 1.0]);

    -- small tables for the vector functions
    CREATE TABLE ref.likes (user_id INT, article_id INT);
    INSERT INTO ref.likes VALUES (1, 1);
    INSERT INTO ref.likes VALUES (1, 2);
    INSERT INTO ref.likes VALUES (2, 3);
    INSERT INTO ref.likes VALUES (2, 8);
    CREATE TABLE ref.article_tags (id INT, tags ARRAY[INT]);       -- tags: sql, performance, food, travel, sport, outdoor
    INSERT INTO ref.article_tags VALUES (1, ARRAY[1, 1, 0, 0, 0, 0]);
    INSERT INTO ref.article_tags VALUES (5, ARRAY[0, 0, 0, 1, 0, 0]);
    INSERT INTO ref.article_tags VALUES (6, ARRAY[0, 0, 0, 1, 1, 1]);
    INSERT INTO ref.article_tags VALUES (7, ARRAY[0, 0, 0, 0, 1, 1]);
    INSERT INTO ref.article_tags VALUES (8, ARRAY[0, 0, 1, 1, 0, 0]);
    CREATE TABLE ref.fingerprints (id INT, bits ARRAY[INT]);        -- one bit per element
    INSERT INTO ref.fingerprints VALUES (1, ARRAY[1, 0, 1, 1, 0, 0, 1, 0]);
    INSERT INTO ref.fingerprints VALUES (2, ARRAY[1, 0, 1, 1, 0, 0, 1, 1]);
    INSERT INTO ref.fingerprints VALUES (3, ARRAY[0, 1, 0, 0, 1, 1, 0, 1]);
    CREATE TABLE ref.simhash (id INT, h ARRAY[INT]);                -- 64 bits in one element
    INSERT INTO ref.simhash VALUES (1, ARRAY[1234567890123456789]);
    INSERT INTO ref.simhash VALUES (2, ARRAY[1234567890123456781]);
    INSERT INTO ref.simhash VALUES (3, ARRAY[-987654321987654321]);
    CREATE TABLE ref.usage (customer VARCHAR(10), day DATE, vec ARRAY[FLOAT]);  -- queries, loads, exports per day
    INSERT INTO ref.usage VALUES ('acme',   '2026-09-01', ARRAY[120, 3, 1]);
    INSERT INTO ref.usage VALUES ('acme',   '2026-09-02', ARRAY[80, 5, 0]);
    INSERT INTO ref.usage VALUES ('globex', '2026-09-01', ARRAY[10, 40, 2]);
    INSERT INTO ref.usage VALUES ('globex', '2026-09-02', ARRAY[20, 35, 4]);
    COMMIT;

### Install and rights

#### make deploy

Builds the library and installs it into the database: the schemas `vvector`
and `vvector_admin`, the tables, the roles, the functions and the procedures
(see [What the install creates](#what-the-install-creates)).

- **When you need it:** once, and again after every update of the source.
  Run it on a Vertica node, as a database user who may create schemas,
  libraries, functions and roles (dbadmin).
- **Limits:** needs g++ with C++17, GNU make and the Vertica SDK on that node.
  `CREATE LIBRARY` copies the library to the other nodes. `make undeploy`
  removes the library and the functions; tables, snapshots and roles stay.

**Syntax**

    make [SDK_HOME=/opt/vertica/sdk]
    make test [DATA_DIR=<directory with SIFT1M>]
    make deploy [FENCED=yes|no|mixed] [SEARCH=role|public]
    make undeploy

| Setting | Values | Default | Meaning |
|---|---|---|---|
| FENCED | yes, no, mixed | yes | where the functions run: `yes` every function in a fenced process (a fault cannot stop the node; about 6 ms more per statement); `no` every function inside the Vertica process; `mixed` build and load fenced, search inside the process |
| SEARCH | role, public | role | who may search: the role `vvector_search`, or every user |
| SDK_HOME | a directory | /opt/vertica/sdk | the Vertica SDK |
| DATA_DIR | a directory | none | `make test` also runs the SIFT1M recall test |

The connection comes from the environment: `VSQL_HOST`, `VSQL_PORT`,
`VSQL_USER`, `VSQL_PASSWORD`, `VSQL_DATABASE`.

**Example 1: install on a new database.** Build, run the engine tests, and
install with the defaults (fenced, search through the role). The last lines
of the output list the functions and whether they are fenced:

    make && make test && make deploy

     library_version | format_version |                           build_flags
    -----------------+----------------+-----------------------------------------------------------------
     0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 x86_64 g++-15.2.0 kernels=avx2

         function_name     | is_fenced
    -----------------------+-----------
     vvector.vector_add    | t
     vvector.vector_avg    | t
     vvector.vinfo         | t
     vvector.vknn          | t
     vvector.vscan         | t
     vvector.vsearch       | t
     vvector.vversion      | t
     vvector_admin.vbuild  | t
     vvector_admin.vconfig | t
     vvector_admin.vload   | t
     vvector_admin.vnode   | t

`kernels=avx2` is the distance code the CPU of the node chose (sse2, avx2 or
avx512 on x86_64; the results are the same on every level).

**Example 2: lower latency for single searches.** An application that runs
one search per request saves the fenced process's cost (about 6 ms per
statement) with `FENCED=mixed`: the search functions run inside Vertica,
the build and load functions stay fenced. Check the result in the catalog,
and go back to the default with `make deploy`:

    make deploy FENCED=mixed

    SELECT schema_name || '.' || function_name AS function_name, MIN(is_fenced::INT)::BOOLEAN AS is_fenced
    FROM user_functions
    WHERE schema_name IN ('vvector', 'vvector_admin') AND function_name IN ('vsearch', 'vknn', 'vbuild', 'vload', 'vector_add')
    GROUP BY 1 ORDER BY 1;

        function_name     | is_fenced
    ----------------------+-----------
     vvector.vector_add   | f
     vvector.vknn         | f
     vvector.vsearch      | f
     vvector_admin.vbuild | t
     vvector_admin.vload  | t

    make deploy

A fault in an unfenced function stops the node: use `mixed` or `no` when the
latency matters more than that risk (measurements in
[Performance and results](#performance-and-results)).

#### vvector_search

The role that may search: `vsearch`, `vknn`, `vscan`, `vinfo`, `vversion` and
the vector functions (everything in schema `vvector` except the procedures).

- **When you need it:** for every user who searches or uses a vector
  function (unless deployed with `SEARCH=public`).
- **Limits:** the role covers every index: whoever has it can search any
  index by its name, without a view. The views protect the journal rows
  only: a search that applies the rows written since the last refresh
  (`freshness='exact'`) also needs SELECT on the `_delta` view of that index.
  Vertica cannot grant a single function that has an ARRAY argument, so the
  rights are given per schema. `vvector_admin` holds this role.

**Syntax**

    GRANT vvector_search TO user_name;
    ALTER USER user_name DEFAULT ROLE vvector_search;   -- or, per session: SET ROLE vvector_search;
    REVOKE vvector_search FROM user_name;

`make deploy SEARCH=public` gives the search functions to every user
instead (a role granted to PUBLIC is not enabled for anyone in Vertica, so
`GRANT vvector_search TO PUBLIC` does not do that).

**Example 1: an analyst who searches.** A new user gets the role as a
default role, and SELECT on the table with the titles:

    CREATE USER analyst IDENTIFIED BY 'Analyst_pw1';
    GRANT vvector_search TO analyst;
    ALTER USER analyst DEFAULT ROLE vvector_search;
    GRANT USAGE ON SCHEMA ref TO analyst;
    GRANT SELECT ON ref.article_info TO analyst;

Connected as `analyst`:

    SELECT n.rank, i.title
    FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='articles', query='[0.0, 0.1, 0.9, 0.0]', k=2)
          FROM dual) n
    JOIN ref.article_info i USING (id)
    ORDER BY n.rank;

     rank |         title
    ------+------------------------
        1 | A week in Lisbon
        2 | Street food in Bangkok

**Example 2: an analyst who also sees the newest rows.** A search with
`freshness='exact'` reads the `_delta` view, which shows journal rows of the
source table: the role alone does not give it. As `analyst`:

    SELECT COUNT(*) FROM ref.articles_delta;

    ERROR 4367:  Permission denied for relation articles_delta

The owner of the index grants the views:

    GRANT SELECT ON ref.articles_snap, ref.articles_delta TO analyst;

Now, as `analyst`, the search sees article 10, written after the last
refresh:

    SELECT r.rank, r.id, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='articles', query='[0.0, 0.6, 0.0, 0.6]', k=2, freshness='exact') OVER()
          FROM ref.articles_delta) r
    ORDER BY r.rank;

     rank | id |       score
    ------+----+-------------------
        1 | 10 | 0.999999940395355
        2 |  7 | 0.776150465011597

#### vvector_admin

The role that builds, loads and manages indexes: every procedure except
`sizing`, the functions in schema `vvector_admin`, and the tables
`vvector.manifest` and `vvector.snapshot`. It holds `vvector_search`.

- **When you need it:** for the users (or the ETL account) that register,
  refresh, schedule and remove indexes.
- **Limits:** such a user also needs USAGE and CREATE on the schema of the
  source table (the views are created there) and SELECT on the table.
  `schedule_refresh`, and `unregister_index` of an index with a schedule,
  also need a superuser (only a superuser may create or drop a trigger).
  `vbuild` is in `vvector_admin` because an incremental build reads a whole
  snapshot from the node cache: open to everyone, it would hand out the
  vectors of every index. Only `vvector_admin` may read the manifest.

**Syntax**

    GRANT vvector_admin TO user_name;
    ALTER USER user_name DEFAULT ROLE vvector_admin;    -- or, per session: SET ROLE vvector_admin;
    REVOKE vvector_admin FROM user_name;

**Example 1: an ETL account that manages its own index.** The account gets
the role, the rights on the schema and on the table:

    CREATE USER etl_user IDENTIFIED BY 'Etl_pw1';
    GRANT vvector_admin TO etl_user;
    ALTER USER etl_user DEFAULT ROLE vvector_admin;
    GRANT USAGE, CREATE ON SCHEMA ref TO etl_user;
    GRANT SELECT, INSERT ON ref.products TO etl_user;

Connected as `etl_user`:

    CALL vvector.register_index('products_etl', 'ref.products', 'id', 'vec', NULL, NULL, 'l2', NULL, 'flat');
    CALL vvector.refresh_index('products_etl');

    NOTICE 2005:  vvector: index products_etl registered (static: no version column). Next: CALL vvector.refresh_index('products_etl'). Queries read products_etl_snap in schema ref; grant SELECT on it to the users who may search the index.
    NOTICE 2005:  vvector: index products_etl refreshed: snapshot 29, full build (first build), 6 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.519 seconds; sent 0 MB of 0 MB (whole)

**Example 2: take the rights away.** After the role is revoked, the same
account can neither refresh nor read the manifest:

    REVOKE vvector_admin FROM etl_user;

As `etl_user`:

    CALL vvector.refresh_index('products_etl');
    SELECT COUNT(*) FROM vvector.manifest;

    ERROR 3457:  Function vvector.refresh_index(unknown) does not exist, or permission is denied for vvector.refresh_index(unknown)
    ERROR 4367:  Permission denied for relation manifest

`tests/sql/test_rights.sh` checks these rights with a user that has no
others.

### Set up and keep an index

#### sizing

Estimates the memory and disk of an index before you load the table: the
vectors, ids, graph and int8 codes, the size of the snapshot and cache file
on every node, the memory a build needs, and a comparison with the memory of
the smallest node.

- **When you need it:** before registering a large table, and to compare
  index types and quantization.
- **Limits:** an estimate (the HNSW graph varies with the data by a few
  percent). Anyone may call it.

**Syntax**

    CALL vvector.sizing(vectors, dimensions, index_type, quantization);

| Argument | Type | Meaning |
|---|---|---|
| vectors | INT | the number of vectors |
| dimensions | INT | the number of elements of each vector |
| index_type | VARCHAR | `hnsw` or `flat` |
| quantization | VARCHAR | `none` or `sq8` |

**Example 1: 10 million text embeddings of 768 numbers in an HNSW index.**

    CALL vvector.sizing(10000000, 768, 'hnsw', 'none');

    NOTICE 2005:  vvector.sizing: 10000000 vectors of 768 dimensions (768 floats per row): vectors 29296.9 MB, ids 76.3 MB, graph 1349.8 MB, sq8 codes 0.0 MB
    NOTICE 2005:  vvector.sizing: snapshot and cache file 30722.9 MB per node; build memory about 31228.4 MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)
    NOTICE 2005:  vvector.sizing: queries read the cache file through the page cache: keep it in memory. Smallest node here: 61.0 GB of memory

The index fits the 61 GB node; on a node with less than twice the index
size, `sizing` warns and recommends sq8 with `memory_mode='compact'`.

**Example 2: an exact index for 2 million vectors of 1536 numbers, with int8
codes** (a flat index reads every vector per search; the codes make that
read four times smaller):

    CALL vvector.sizing(2000000, 1536, 'flat', 'sq8');

    NOTICE 2005:  vvector.sizing: 2000000 vectors of 1536 dimensions (1536 floats per row): vectors 11718.8 MB, ids 15.3 MB, graph 0.0 MB, sq8 codes 2937.3 MB
    NOTICE 2005:  vvector.sizing: snapshot and cache file 14671.3 MB per node; build memory about 14679.0 MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)
    NOTICE 2005:  vvector.sizing: queries read the cache file through the page cache: keep it in memory. Smallest node here: 61.0 GB of memory

#### register_index

Registers a table as an index: it records the source, the metric and the
index type in `vvector.manifest`, and creates the two views the searches
read (see [The views _snap and _delta](#the-views-_snap-and-_delta)).
Nothing is built yet: [refresh_index](#refresh_index) builds.

- **When you need it:** once for every table you want to search.
- **Limits:** the metric is fixed for the life of the index (unregister and
  register again to change it). Every vector of one index has the same
  number of elements. The version column must be NOT NULL. Index names are
  1 to 64 letters, digits or underscores and are unique in the database.
  The views are created in the schema of the source table, so the caller
  needs CREATE there. Rights: `vvector_admin`.

**Syntax**

    CALL vvector.register_index(index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin [, index_type]);

| Argument | Type | Meaning |
|---|---|---|
| index_name | VARCHAR | the name of the index: 1 to 64 letters, digits or underscores |
| source_table | VARCHAR | `schema.table` |
| id_col | VARCHAR | the id column (INT) |
| vec_col | VARCHAR | the vector column: `ARRAY[FLOAT]`, `ARRAY[INT]` or `ARRAY[NUMERIC]` |
| op_col | VARCHAR | the delete flag (BOOLEAN, true = deleted; or INT, +1 / -1), or NULL when rows are only added; needs ver_col |
| ver_col | VARCHAR | the version (TIMESTAMPTZ, TIMESTAMP or INT, NOT NULL), or NULL for a static index: then every id appears once and every refresh is a full build |
| metric | VARCHAR | `l2` (the score of `VECTOR_L2`, smaller is closer), `cosine` (`COSINE_SIMILARITY`, larger is closer), `dot` (`DOT_PRODUCT`, larger is closer) or `l1` (Manhattan distance, smaller is closer) |
| margin | INT or NULL | how far the delta boundary stays behind the refresh: seconds for a timestamp version (NULL = 60; 0 is safe on one node with `CLOCK_TIMESTAMP()` versions, a cluster needs a few seconds for clock differences); units of the column for an INT version (required). A refresh builds the rows up to the boundary; newer rows stay in the delta until the next refresh (see [Freshness explained](#freshness-explained)) |
| index_type | VARCHAR | `hnsw` (default: a graph, fast and approximate) or `flat` (exact: reads every vector); see [Terms](#terms) |

**Example 1: articles that change all the time.** The journal table
`ref.articles` has a delete flag and a version column; the index uses cosine
similarity and the default HNSW graph:

    CALL vvector.register_index('articles', 'ref.articles', 'id', 'vec', 'del', 'ts', 'cosine', 0);

    NOTICE 2005:  vvector: index articles registered. Next: CALL vvector.refresh_index('articles'). Queries read articles_snap (snapshot only) or articles_delta (with the changes since the refresh) in schema ref; grant SELECT on them to the users who may search the index.
    NOTICE 2005:  vvector: index articles: journal replica: none: a single node reads the delta locally already

**Example 2: a static product catalogue with an exact index.** No delete
flag and no version: the catalogue is reloaded as a whole, and every refresh
builds the index again. The index is flat (exact) and uses the straight-line
distance:

    CALL vvector.register_index('products', 'ref.products', 'id', 'vec', NULL, NULL, 'l2', NULL, 'flat');

    NOTICE 2005:  vvector: index products registered (static: no version column). Next: CALL vvector.refresh_index('products'). Queries read products_snap in schema ref; grant SELECT on it to the users who may search the index.

#### The views _snap and _delta

`register_index` creates, in the schema of the source table:

- `<index>_snap`: one row (the sentinel) that carries the id of the active
  snapshot. A search over it reads the index only: the fastest statement.
- `<index>_delta` (only with a version column): the journal rows written
  after the boundary of the last refresh, plus the sentinel. A search over it
  with `freshness='exact'` also sees every committed change since the
  refresh.

Both have the columns `(qid, qvec, id, vec, del, ver, snapshot_id)`, the
input of [vsearch](#vsearch). `refresh_index` replaces them with new
boundaries and snapshot ids.

- **When you need them:** as the input of `vsearch`; the snapshot id lets a
  node that missed a refresh fail ("snapshot cache stale") instead of
  answering from an old snapshot.
- **Limits:** grant SELECT on them to the users who search with them; the
  `_delta` view shows journal rows of the source table. `vknn`, `vscan` and
  `vsearch ... FROM dual` need no view.

**Syntax**

    SELECT ... FROM <schema>.<index>_snap;
    SELECT ... FROM <schema>.<index>_delta;

**Example 1: the fastest search.** The `_snap` view is one row; `vsearch`
over it with the `query` parameter searches the index and reads no table
(the example relies on the first refresh, in [refresh_index](#refresh_index)):

    SELECT * FROM ref.articles_snap;
    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='articles', query='[0.0, 0.1, 0.9, 0.0]', k=2) OVER()
    FROM ref.articles_snap;

     qid | qvec | id | vec | del | ver | snapshot_id
    -----+------+----+-----+-----+-----+-------------
         |      |    |     |     |     |          26

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  5 |                 1 |    1
       0 |  8 | 0.826480686664581 |    2

**Example 2: see what the index does not have yet.** A new article is
written; the `_delta` view shows it until the next refresh (the row without
an id is the sentinel):

    INSERT INTO ref.articles (id, vec) VALUES (10, ARRAY[0.0, 0.6, 0.0, 0.6]);
    COMMIT;
    SELECT id, vec, del, ver IS NOT NULL AS has_ver, snapshot_id FROM ref.articles_delta ORDER BY id;

     id |        vec        | del | has_ver | snapshot_id
    ----+-------------------+-----+---------+-------------
        |                   |     | f       |          26
     10 | [0.0,0.6,0.0,0.6] | f   | t       |          26

#### refresh_index

Builds the index: takes the delta boundary, builds a new snapshot, stores it
in `vvector.snapshot`, loads it on every node, writes the index defaults to
every node, updates the manifest and the views, and deletes the stored
snapshots that are no longer needed. Queries keep working during a refresh.

- **When you need it:** after `register_index` (the first build), and then
  whenever the index should take in the new rows: by hand, from your load
  job, or on a schedule ([schedule_refresh](#schedule_refresh)).
- **Limits:** one refresh of an index runs at a time; a second one stops at
  once with an error that names the running one. In Eon a refresh loads the
  nodes of its own subcluster; other subclusters run
  [load_all](#load_all). Rights: `vvector_admin`.

**Syntax**

    CALL vvector.refresh_index(index_name [, mode]);

| mode | Builds |
|---|---|
| (the index's `refresh_mode`, default `auto`) | `auto`: incremental; full when the tombstones exceed `tombstone_ratio` (default 0.2) of the snapshot, or after `rebuild_every` incremental refreshes (default: never by count) |
| `incremental` | incremental; the ratio and the count are ignored (`status` warns when the tombstones pass the ratio) |
| `full` | full, always |

- **incremental**: starts from the active snapshot in the cache of the node
  that runs the refresh and reads only the journal rows written after the
  previous boundary (the latest row of each id). A new or changed vector is
  added (and inserted into the graph of an HNSW index); the old vector of a
  changed or deleted id stays as a **tombstone** that searches skip. When
  nothing changed, the snapshot is kept and only the boundary moves.
- **full**: reads every row (the latest row of each id, deletes left out)
  and builds a new snapshot without tombstones.

In every mode the build is full when an incremental one cannot give the
right answer: the first build, a static index, changed build options
(`index_type`, `m`, `ef_construction`, `quantization`), a node whose cache
lacks the active snapshot, and journal rows up to the previous boundary that
changed since the last refresh (a physical DELETE or UPDATE, a dropped
partition). The manifest keeps the count and a digest of those rows; the
index option `verify_every` says how often a refresh checks them (1 = every
refresh, the default; N = every N refreshes; 0 = never: then run
`refresh_index(name, 'full')` after every physical change yourself). The
first line of the output says what was built and why.

**Example 1: the first build.** Both example indexes are built for the first
time:

    CALL vvector.refresh_index('articles');
    CALL vvector.refresh_index('products');

    NOTICE 2005:  vvector: index articles refreshed: snapshot 23, full build (first build), 8 vectors of 4 dimensions, 0 tombstones, 0 MB, 0.749 seconds; sent 0 MB of 0 MB (whole); journal digest taken in 0.017 seconds
    NOTICE 2005:  vvector: index articles: journal replica: none: a single node reads the delta locally already
    NOTICE 2005:  vvector: index products refreshed: snapshot 24, full build (first build), 6 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.491 seconds; sent 0 MB of 0 MB (whole)

**Example 2: take in changes, then remove the tombstones.** A new article, a
changed one and a deleted one; the refresh adds only these to the snapshot
and sends only the changed bytes to the nodes. A full refresh then builds a
snapshot without the two tombstones (the old vector of article 4 and
article 6):

    INSERT INTO ref.articles (id, vec) VALUES (9, ARRAY[0.7, 0.0, 0.3, 0.0]);   -- a new article
    INSERT INTO ref.articles (id, vec) VALUES (4, ARRAY[0.1, 0.9, 0.0, 0.0]);   -- article 4 changed
    INSERT INTO ref.articles (id, del) VALUES (6, TRUE);                        -- article 6 deleted
    COMMIT;
    CALL vvector.refresh_index('articles');
    CALL vvector.refresh_index('articles', 'full');

    NOTICE 2005:  vvector: index articles refreshed: snapshot 25, incremental from snapshot 23 (2 vectors appended, 2 tombstoned), 8 vectors of 4 dimensions, 2 tombstones, 0 MB, 0.584 seconds; sent 0 MB of 0 MB (a patch on snapshot 23 in every node cache; the table holds a chain of 2 snapshots, 0 MB of patches since the whole copy 23); journal verified in 0.026 seconds
    NOTICE 2005:  vvector: index articles: journal replica: none: a single node reads the delta locally already
    NOTICE 2005:  vvector: index articles refreshed: snapshot 26, full build (mode full), 8 vectors of 4 dimensions, 0 tombstones, 0 MB, 0.695 seconds; sent 0 MB of 0 MB (whole); journal digest taken in 0.017 seconds
    NOTICE 2005:  vvector: index articles: journal replica: none: a single node reads the delta locally already

The cost of an incremental refresh grows with the changes, not with the
index (900,000 vectors of 128 dimensions, the test VM, fenced):

| Change since the last refresh | flat | hnsw |
|---|---:|---:|
| none (the snapshot is kept), journal verified (`verify_every` 1) | 1.0 s | 1.0 s |
| none, not verified (`verify_every` 0) | 0.6 s | 0.6 s |
| 1,000 adds, 500 deletes | 5.0 s | 6.9 s |
| 50,000 adds, 25,000 deletes | 5.7 s | 9.5 s |
| full build of the same 930,550 vectors | 8.8 s | 42.1 s |

The refresh marks the manifest row while it runs (`refresh_started_at`,
`refresh_started_by`) and removes the mark when it ends, also when it fails.
When its session was killed or its node went down, the next refresh ignores
the mark as soon as the session is gone from `v_monitor.sessions` (seen by a
superuser, or by the same user), else after 6 hours (or 4 times the last
build time, when longer); the error message gives the statement that removes
it by hand. How the snapshot is sent to the nodes is described in
[Operations](#operations).

#### set_index_options

Changes the options of an index: how it is built (applies at the next
refresh) and the defaults of its searches (apply at once on every node,
within 200 ms).

- **When you need it:** to tune an index after registration: its type,
  graph, int8 codes, refresh behaviour, query defaults, cache directory.
- **Limits:** NULL keeps an option as it is; `'default'` (text) or `0`
  (numbers) sets a query default back to the built-in one. A change of a
  build option means a full build at the next refresh. The shorter forms
  (13, 14 and 15 arguments) leave the options they do not name as they are.
  Rights: `vvector_admin`.

**Syntax**

    CALL vvector.set_index_options(index_name, index_type, m, ef_construction, quantization, refresh_mode,
                                   tombstone_ratio, rebuild_every, memory_mode, precision_default,
                                   freshness_default, ef_search_default, threads_default
                                   [, verify_every [, cache_dir [, reachability]]]);

| Option | Values | Default | Effect |
|---|---|---|---|
| index_type | flat, hnsw | as registered | the index type (next refresh) |
| m, ef_construction | 2 to 256, 1 to 100000 | 16, 200 | HNSW: links per vector and the candidate list of the build; higher = better recall, more memory, slower build (next refresh) |
| quantization | none, sq8 | none | sq8 adds one byte per element; searches rank by those bytes and rescore the best candidates (see [int8 quantisation](#int8-quantisation-sq8)); next refresh |
| refresh_mode | auto, incremental, full | auto | see [refresh_index](#refresh_index) |
| tombstone_ratio | above 0 to 1 | 0.2 | `auto` builds in full once the tombstones exceed this share |
| rebuild_every | 0 (never) or more | never | `auto` builds in full after this many incremental refreshes |
| memory_mode | ram, compact | ram | `compact` (needs sq8) reads ahead only the bytes, ids and graph, not the floats; with the next snapshot a node maps |
| precision_default | fast, balanced, best, exact | balanced | the query default (HNSW; a flat index is always exact) |
| freshness_default | snapshot, exact | snapshot | the query default |
| ef_search_default | 0 to 100000 | 0 (the preset of precision) | the query default (HNSW) |
| threads_default | 0 (one per core) to 64 | 0 | the query default |
| verify_every | 0 (never), 1 (every refresh) or more | 1 | how often a refresh verifies the journal rows up to the boundary |
| cache_dir | an absolute path, or `default` | `/tmp/vvector` | where every node keeps the cache files of this index; the active snapshot is loaded there at once (the option changes only when every node has it); the files in the old directory stay |
| reachability | auto, on, off | auto | HNSW: whether a build counts the vectors no search can reach (see [Unreachable vectors](#unreachable-vectors)) |

**Example 1: every search of the index sees the newest rows.** With
`freshness_default` `exact`, a search over the `_delta` view applies the
journal without a `freshness` parameter, so the article written after the
last refresh is found:

    CALL vvector.set_index_options('articles', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'exact', NULL, NULL);
    SELECT r.rank, r.id, i.title
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='articles', query='[0.0, 0.6, 0.0, 0.6]', k=2) OVER()
          FROM ref.articles_delta) r JOIN ref.article_info i USING (id)
    ORDER BY r.rank;

    NOTICE 2005:  vvector: index articles options changed. Build options apply at the next refresh; query defaults apply now.
     rank | id |        title
    ------+----+---------------------
        1 | 10 | Cooking for runners
        2 |  7 | Marathon training

**Example 2: int8 codes and the best precision.** `quantization` is a build
option, so the next refresh builds in full; `precision_default` applies to
every search at once:

    CALL vvector.set_index_options('articles', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, 'best', NULL, NULL, NULL);
    CALL vvector.refresh_index('articles');

    NOTICE 2005:  vvector: index articles options changed. Build options apply at the next refresh; query defaults apply now.
    NOTICE 2005:  vvector: index articles refreshed: snapshot 27, full build (build options changed from hnsw cosine none m=16 ef_construction=200 to hnsw cosine sq8 m=16 ef_construction=200), 9 vectors of 4 dimensions, 0 tombstones, 0 MB, 0.759 seconds; sent 0 MB of 0 MB (whole); journal digest taken in 0.018 seconds
    NOTICE 2005:  vvector: index articles: journal replica: none: a single node reads the delta locally already

More examples:

    -- keep the node caches of the index on a data disk (loaded there at once):
    CALL vvector.set_index_options('articles', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '/data/vvector');
    -- verify the journal every 10 refreshes instead of at every one:
    CALL vvector.set_index_options('articles', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 10);

#### set_journal_replica

Makes, keeps or drops a replicated projection of the journal table
(`<schema>.<index>_journal_rep`, UNSEGMENTED ALL NODES, sorted by the
version), so the `_delta` view is read on the node that runs the search
instead of on every node.

- **When you need it:** on a cluster, for searches with
  `freshness='exact'`: without the replica a statement over the `_delta`
  view costs 15 to 17 ms more on the 3-node test cluster. `register_index`
  and every refresh apply the mode, so you call this only to change it.
- **Limits:** the replica is a full copy of the journal on every node and
  makes bulk loads into the journal about 2.5 times slower. `auto` makes it
  on more than one node when one copy of the journal is at most 2048 MB and
  10% of the smallest free disk. Failures (no right to create a projection,
  a load that holds a lock) are reported in `replica_note`, never an error.
  Rights: `vvector_admin`, and the right to create a projection on the table.

**Syntax**

    CALL vvector.set_journal_replica(index_name, mode);   -- mode: auto | on | off

**Example 1: on one node.** A replica gains nothing on one node; `on`
creates it anyway and says so:

    CALL vvector.set_journal_replica('articles', 'on');

    NOTICE 2005:  vvector: index articles: journal replica on: created ref.articles_journal_rep (journal 0 MB, one copy on each of 1 nodes); a single node gains nothing from it

**Example 2 (Eon): on a 5-node cluster.** `register_index` made the replica
by itself (`auto`); `off` drops it, `auto` makes it again:

    CALL vvector.register_index('articles', 'ref.articles', 'id', 'vec', 'del', 'ts', 'cosine', 5);

    NOTICE 2005:  vvector: index articles registered. Next: CALL vvector.refresh_index('articles'). Queries read articles_snap (snapshot only) or articles_delta (with the changes since the refresh) in schema ref; grant SELECT on them to the users who may search the index.
    NOTICE 2005:  vvector: index articles: journal replica: created ref.articles_journal_rep (journal 0 MB, one copy on each of 5 nodes)

    CALL vvector.set_journal_replica('articles', 'off');

    NOTICE 2005:  vvector: index articles: journal replica off: none: journal_replica is off; dropped ref.articles_journal_rep

    CALL vvector.set_journal_replica('articles', 'auto');

    NOTICE 2005:  vvector: index articles: journal replica auto: created ref.articles_journal_rep (journal 0 MB, one copy on each of 5 nodes)

    SELECT projection_name, is_segmented FROM projections
    WHERE projection_schema = 'ref' AND anchor_table_name = 'articles' ORDER BY 1;

       projection_name    | is_segmented
    ----------------------+--------------
     articles_journal_rep | f
     articles_journal_rep | f
     articles_journal_rep | f
     articles_super       | t

(Eon lists an unsegmented projection once per shard subscription.)

#### schedule_refresh

Refreshes an index on a schedule: creates `vvector.<index>_refresh_schedule`
(Vertica's CRON schedule) and `vvector.<index>_refresh_trigger`, which calls
`refresh_index` as the definer. Calling it again replaces the schedule.

- **When you need it:** to keep an index current without an outside job.
- **Limits:** needs a superuser (Vertica lets only a superuser create a
  trigger). Vertica starts due schedules every 30 seconds. A scheduled
  refresh runs in a session that `v_monitor.sessions` does not show: read
  the manifest to see what it did. In Eon it loads the primary subcluster.

**Syntax**

    CALL vvector.schedule_refresh(index_name, cron_expression);

`cron_expression`: minute, hour, day of month, month, day of week, as in cron
(`'*/15 * * * *'` every 15 minutes, `'0 2 * * *'` every night at 02:00).

**Example 1: refresh every minute.** An article is written after the
schedule is set; a minute and a half later the manifest shows that a
scheduled refresh took it in (10 vectors):

    CALL vvector.schedule_refresh('articles', '* * * * *');
    INSERT INTO ref.articles (id, vec) VALUES (11, ARRAY[0.5, 0.5, 0.0, 0.0]);
    COMMIT;

(95 seconds later)

    SELECT vector_count, refresh_note FROM vvector.manifest WHERE index_name = 'articles';

     vector_count |                                                                                                                                                               refresh_note
    --------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
               10 | refreshed: snapshot 30, incremental from snapshot 28 (1 vectors appended, 0 tombstoned), 10 vectors of 4 dimensions, 0 tombstones, 0 MB, 0.61 seconds; sent 0 MB of 0 MB (a patch on snapshot 28 in every node cache; the table holds a chain of 2 snapshots, 0 MB of patches since the whole copy 28); journal verified in 0.027 seconds

**Example 2: pause the schedule during a bulk load, then refresh at night.**
Drop the trigger and the schedule; set a new schedule when the load is done:

    DROP TRIGGER vvector.articles_refresh_trigger;
    DROP SCHEDULE vvector.articles_refresh_schedule;

    CALL vvector.schedule_refresh('articles', '0 2 * * *');
    SELECT schedule_name, attached_trigger, date_time_string FROM v_catalog.user_schedules WHERE schedule_name ILIKE 'articles%';

    NOTICE 2005:  vvector: index articles is refreshed on schedule 0 2 * * *
           schedule_name       |     attached_trigger     | date_time_string
    ---------------------------+--------------------------+------------------
     articles_refresh_schedule | articles_refresh_trigger | 0 2 * * *

#### status

Reports on an index: live vectors and tombstones, refresh mode, what the
last refresh did, a refresh that runs now, how many nodes hold the active
snapshot, the rows in the delta and how long they take to read, the journal
replica, open transactions that write to the table, versions in the future,
and a sizing check (index size against node memory, the memory a refresh
needs against `FencedUDxMemoryLimitMB`, `threads_default` against cores).
Each problem is a WARNING with the recommended fix.

- **When you need it:** to check an index, and before and after changing
  its options. It never changes anything.
- **Limits:** it reads the delta view and the catalog, so it takes as long
  as reading the delta. Rights: `vvector_admin`.

**Syntax**

    CALL vvector.status(index_name);

**Example 1: a journal index with a row in the delta.**

    CALL vvector.status('articles');

    NOTICE 2005:  vvector: index articles: hnsw index, 8 live vectors, 0 tombstones (tombstone_ratio 0.2), refresh_mode auto, 0 incremental refreshes since the last full build (rebuild_every never)
    NOTICE 2005:  vvector: index articles: vectors no search can reach: 0 (reachability auto)
    NOTICE 2005:  vvector: index articles: journal digest verified at every refresh (verify_every 1); 0 refreshes since the last verification or full build
    NOTICE 2005:  vvector: index articles: node cache directory: the default (/tmp/vvector, or the cache_dir session parameter)
    NOTICE 2005:  vvector: index articles: active snapshot 26 in the cache of 1 of 1 nodes
    NOTICE 2005:  vvector: index articles: last refresh: refreshed: snapshot 26, full build (mode full), 8 vectors of 4 dimensions, 0 tombstones, 0 MB, 0.695 seconds; sent 0 MB of 0 MB (whole); journal digest taken in 0.017 seconds
    NOTICE 2005:  vvector: index articles: snapshot pieces in vvector.snapshot: a chain of 1 snapshots from the whole copy 26, 0 MB of patches after it; the layout has room for 4096 more vectors before a refresh sends the whole snapshot again
    NOTICE 2005:  vvector: index articles: 1 journal rows in the delta, read in 9 ms
    NOTICE 2005:  vvector: index articles: journal replica auto: none: a single node reads the delta locally already
    NOTICE 2005:  vvector: index articles: sizing: index 0 MB, all indexes 1 MB, build about 0 MB, smallest node 62464 MB of memory (59240 MB free or cache), 22 cores

**Example 2: a static index.** No delta, no journal checks:

    CALL vvector.status('products');

    NOTICE 2005:  vvector: index products: flat index, 6 live vectors, 0 tombstones (tombstone_ratio 0.2), refresh_mode auto, 0 incremental refreshes since the last full build (rebuild_every never)
    NOTICE 2005:  vvector: index products: node cache directory: the default (/tmp/vvector, or the cache_dir session parameter)
    NOTICE 2005:  vvector: index products: active snapshot 24 in the cache of 1 of 1 nodes
    NOTICE 2005:  vvector: index products: last refresh: refreshed: snapshot 24, full build (first build), 6 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.491 seconds; sent 0 MB of 0 MB (whole)
    NOTICE 2005:  vvector: index products: snapshot pieces in vvector.snapshot: a chain of 1 snapshots from the whole copy 24, 0 MB of patches after it; the layout has room for 4096 more vectors before a refresh sends the whole snapshot again
    NOTICE 2005:  vvector: index products: sizing: index 0 MB, all indexes 1 MB, build about 0 MB, smallest node 62464 MB of memory (59202 MB free or cache), 22 cores

#### load_all

Loads the active snapshot and the index defaults on every node that lacks
them. It first asks every node, so when all have the snapshot it only
rewrites the defaults (milliseconds): it can run as often as wanted.

- **When you need it:** a node that was down during a refresh, a node whose
  cache directory was cleaned (`/tmp` at a reboot), a new node, and in Eon
  every subcluster that did not run the refresh (see
  [Operations](#operations)).
- **Limits:** loads the nodes of the session's subcluster only (in Eon). An
  error of one index stops `load_all()` and names it. Rights:
  `vvector_admin`.

**Syntax**

    CALL vvector.load_all(index_name);   -- one index
    CALL vvector.load_all();             -- every registered index with a snapshot, in name order

**Example 1: a node lost its cache.** The cache directory of the index is
removed (as a reboot that cleans `/tmp` would do); a search fails until
`load_all` restores it:

    rm -rf /tmp/vvector/articles

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='articles', query='[0.9, 0.1, 0.0, 0.0]', k=1) OVER()
    FROM ref.articles_snap;

    ERROR 3399:  Failure in UDx RPC call InvokeProcessPartition(): Error calling processPartition() in User Defined Object [vsearch] at [src/udx/vsearch.cpp:234], error code: 0, message: vsearch: no snapshot cache for index 'articles' in /tmp/vvector: run vload

    CALL vvector.load_all('articles');
    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='articles', query='[0.9, 0.1, 0.0, 0.0]', k=1) OVER()
    FROM ref.articles_snap;

    NOTICE 2005:  vvector: index articles: snapshot 28 loaded on all nodes (1 of the chain 28)
     qid | id | score | rank
    -----+----+-------+------
       0 |  1 |     1 |    1

**Example 2 (Eon): a second subcluster.** A refresh run on the primary
subcluster loads its own nodes and says so:

    CALL vvector.refresh_index('articles');

    NOTICE 2005:  vvector: index articles refreshed: snapshot 4445, full build (first build), 8 vectors of 4 dimensions, 0 tombstones, 0 MB, 2.404 seconds; sent 0 MB of 0 MB (whole); journal digest taken in 0.034 seconds
    NOTICE 2005:  vvector: index articles: journal replica: kept ref.articles_journal_rep
    NOTICE 2005:  vvector: index articles: loaded on the nodes of subcluster default_subcluster only; every other subcluster loads it with CALL vvector.load_all('articles') from a session there (until then a search there gets "snapshot cache stale")

In a session on a node of the secondary subcluster `sc_secondary_1`, the
search fails until `load_all()` loads every index there (the test cluster
had three more indexes):

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='articles', query='[0.9, 0.1, 0.0, 0.0]', k=2) OVER()
    FROM ref.articles_snap;

    ERROR 3399:  Failure in UDx RPC call InvokeProcessPartition(): ... message: vsearch: no snapshot cache for index 'articles' in /tmp/vvector: run vload

    CALL vvector.load_all();

    NOTICE 2005:  vvector: index articles: snapshot 4445 loaded on all nodes of subcluster sc_secondary_1 (1 of the chain 4445)
    NOTICE 2005:  vvector: index gen: snapshot 4356 loaded on all nodes of subcluster sc_secondary_1 (1 of the chain 4356)
    NOTICE 2005:  vvector: index gen_hnsw: snapshot 4357 loaded on all nodes of subcluster sc_secondary_1 (1 of the chain 4357)
    NOTICE 2005:  vvector: index gen_sq8: snapshot 4358 loaded on all nodes of subcluster sc_secondary_1 (1 of the chain 4358)
    NOTICE 2005:  vvector.load_all: 4 indexes: 4 loaded, 0 already in the cache of all 2 nodes of subcluster sc_secondary_1, 0 without a snapshot

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  1 |                 1 |    1
       0 |  2 | 0.978709042072296 |    2

Run `CALL vvector.load_all();` from a scheduled job on every secondary
subcluster, after the refresh times of the primary one.

#### unregister_index

Removes an index: its schedule, its views, its journal replica, its stored
snapshots and its manifest row. The source table is not touched.

- **When you need it:** when an index is no longer used, or to register the
  table again with another metric.
- **Limits:** the cache files stay on the nodes: remove
  `<cache_dir>/<index_name>` on every node by hand. An index with a schedule
  can be removed by a superuser only (it drops the trigger); for anyone else
  it stops before it removes anything. Rights: `vvector_admin`.

**Syntax**

    CALL vvector.unregister_index(index_name);

**Example 1: remove the product index and its cache files.**

    CALL vvector.unregister_index('products');

    NOTICE 2005:  vvector: index products unregistered. Cache files under <cache_dir>/products stay on the nodes.

On every node:

    rm -rf /tmp/vvector/products

**Example 2: an index with a schedule.** The ETL account (`vvector_admin`,
no superuser) cannot remove it; a superuser can:

    CALL vvector.unregister_index('articles');

    ERROR 2005:  vvector.unregister_index: index articles has a refresh schedule; only a superuser can remove it (Vertica allows only superusers to drop triggers): a superuser runs CALL vvector.unregister_index('articles')

As a superuser:

    CALL vvector.unregister_index('articles');

    NOTICE 2005:  vvector: index articles unregistered. Cache files under <cache_dir>/articles stay on the nodes.

### Search functions

All three need the role `vvector_search`. The score of every result is what
the built-in function of the metric returns: `VECTOR_L2` distance (smaller
is closer), `COSINE_SIMILARITY` and `DOT_PRODUCT` (larger is closer), the
Manhattan distance for l1 (smaller is closer). Rank 1 is the closest; equal
scores are ranked by id. Results never depend on the number of threads, the
batch size, the CPU or the order of the input rows. Whether a search is
exact or approximate is explained in [Terms](#terms).

#### vsearch

Returns the k nearest neighbours of every query row, from the index, and,
with `freshness='exact'`, also from the journal rows written since the last
refresh.

- **When you need it:** for most searches: one query or thousands in one
  statement, with or without the newest rows, filtered by an allow-list,
  limited by a radius. Its input is a view of the index, so a node with an
  old cache fails instead of answering from it.
- **Limits:** a transform function: it reads all rows of its input, then
  returns; write it with `OVER()`. At most k = 16384. `precision` and
  `ef_search` matter on an HNSW index only; a flat index without sq8 is
  always exact.

**Syntax**

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='name' [, parameter=value ...]) OVER()
    FROM <schema>.<index>_snap | <schema>.<index>_delta | (<a view> UNION ALL <query rows> [UNION ALL <allow-list rows>]);

Output: `(qid, id, score, rank)`. The input has seven columns; the role of a
row is given by which of them are NULL:

| qid | qvec | id | vec | del | Role |
|---|---|---|---|---|---|
| set | set | NULL | NULL | NULL | a query |
| NULL | NULL | set | set | false | a journal row: add or change (from the `_delta` view) |
| NULL | NULL | set | NULL | true | a journal row: delete (from the `_delta` view) |
| NULL | NULL | set | NULL | NULL | an allow-list member: only allowed ids are returned (see [Filtered search](#filtered-search)) |
| NULL | NULL | NULL | NULL | NULL | the sentinel row of a view: carries `snapshot_id` only |

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| index_name | (required) | | the index |
| k | 10 | 1 to 16384 | neighbours per query |
| query | | `'[x1, x2, ...]'` | one query vector as text, with qid 0; beside query rows or alone. Faster than an `ARRAY[...]` literal in the statement (Vertica parses a literal of 128 numbers in about 7 ms) |
| freshness | snapshot | snapshot, exact | `exact` applies the journal rows of the input; `snapshot` ignores them. It decides which rows are searched, not whether the search is exact |
| precision | balanced | fast, balanced, best, exact | the speed and recall trade-off (HNSW; fast: ef_search 2 x k, at least 32; balanced: 100; best: 400), and with sq8 the rescoring (fast: none; balanced: 2 x k candidates, 4 x k from 512 dimensions on; best: 4 x k). `exact` reads every vector |
| ef_search | 0 (preset) | 0 to 100000 | HNSW: the length of the candidate list; overrides the preset of `precision`; below k it is raised to k |
| exact | false | true, false | `true` = `precision='exact'` |
| radius | off | a number | only neighbours within it, at most k: l2 and l1 `score <= radius`, cosine and dot `score >= radius` (see [Range search](#range-search)) |
| filtered | false | true, false | `true` returns only allow-listed ids even when the input has no allow-list row (a filter that matched nothing returns nothing) |
| rescore, oversampling | preset of precision | true, false; 1 to 100 | sq8 only: rescore the best k x oversampling candidates with the float vectors, or return the k best with approximate scores |
| threads | 0 | 0 (one per core) to 64 | threads for one statement |
| cache_dir | the index option, else `/tmp/vvector` | an absolute path | where the node cache is |

Every parameter except `index_name`, `query`, `radius` and `filtered` can
also be a [session default](#session-defaults), and `precision`,
`freshness`, `ef_search` and `threads` an index default
([set_index_options](#set_index_options)). The first that is set wins:
function parameter, session parameter, index default, built-in default.

**Example 1: answer one question.** The three articles closest to a question
about query speed, with their titles; the `_snap` view makes this the
fastest statement:

    SELECT r.rank, r.id, i.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='articles', query='[0.85, 0.0, 0.1, 0.05]', k=3) OVER()
          FROM ref.articles_snap) r
    JOIN ref.article_info i USING (id)
    ORDER BY r.rank;

     rank | id |             title             |       score
    ------+----+-------------------------------+-------------------
        1 |  2 | Tuning a column store         | 0.997859001159668
        2 |  1 | Vertica projections explained | 0.985396087169647
        3 |  9 | Databases on the road         | 0.957243323326111

**Example 2: many questions at once, with the newest articles.** All
questions of a table in one statement (one statement is much faster than
one per question); `freshness='exact'` over the `_delta` view also finds
article 10, written after the last refresh ("Cooking for runners"):

    SELECT q.question, r.rank, i.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='articles', k=3, freshness='exact') OVER()
          FROM (SELECT * FROM ref.articles_delta
                UNION ALL
                SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM ref.questions) input) r
    JOIN ref.questions q USING (qid)
    JOIN ref.article_info i USING (id)
    ORDER BY r.qid, r.rank;

              question          | rank |             title             |       score
    ----------------------------+------+-------------------------------+-------------------
     How do I speed up queries? |    1 | Tuning a column store         | 0.997859001159668
     How do I speed up queries? |    2 | Vertica projections explained | 0.985396087169647
     How do I speed up queries? |    3 | Databases on the road         | 0.957243323326111
     What can I cook tonight?   |    1 | Bread baking at home          | 0.996946513652802
     What can I cook tonight?   |    2 | Pasta in ten minutes          | 0.996946513652802
     What can I cook tonight?   |    3 | Cooking for runners           | 0.704934418201447
     Where can I travel to run? |    1 | Marathon training             | 0.776150465011597
     Where can I travel to run? |    2 | A week in Lisbon              | 0.702781915664673
     Where can I travel to run? |    3 | Street food in Bangkok        | 0.536875486373901

More patterns (allow-lists, range search, joins, recipes) are in
[Search](#search).

#### vknn

Searches one vector per input row and returns its k neighbours as rows
`(id, score, rank)`, beside the other columns of the row.

- **When you need it:** when the query vectors are a column of a table and
  you want the neighbours next to each row, without a view and without
  `OVER()`.
- **Limits:** searches the snapshot only: it does not apply the journal, and
  it does not check that the node's cache is the active snapshot (right
  after a refresh a node may answer from the previous one for up to 200 ms;
  a node that missed the refresh answers from its old snapshot until
  `load_all`). Searches row by row: for many queries in one statement,
  `vsearch` is faster. Like vsearch it is approximate on an HNSW index.

**Syntax**

    SELECT [other columns,] vvector.vknn(vector_expression USING PARAMETERS index_name='name' [, parameter=value ...])
    FROM table;

    SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='name', query='[x1, x2, ...]' [, ...]) FROM dual;

Parameters: those of `vsearch` except `freshness` and `filtered`. A row
whose vector is NULL uses the `query` parameter; without it, it gives no
rows.

**Example 1: related articles.** For every article, the closest other
article (k = 2 returns the article itself and its nearest neighbour; the
outer query drops the article itself):

    SELECT n.article, a.title, n.id AS related, r.title AS related_title, n.score
    FROM (SELECT l.id AS article, vvector.vknn(l.vec USING PARAMETERS index_name='articles', k=2) FROM ref.articles_live l) n
    JOIN ref.article_info a ON a.id = n.article
    JOIN ref.article_info r ON r.id = n.id
    WHERE n.id <> n.article
    ORDER BY n.article;

     article |             title             | related |         related_title         |       score
    ---------+-------------------------------+---------+-------------------------------+-------------------
           1 | Vertica projections explained |       2 | Tuning a column store         | 0.978709042072296
           2 | Tuning a column store         |       1 | Vertica projections explained | 0.978709042072296
           3 | Bread baking at home          |       4 | Pasta in ten minutes          | 0.987804889678955
           4 | Pasta in ten minutes          |       3 | Bread baking at home          | 0.987804889678955
           5 | A week in Lisbon              |       8 | Street food in Bangkok        | 0.826480686664581
           7 | Marathon training             |      10 | Cooking for runners           | 0.776150465011597
           8 | Street food in Bangkok        |       5 | A week in Lisbon              | 0.826480686664581
           9 | Databases on the road         |       2 | Tuning a column store         | 0.953599572181702
          10 | Cooking for runners           |       7 | Marathon training             | 0.776150465011597

**Example 2: one ad-hoc query on the product index.** A customer looks for
something like trail running gear; the flat index returns the exact nearest
products by straight-line distance:

    SELECT n.rank, i.name, n.score
    FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='products', query='[0.45, 0.85, 0.7]', k=3)
          FROM dual) n
    JOIN ref.products i USING (id)
    ORDER BY n.rank;

     rank |     name      |       score
    ------+---------------+-------------------
        1 | running shoes | 0.122474439442158
        2 | trail shoes   | 0.212132036685944
        3 | rain jacket   | 0.604152321815491

#### vscan

An exact search over any table or query result, with no index: every row is
compared, in parallel on every node.

- **When you need it:** for tables that have no index (too large, or rarely
  searched), for a filter that is any SQL predicate, and as the exact
  reference when you measure the recall of an index.
- **Limits:** reads every row of its input in every statement (1M vectors of
  128 dimensions: 0.5 to 0.6 s on the 4-node test cluster; 100M: 19 s). Each
  instance returns its own k best per query; the SQL around it merges them.
  Rows with a NULL id or vector are skipped; a vector of another length is
  an error. A journal table must be consolidated first (the latest row per
  id, deletes left out; the view `ref.articles_live` of the example data).

**Syntax**

    SELECT ... FROM (SELECT vvector.vscan(id, vec USING PARAMETERS query='[x1, ...]' | queries='[..];[..]' [, parameter=value ...])
                            OVER(PARTITION BEST)
                     FROM table [WHERE ...]) s
    ORDER BY score [DESC] LIMIT k;          -- or ROW_NUMBER() OVER(PARTITION BY qid ...) for several queries

Output: `(qid, id, score)`.

| Parameter | Default | Meaning |
|---|---|---|
| query | | one vector as text; qid 0 |
| queries | | several vectors separated by `;` (a LONG VARCHAR of up to 32 MB); qid 1, 2, ... |
| k | 10 | results per query and instance, 1 to 16384 |
| metric | l2 | l2, cosine, dot, l1 |
| radius | off | only rows within it: l2 and l1 score <= radius, cosine and dot score >= radius |
| threads | 1 | Vertica supplies the parallelism; more only for `OVER()` on one instance |

**Example 1: exact search with any filter.** The three English articles
closest to a travel question, without an index; the filter is a join:

    SELECT s.id, i.title, s.score
    FROM (SELECT vvector.vscan(l.id, l.vec USING PARAMETERS query='[0.0, 0.2, 0.8, 0.0]', k=3, metric='cosine')
                 OVER(PARTITION BEST)
          FROM ref.articles_live l JOIN ref.article_info i ON i.id = l.id
          WHERE i.lang = 'en') s
    JOIN ref.article_info i USING (id)
    ORDER BY s.score DESC LIMIT 3;

     id |         title          |       score
    ----+------------------------+-------------------
      5 | A week in Lisbon       | 0.990992426872253
      8 | Street food in Bangkok | 0.894427180290222
      9 | Databases on the road  | 0.382157862186432

**Example 2: several queries in one statement.** Two queries on the product
table; `ROW_NUMBER()` keeps the two best per query:

    SELECT r.qid, r.rank, p.name, r.score
    FROM (SELECT qid, id, score, ROW_NUMBER() OVER(PARTITION BY qid ORDER BY score, id) AS rank
          FROM (SELECT vvector.vscan(id, vec USING PARAMETERS queries='[0.4, 0.9, 0.6];[0.7, 0.1, 0.0]', k=2)
                       OVER(PARTITION BEST)
                FROM ref.products) s) r
    JOIN ref.products p USING (id)
    WHERE r.rank <= 2
    ORDER BY r.qid, r.rank;

     qid | rank |     name      |       score
    -----+------+---------------+-------------------
       1 |    1 | running shoes |                 0
       1 |    2 | trail shoes   |  0.33166241645813
       2 |    1 | office chair  | 0.100000001490116
       2 |    2 | yoga mat      | 0.787400782108307

#### Session defaults

Sets the search parameters for the rest of the session, so the statements
need not repeat them.

- **When you need it:** for a batch job or a notebook that runs many
  searches with the same settings.
- **Limits:** the values are text (`'20'`, not `20`). A function parameter
  still wins over the session value. `index_name`, `query`, `radius` and
  `filtered` cannot be set for a session. `cache_dir` can.

**Syntax**

    ALTER SESSION SET UDPARAMETER FOR vvector name = 'value';
    ALTER SESSION CLEAR UDPARAMETER FOR vvector name;
    ALTER SESSION CLEAR UDPARAMETER ALL;
    SHOW SESSION UDPARAMETER ALL;

`name`: k, precision, freshness, ef_search, exact, threads, rescore,
oversampling, cache_dir.

**Example 1: every search of the session sees the newest rows.**

    ALTER SESSION SET UDPARAMETER FOR vvector freshness = 'exact';
    SELECT r.rank, i.title
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='articles', query='[0.0, 0.6, 0.0, 0.6]', k=2) OVER()
          FROM ref.articles_delta) r JOIN ref.article_info i USING (id)
    ORDER BY r.rank;
    ALTER SESSION CLEAR UDPARAMETER FOR vvector freshness;

     rank |        title
    ------+---------------------
        1 | Cooking for runners
        2 | Marathon training

**Example 2: a batch job with its own k and precision.**

    ALTER SESSION SET UDPARAMETER FOR vvector k = '1';
    ALTER SESSION SET UDPARAMETER FOR vvector precision = 'best';
    SHOW SESSION UDPARAMETER ALL;
    SELECT q.qid, vvector.vknn(q.qvec USING PARAMETERS index_name='articles') FROM ref.questions q ORDER BY 1;
    ALTER SESSION CLEAR UDPARAMETER ALL;

     schema | library |    key    | value
    --------+---------+-----------+-------
     public | vvector | k         | 1
     public | vvector | precision | best

     qid | id |       score       | rank
    -----+----+-------------------+------
     100 |  2 | 0.997859001159668 |    1
     200 |  3 | 0.996946513652802 |    1
     300 |  7 | 0.776150465011597 |    1

### Inspect

#### vinfo

Shows, per node, what the node has in its cache: the snapshot, its size and
type, the index defaults, whether it can be read, how much of it is in
memory.

- **When you need it:** to check that every node has the active snapshot
  after a refresh or a `load_all`, and how much memory the indexes use.
- **Limits:** reads the cache directory only (the default, the session's, or
  the `cache_dir` parameter; an index with its own `cache_dir` option is
  found by its name). Rights: `vvector_search`.

**Syntax**

    SELECT ... FROM (SELECT vvector.vinfo([USING PARAMETERS index_name='name'] [, cache_dir='/dir'])
                            OVER(PARTITION NODES) FROM vvector.probe) i;

Output, one row per node and index: node_name, index_name, snapshot_id,
max_ver, vector_count, dims, metric, index_type, quantization, graph_bytes,
tombstones, base_snapshot, precision_default, freshness_default,
ef_search_default, threads_default, cache_file (or why the cache cannot be
read), loaded, resident_mb (how much of the file is in memory now),
capacity (the vectors the layout has room for), file_bytes, unreachable
(HNSW: live vectors no search can reach; NULL when not counted).
`vvector.probe` makes the function run once on every node.

**Example 1: does every node have the active snapshot?** Compare with the
manifest:

    SELECT node_name, index_name, snapshot_id, vector_count, dims, metric, index_type, loaded
    FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='articles') OVER(PARTITION NODES) FROM vvector.probe) i;
    SELECT active_snapshot FROM vvector.manifest WHERE index_name = 'articles';

       node_name    | index_name | snapshot_id | vector_count | dims | metric | index_type | loaded
    ----------------+------------+-------------+--------------+------+--------+------------+--------
     v_vdb_node0001 | articles   |          28 |            9 |    4 | cosine | hnsw       | t

     active_snapshot
    -----------------
                  28

**Example 2: the size and memory of every index on every node.** Without
`index_name` it lists every index in the cache directory:

    SELECT node_name, index_name, index_type, quantization, file_bytes, resident_mb
    FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) i
    ORDER BY node_name, index_name;

       node_name    | index_name | index_type | quantization | file_bytes | resident_mb
    ----------------+------------+------------+--------------+------------+-------------
     v_vdb_node0001 | articles   | hnsw       | none         |     912640 |           0
     v_vdb_node0001 | products   | flat       | none         |     312640 |           0

#### vversion

Returns the version of the library, the snapshot format and the build flags
(compiler, CPU, and the distance code chosen at load time).

- **When you need it:** to check what is deployed, for example in a support
  question or after an upgrade.
- **Limits:** none. Rights: `vvector_search`.

**Syntax**

    SELECT vvector.vversion() OVER();

Output: `(library_version, format_version, build_flags)`.

**Example 1: what is deployed.**

    SELECT vvector.vversion() OVER();

     library_version | format_version |                           build_flags
    -----------------+----------------+-----------------------------------------------------------------
     0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 x86_64 g++-15.2.0 kernels=avx2

**Example 2: every node runs the same library.** One row per distinct
version and CPU level:

    SELECT library_version, format_version, build_flags, COUNT(*) AS instances
    FROM (SELECT vvector.vversion() OVER(PARTITION NODES) FROM vvector.probe) v
    GROUP BY 1, 2, 3;

     library_version | format_version |                           build_flags                           | instances
    -----------------+----------------+-----------------------------------------------------------------+-----------
     0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 x86_64 g++-15.2.0 kernels=avx2 |         1

### Vector functions

Vertica 26.2 has no arithmetic on arrays (`ARRAY[1, 2] + ARRAY[3, 4]` is an
error). These functions add it. They are in schema `vvector`, for the role
`vvector_search` (every user with `make deploy SEARCH=public`), and run
fenced unless deployed with `FENCED=no` or `mixed`.

Common rules: they compute in FLOAT64; `ARRAY[INT]` and `ARRAY[NUMERIC]`
arguments are cast to `ARRAY[FLOAT]` (except Hamming and Jaccard, which take
`ARRAY[INT]`). A NULL argument gives NULL. Vectors of different lengths and
NULL elements are errors. Use Vertica's own functions where they exist:
`VECTOR_L2`, `COSINE_SIMILARITY`, `DOT_PRODUCT`, `VECTOR_MAGNITUDE`,
`APPLY_SUM` (the sum of the elements of one vector), `'[1.5, 2]'::ARRAY[FLOAT]`
from text and `TO_JSON(a)` to text.

#### vector_add

The element-wise sum of two vectors: `a[i] + b[i]`.

- **When you need it:** to combine vectors: two topics into one query, or a
  shift of a query towards a direction.
- **Limits:** the common rules above.

**Syntax**

    vvector.vector_add(a, b)   -- ARRAY[FLOAT]

**Example 1: two topics in one vector.**

    SELECT vvector.vector_add(ARRAY[0.9, 0.1, 0.0, 0.0], ARRAY[0.0, 0.1, 0.9, 0.0]) AS databases_and_travel;

     databases_and_travel
    ----------------------
     [0.9,0.2,0.9,0.0]

**Example 2: search for "databases" and "travel" together.** The sum of the
vectors of article 1 (databases) and article 5 (travel) is the query:

    SELECT n.rank, i.title, n.score
    FROM (SELECT vvector.vknn(vvector.vector_add(a.vec, b.vec) USING PARAMETERS index_name='articles', k=3)
          FROM ref.articles_live a, ref.articles_live b
          WHERE a.id = 1 AND b.id = 5) n
    JOIN ref.article_info i USING (id)
    ORDER BY n.rank;

     rank |             title             |       score
    ------+-------------------------------+-------------------
        1 | Databases on the road         | 0.917222082614899
        2 | Tuning a column store         | 0.773853957653046
        3 | Vertica projections explained | 0.711405336856842

#### vector_sub

The element-wise difference of two vectors: `a[i] - b[i]`.

- **When you need it:** to see how a vector changed, or for "more like
  this, less like that" queries.
- **Limits:** the common rules above.

**Syntax**

    vvector.vector_sub(a, b)   -- ARRAY[FLOAT]

**Example 1: how much did article 4 change?** The difference between its two
versions in the journal, and the size of that difference:

    SELECT vvector.vector_sub(v2.vec, v1.vec) AS change, VECTOR_MAGNITUDE(vvector.vector_sub(v2.vec, v1.vec)) AS size
    FROM ref.articles v1, ref.articles v2
    WHERE v1.id = 4 AND v2.id = 4 AND v2.ts > v1.ts;

                   change               |       size
    ------------------------------------+------------------
     [0.0,0.09999999999999998,0.0,-0.1] | 0.14142135623731

**Example 2: more like "Marathon training", less like "A week in Lisbon".**

    SELECT n.rank, i.title, n.score
    FROM (SELECT vvector.vknn(vvector.vector_sub(liked.vec, disliked.vec) USING PARAMETERS index_name='articles', k=3)
          FROM ref.articles_live liked, ref.articles_live disliked
          WHERE liked.id = 7 AND disliked.id = 5) n
    JOIN ref.article_info i USING (id)
    ORDER BY n.rank;

     rank |         title         |       score
    ------+-----------------------+--------------------
        1 | Marathon training     |  0.665426015853882
        2 | Cooking for runners   |   0.52849817276001
        3 | Tuning a column store | 0.0102221891283989

#### vector_mul

The element-wise product of two vectors: `a[i] x b[i]`.

- **When you need it:** to weight dimensions (a feature counts more) or to
  mask them (a 0 removes a dimension).
- **Limits:** the common rules above. A weighted distance is computed in
  SQL over every row; an index cannot use it.

**Syntax**

    vvector.vector_mul(a, b)   -- ARRAY[FLOAT]

**Example 1: the price counts three times as much.** Products nearest to the
running shoes when the price level is weighted by 3:

    SELECT p.name, VECTOR_L2(vvector.vector_mul(p.vec, ARRAY[3, 1, 1]), vvector.vector_mul(ARRAY[0.4, 0.9, 0.6], ARRAY[3, 1, 1])) AS weighted_distance
    FROM ref.products p
    ORDER BY 2 LIMIT 3;

         name      | weighted_distance
    ---------------+-------------------
     running shoes |                 0
     trail shoes   | 0.435889894354067
     yoga mat      | 0.806225774829855

**Example 2: keep only the sport and outdoor dimensions.**

    SELECT p.name, vvector.vector_mul(p.vec, ARRAY[0, 1, 1]) AS sport_and_outdoor_only
    FROM ref.products p
    ORDER BY p.id;

         name      | sport_and_outdoor_only
    ---------------+------------------------
     running shoes | [0.0,0.9,0.6]
     trail shoes   | [0.0,0.8,0.9]
     rain jacket   | [0.0,0.3,0.9]
     office chair  | [0.0,0.0,0.0]
     yoga mat      | [0.0,0.7,0.1]
     tent          | [0.0,0.2,1.0]

#### scalar_vector_mul

A vector multiplied by a number: `s x a[i]`.

- **When you need it:** for weighted combinations (with `vector_add`) and to
  change units.
- **Limits:** the common rules above; the number comes first.

**Syntax**

    vvector.scalar_vector_mul(s, a)   -- ARRAY[FLOAT]

**Example 1: one vector for a document, from its title (70%) and its body
(30%).**

    SELECT vvector.vector_add(vvector.scalar_vector_mul(0.7, title_vec), vvector.scalar_vector_mul(0.3, body_vec)) AS document_vec
    FROM (SELECT ARRAY[1.0, 0.0, 0.0, 0.0] AS title_vec, ARRAY[0.2, 0.4, 0.4, 0.0] AS body_vec) d;

         document_vec
    ----------------------
     [0.76,0.12,0.12,0.0]

**Example 2: daily usage counts as per-hour rates.**

    SELECT customer, day, vvector.scalar_vector_mul(1.0 / 24, vec) AS per_hour
    FROM ref.usage
    ORDER BY customer, day;

     customer |    day     |                           per_hour
    ----------+------------+--------------------------------------------------------------
     acme     | 2026-09-01 | [5.0,0.125,0.041666666666666667]
     acme     | 2026-09-02 | [3.333333333333333,0.20833333333333332,0.0]
     globex   | 2026-09-01 | [0.41666666666666665,1.6666666666666666,0.08333333333333333]
     globex   | 2026-09-02 | [0.8333333333333333,1.4583333333333333,0.16666666666666667]

#### vector_normalize

A vector divided by its length, so its length is 1.

- **When you need it:** before storing vectors for a `dot` index (the dot
  product of unit vectors is the cosine), and to compare the mix of a
  vector without its size.
- **Limits:** the common rules above. A zero vector stays zero. A `cosine`
  index normalises by itself at the build; you need not do it.

**Syntax**

    vvector.vector_normalize(a)   -- ARRAY[FLOAT]

**Example 1: a unit vector.**

    SELECT vvector.vector_normalize(ARRAY[3, 4]) AS unit, VECTOR_MAGNITUDE(vvector.vector_normalize(ARRAY[3, 4])) AS length;

       unit    | length
    -----------+--------
     [0.6,0.8] |      1

**Example 2: compare the usage mix of customers, not their volume.** acme
mostly queries, globex mostly loads, whatever the counts:

    SELECT customer, day, vvector.vector_normalize(vec) AS usage_mix
    FROM ref.usage
    ORDER BY customer, day;

     customer |    day     |                           usage_mix
    ----------+------------+---------------------------------------------------------------
     acme     | 2026-09-01 | [0.9996529585180931,0.02499132396295233,0.008330441320984109]
     acme     | 2026-09-02 | [0.9980525784828885,0.06237828615518053,0.0]
     globex   | 2026-09-01 | [0.2422507915557546,0.9690031662230184,0.04845015831115092]
     globex   | 2026-09-02 | [0.49371429861131246,0.8640000225697968,0.09874285972226249]

#### vector_l1

The Manhattan distance: the sum of `ABS(a[i] - b[i])`. The score of an `l1`
index.

- **When you need it:** for a distance that counts every difference equally
  (no squaring), for example between rating or count vectors.
- **Limits:** the common rules above.

**Syntax**

    vvector.vector_l1(a, b)   -- FLOAT

**Example 1: how far apart are two products?**

    SELECT a.name, b.name, vvector.vector_l1(a.vec, b.vec) AS l1_distance
    FROM ref.products a, ref.products b
    WHERE a.id = 10 AND b.id IN (11, 13)
    ORDER BY 3;

         name      |     name     | l1_distance
    ---------------+--------------+-------------
     running shoes | trail shoes  |         0.5
     running shoes | office chair |         1.8

**Example 2: the nearest products by l1, in plain SQL.** The same order an
`l1` index returns (it reads every row):

    SELECT name, vvector.vector_l1(vec, ARRAY[0.45, 0.85, 0.7]) AS l1_distance
    FROM ref.products
    ORDER BY 2 LIMIT 3;

         name      | l1_distance
    ---------------+-------------
     running shoes |         0.2
     trail shoes   |         0.3
     rain jacket   |         0.9

#### vector_l2sq

The squared straight-line distance: the sum of `(a[i] - b[i])^2`, which is
`VECTOR_L2` squared.

- **When you need it:** for thresholds and sums of squares, where the square
  root is not needed (cheaper, and sums of squares add up).
- **Limits:** the common rules above.

**Syntax**

    vvector.vector_l2sq(a, b)   -- FLOAT

**Example 1: pairs of near-identical products** (squared distance below
0.15):

    SELECT a.name, b.name, vvector.vector_l2sq(a.vec, b.vec) AS l2sq
    FROM ref.products a, ref.products b
    WHERE a.id < b.id AND vvector.vector_l2sq(a.vec, b.vec) < 0.15
    ORDER BY 3;

         name      |    name     | l2sq
    ---------------+-------------+------
     rain jacket   | tent        | 0.06
     running shoes | trail shoes | 0.11

**Example 2: how spread out are the articles of each customer?** The sum of
the squared distances to the customer's centroid (small = focused):

    SELECT i.customer, SUM(vvector.vector_l2sq(l.vec, c.centre)) AS spread
    FROM ref.articles_live l
    JOIN ref.article_info i ON i.id = l.id
    JOIN (SELECT customer, vector_avg AS centre
          FROM (SELECT i.customer, vvector.vector_avg(l.vec) OVER(PARTITION BY i.customer)
                FROM ref.articles_live l JOIN ref.article_info i ON i.id = l.id) a) c ON c.customer = i.customer
    GROUP BY i.customer
    ORDER BY 2;

     customer |      spread
    ----------+-------------------
     globex   | 0.353333333333333
     acme     | 0.986666666666667
     initech  |                 1

#### vector_hamming

The number of bits that differ between two bit vectors (`ARRAY[INT]`):
elements 0 and 1, or 64 bits packed in each element.

- **When you need it:** for binary fingerprints and hashes (SimHash,
  perceptual hashes): near-duplicates differ in few bits.
- **Limits:** `ARRAY[INT]` only; both arguments the same length.

**Syntax**

    vvector.vector_hamming(a, b)   -- INT

**Example 1: near-duplicate fingerprints** (one bit per element):

    SELECT a.id, b.id, vvector.vector_hamming(a.bits, b.bits) AS differing_bits
    FROM ref.fingerprints a, ref.fingerprints b
    WHERE a.id < b.id
    ORDER BY 3;

     id | id | differing_bits
    ----+----+----------------
      1 |  2 |              1
      2 |  3 |              7
      1 |  3 |              8

**Example 2: 64-bit hashes** (one element holds 64 bits):

    SELECT a.id, b.id, vvector.vector_hamming(a.h, b.h) AS differing_bits
    FROM ref.simhash a, ref.simhash b
    WHERE a.id < b.id
    ORDER BY 3;

     id | id | differing_bits
    ----+----+----------------
      1 |  2 |              2
      2 |  3 |             32
      1 |  3 |             34

#### vector_jaccard

The Jaccard (Tanimoto) similarity of two bit vectors: the bits set in both
divided by the bits set in either; 1 when neither has a bit set.

- **When you need it:** for sets coded as bits: tags, features, shopping
  baskets.
- **Limits:** `ARRAY[INT]` only (0 and 1 per element, or 64 packed bits).

**Syntax**

    vvector.vector_jaccard(a, b)   -- FLOAT

**Example 1: articles with tags like those of "Hiking the Alps".** Tags as
bits: sql, performance, food, travel, sport, outdoor:

    SELECT t.id, i.title, vvector.vector_jaccard(t.tags, s.tags) AS similarity
    FROM ref.article_tags t, ref.article_tags s
    JOIN ref.article_info i ON TRUE
    WHERE s.id = 6 AND t.id <> 6 AND i.id = t.id
    ORDER BY 3 DESC;

     id |             title             |    similarity
    ----+-------------------------------+-------------------
      7 | Marathon training             | 0.666666666666667
      5 | A week in Lisbon              | 0.333333333333333
      8 | Street food in Bangkok        |              0.25
      1 | Vertica projections explained |                 0

**Example 2: pairs of fingerprints that share at least half their bits.**

    SELECT a.id, b.id, vvector.vector_jaccard(a.bits, b.bits) AS similarity
    FROM ref.fingerprints a, ref.fingerprints b
    WHERE a.id < b.id AND vvector.vector_jaccard(a.bits, b.bits) >= 0.5;

     id | id | similarity
    ----+----+------------
      1 |  2 |        0.8

#### vector_sum

The element-wise sum of the vectors of a partition.

- **When you need it:** for totals per group: usage per customer, counts
  per day.
- **Limits:** a transform function, not an aggregate (Vertica 26.2
  aggregates cannot take an array): write it with `OVER()` for the whole
  input or `OVER(PARTITION BY ...)` per group, with only the partition
  columns beside it, and put anything else in an outer query. NULL vectors
  are skipped; a partition of only NULL vectors gives NULL.

**Syntax**

    SELECT [partition columns,] vvector.vector_sum(vec) OVER([PARTITION BY ...]) FROM table;   -- output column vector_sum

**Example 1: the usage of each customer.**

    SELECT customer, vector_sum AS total
    FROM (SELECT customer, vvector.vector_sum(vec) OVER(PARTITION BY customer) FROM ref.usage) s
    ORDER BY customer;

     customer |      total
    ----------+-----------------
     acme     | [200.0,8.0,1.0]
     globex   | [30.0,75.0,6.0]

**Example 2: the usage of all customers.**

    SELECT vvector.vector_sum(vec) OVER() AS all_customers FROM ref.usage;

      all_customers
    ------------------
     [230.0,83.0,7.0]

#### vector_avg

The element-wise average of the vectors of a partition: the centroid.

- **When you need it:** for the typical vector of a group (a customer, a
  topic, a user's likes), and as a query built from several vectors.
- **Limits:** a transform function like `vector_sum` (same rules).

**Syntax**

    SELECT [partition columns,] vvector.vector_avg(vec) OVER([PARTITION BY ...]) FROM table;   -- output column vector_avg

**Example 1: the centroid of each customer's articles.**

    SELECT customer, vector_avg AS centroid
    FROM (SELECT i.customer, vvector.vector_avg(l.vec) OVER(PARTITION BY i.customer)
          FROM ref.articles_live l JOIN ref.article_info i ON i.id = l.id) c
    ORDER BY customer;

     customer |                                    centroid
    ----------+---------------------------------------------------------------------------------
     acme     | [0.5666666666666668,0.06666666666666667,0.3333333333333333,0.03333333333333333]
     globex   | [0.03333333333333333,0.7999999999999999,0.26666666666666669,0.0]
     initech  | [0.2333333333333333,0.2333333333333333,0.13333333333333334,0.5]

**Example 2: recommend articles to a user.** The centroid of what user 1
liked is the query; the articles the user already liked are left out:

    SELECT i.title, n.score
    FROM (SELECT vvector.vknn(p.profile USING PARAMETERS index_name='articles', k=4)
          FROM (SELECT vector_avg AS profile
                FROM (SELECT vvector.vector_avg(l.vec) OVER()
                      FROM ref.articles_live l JOIN ref.likes k ON k.article_id = l.id
                      WHERE k.user_id = 1) a) p) n
    JOIN ref.article_info i USING (id)
    WHERE n.id NOT IN (SELECT article_id FROM ref.likes WHERE user_id = 1)
    ORDER BY n.score DESC;

             title         |       score
    -----------------------+-------------------
     Databases on the road | 0.937463581562042
     Pasta in ten minutes  | 0.168025434017181

### Build and load by hand

`refresh_index` and `load_all` call these functions; you need them only to
build or load a snapshot yourself, to check nodes, or to repair. They are in
schema `vvector_admin`, for the role `vvector_admin`, and run fenced unless
deployed with `FENCED=no`.

#### vnode

Returns one row per node: `(node_name, k)`, where k is one value of
`vvector.probe` stored on that node.

- **When you need it:** to send something to every node exactly once (the
  load joins the snapshot to these k values), and to check that every node
  answers.
- **Limits:** in Eon it reaches the nodes of the session's subcluster.

**Syntax**

    SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe;

**Example 1: the nodes that run vvector.**

    SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe;

       node_name    | k
    ----------------+---
     v_vdb_node0001 | 1

**Example 2: does every node that is up answer?**

    SELECT (SELECT COUNT(*) FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n) AS nodes_reached,
           (SELECT COUNT(*) FROM nodes WHERE node_state = 'UP') AS nodes_up;

     nodes_reached | nodes_up
    ---------------+----------
                 1 |        1

#### vconfig

Writes the query defaults of an index (the `OPTIONS` file) into the cache
directory of every node, where searches read them (a search cannot read the
manifest).

- **When you need it:** rarely: `set_index_options`, `refresh_index` and
  `load_all` call it with the defaults of the manifest. By hand it changes
  the defaults of the nodes until the next of those calls.
- **Limits:** `options` takes precision, freshness, ef_search, threads and
  memory_mode as `name=value` items; a missing name is the built-in default.

**Syntax**

    SELECT vvector_admin.vconfig(k USING PARAMETERS index_name='name', options='name=value,...'
                                 [, cache_dir='/dir'] [, index_cache_dir='/dir' | ''])
           OVER(PARTITION NODES) FROM vvector.probe;

Output: `(node_name, status)`. `index_cache_dir` (the index option; `''` =
none) writes the defaults there, plus an `OPTIONS` file that names that
directory in the default one, so a search finds the index.

**Example 1: try precision best and 2 threads on every node.**

    SELECT vvector_admin.vconfig(k USING PARAMETERS index_name='articles', options='precision=best,threads=2')
           OVER(PARTITION NODES) FROM vvector.probe;
    SELECT node_name, precision_default, threads_default
    FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='articles') OVER(PARTITION NODES) FROM vvector.probe) i;

       node_name    | status
    ----------------+---------
     v_vdb_node0001 | written

       node_name    | precision_default | threads_default
    ----------------+-------------------+-----------------
     v_vdb_node0001 | best              | 2

**Example 2: back to the built-in defaults.** An empty `options`; the
manifest's defaults come back with the next `load_all` or refresh:

    SELECT vvector_admin.vconfig(k USING PARAMETERS index_name='articles', options='')
           OVER(PARTITION NODES) FROM vvector.probe;
    SELECT node_name, precision_default, threads_default
    FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='articles') OVER(PARTITION NODES) FROM vvector.probe) i;

       node_name    | status
    ----------------+---------
     v_vdb_node0001 | written

       node_name    | precision_default | threads_default
    ----------------+-------------------+-----------------
     v_vdb_node0001 |                   |

#### vbuild

Turns `(id, vector)` rows into a snapshot and returns it as pieces
`(byte_offset, chunk, base_snapshot, vector_count, dims, max_ver,
format_version)`: chunks of 8 MB (all-zero ones left out), or for an
incremental build only the changed bytes.

- **When you need it:** to build a snapshot yourself (for example of a
  query result that is no registered table), or to see its size before a
  refresh.
- **Limits:** one row per id and no delete rows for a full build (read the
  live rows, not the journal). No ORDER BY: it sorts by id. It runs on one
  node and needs the memory of the snapshot there (see [sizing](#sizing));
  `build_in='file'` builds in a file instead. Rights: `vvector_admin`.

**Syntax**

    SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS index_name='name' [, parameter=value ...]) OVER()
    FROM rows;

| Parameter | Default | Meaning |
|---|---|---|
| index_name | (required) | the name the snapshot is for |
| metric | l2 | l2, cosine, dot, l1 |
| index_type | flat | flat or hnsw (the procedures pass the index's type) |
| m, ef_construction | 16, 200 | HNSW graph |
| threads | 0 (one per core) | threads of the graph build |
| quantization | none | none or sq8 |
| max_ver | | the version boundary written into the snapshot |
| base_snapshot | | build incrementally from this snapshot in the cache of the node; the rows are the changes (`del = true` deletes) |
| growth | 5 | room to grow, percent of the vectors (at least 4096; 0 = none) |
| send | patch | incremental builds: `patch` returns only the changed bytes, `whole` the whole snapshot |
| patch_row_mb | 1 | the size of a patch row, 1 to 8 MB |
| reachability | auto | HNSW: auto, on, off (count the vectors no search can reach) |
| build_in | ram | ram, or file (an unlinked file in the index's cache directory) |
| cache_dir | | the cache directory (for base_snapshot and build_in='file') |

**Example 1: how large would the snapshot be?** Build it and only count the
pieces:

    SELECT COUNT(*) AS pieces, SUM(LENGTH(chunk)) AS bytes, MAX(vector_count) AS vectors, MAX(dims) AS dims
    FROM (SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='articles_copy', metric='cosine', index_type='hnsw')
                 OVER()
          FROM ref.articles_live) b;

     pieces | bytes  | vectors | dims
    --------+--------+---------+------
          1 | 912640 |       9 |    4

**Example 2: build a snapshot of the live articles and store it.** The
snapshot id comes from you (use `NEXTVAL('vvector.snapshot_seq')` if the
index is also refreshed by the procedures); [vload](#vload) then loads it:

    INSERT INTO vvector.snapshot (index_name, snapshot_id, byte_offset, chunk, base_snapshot)
    SELECT 'articles_copy', 900, byte_offset, chunk, base_snapshot
    FROM (SELECT vvector_admin.vbuild(id, vec, FALSE USING PARAMETERS index_name='articles_copy', metric='cosine', index_type='hnsw')
                 OVER()
          FROM ref.articles_live) b;
    COMMIT;
    SELECT snapshot_id, COUNT(*) AS pieces FROM vvector.snapshot WHERE index_name = 'articles_copy' GROUP BY 1;

     snapshot_id | pieces
    -------------+--------
             900 |      1

#### vload

Writes a snapshot from its pieces into the cache of every node, verifies it
and makes it the active one. Returns `(node_name, snapshot_id, bytes,
status)`.

- **When you need it:** to load a snapshot you built with `vbuild`, or into
  another cache directory. `load_all` does it for registered indexes.
- **Limits:** every node must receive every piece: the pieces are joined to
  one probe row per node, and the two hints make Vertica broadcast them (on
  one node Vertica warns that the hint is not feasible and runs the
  statement as written). The join holds the pieces in memory on every node,
  so the procedures load a snapshot above 2 GB in passes (`part`, `pass`,
  `passes`). A node refuses a snapshot id lower than the one of the views.
  Rights: `vvector_admin`.

**Syntax**

    SELECT vvector_admin.vload(byte_offset, chunk, base_snapshot
                               USING PARAMETERS index_name='name', snapshot_id=n [, cache_dir='/dir'] [, part='p', pass=i, passes=n])
           OVER(PARTITION NODES)
    FROM (<the pieces, joined to one vvector.probe row per node>) c;

A piece with `base_snapshot` set is a patch: the node starts from its file
of that snapshot (an error when it is missing: run `load_all`) and writes
the changed bytes over it.

**Example 1: load the snapshot of the `vbuild` example and search it.** The
index `articles_copy` exists only in the node caches (no manifest row), so
`vknn` searches it by name:

    SELECT vvector_admin.vload(byte_offset, chunk, base_snapshot USING PARAMETERS index_name='articles_copy', snapshot_id=900)
           OVER(PARTITION NODES)
    FROM (SELECT /*+SYNTACTIC_JOIN*/ s.byte_offset, s.chunk, s.base_snapshot
          FROM vvector.probe p JOIN /*+DISTRIB(L,B)*/ vvector.snapshot s ON TRUE
          WHERE s.index_name = 'articles_copy' AND s.snapshot_id = 900
            AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c;
    SELECT n.rank, n.id, n.score
    FROM (SELECT vvector.vknn(NULL::ARRAY[FLOAT] USING PARAMETERS index_name='articles_copy', query='[0.9, 0.1, 0.0, 0.0]', k=2)
          FROM dual) n
    ORDER BY n.rank;

    WARNING 6818:  Input operations specified for Hint Distrib(L,B) is not feasible and will be ignored
       node_name    | snapshot_id | bytes  | status
    ----------------+-------------+--------+--------
     v_vdb_node0001 |         900 | 912640 | loaded

     rank | id |       score
    ------+----+-------------------
        1 |  1 |                 1
        2 |  2 | 0.978709042072296

**Example 2: load it into another directory.** For a test next to the
production caches:

    SELECT vvector_admin.vload(byte_offset, chunk, base_snapshot
                               USING PARAMETERS index_name='articles_copy', snapshot_id=900, cache_dir='/tmp/vvector_test')
           OVER(PARTITION NODES)
    FROM (SELECT /*+SYNTACTIC_JOIN*/ s.byte_offset, s.chunk, s.base_snapshot
          FROM vvector.probe p JOIN /*+DISTRIB(L,B)*/ vvector.snapshot s ON TRUE
          WHERE s.index_name = 'articles_copy' AND s.snapshot_id = 900
            AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c;
    SELECT node_name, index_name, snapshot_id, cache_file
    FROM (SELECT vvector.vinfo(USING PARAMETERS cache_dir='/tmp/vvector_test') OVER(PARTITION NODES) FROM vvector.probe) i;

    WARNING 6818:  Input operations specified for Hint Distrib(L,B) is not feasible and will be ignored
       node_name    | snapshot_id | bytes  | status
    ----------------+-------------+--------+--------
     v_vdb_node0001 |         900 | 912640 | loaded

       node_name    |  index_name   | snapshot_id |               cache_file
    ----------------+---------------+-------------+----------------------------------------
     v_vdb_node0001 | articles_copy |         900 | /tmp/vvector_test/articles_copy/900.vv

## Search

Patterns for the search functions on the index of the [Quick start](#quick-start)
(`app.docs`). The syntax and every parameter are in the reference:
[vsearch](#vsearch), [vknn](#vknn), [vscan](#vscan).

### One query, snapshot only (the fastest statement)

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

`app.docs_snap` is one row that carries the active snapshot id. The search
itself is the index's: on an HNSW index (the default) the graph with
`precision='balanced'`, on a flat index every vector. Pass the
query vector as the `query` parameter, not as an `ARRAY[...]` literal:
Vertica needs about 7 ms to parse a literal of 128 numbers, the parameter
costs nothing measurable. To build the text from a stored vector:
`SELECT TO_JSON(qvec) FROM ...` (it prints enough digits to read back the
same value).

The same search without a view reads no table at all. It gives up the stale
check (a node that missed a refresh answers from its old snapshot, see
[vknn](#vknn)), and the input columns must be
typed NULLs:

    SELECT vvector.vsearch(NULL::INT, NULL::ARRAY[FLOAT], NULL::INT, NULL::ARRAY[FLOAT], NULL::BOOLEAN, NULL::INT, NULL::INT
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM dual;

### One query as a row

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', k=2) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT 7, ARRAY[0.0, 1.0, 0.1], NULL, NULL, NULL, NULL, NULL) q;

     qid | id |       score       | rank
    -----+----+-------------------+------
       7 |  3 | 0.995037198066711 |    1
       7 |  5 | 0.703597545623779 |    2

### Many queries from a table

The queries are rows of a table:

    CREATE TABLE app.questions (qid INT, qvec ARRAY[FLOAT]);
    INSERT INTO app.questions VALUES (100, ARRAY[1.0, 0.2, 0.0]);
    INSERT INTO app.questions VALUES (200, ARRAY[0.1, 0.1, 1.0]); COMMIT;

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', k=2) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM app.questions) q
    ORDER BY qid, rank;

     qid | id |       score       | rank
    -----+----+-------------------+------
     100 |  2 | 0.996240615844727 |    1
     100 |  1 | 0.980580687522888 |    2
     200 |  4 | 0.990147531032562 |    1
     200 |  5 | 0.140028014779091 |    2

One statement with many queries is much faster than one statement per query:
on the test machine 1000 queries on 1M vectors take 25 to 35 ms with HNSW at
the default precision (29,000 to 40,000 queries per second; 12 to 22 ms with
`precision='fast'`) and about 1.1 s with a flat index.

### Precision, ef_search and exact search on an HNSW index

HNSW finds most, not always all, of the true nearest neighbours. `precision`
chooses how hard it looks; `ef_search` sets the same thing as a number.
`precision='exact'` or `exact=true` reads every vector, as a flat index does:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, precision='exact') OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  2 | 0.996240615844727 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, ef_search=200) OVER()
    FROM app.docs_snap;

(same result on this small index). What each level gives on 1M vectors is in
[Index types and tuning](#index-types-and-tuning).

### With the changes since the refresh

After the refresh above, vector 6 is added and vector 2 is deleted:

    INSERT INTO app.docs (id, vec) VALUES (6, ARRAY[1.0, 0.25, 0.0]);
    INSERT INTO app.docs (id, del) VALUES (2, TRUE);
    COMMIT;

The delta view now holds these two rows and the sentinel (the row that
carries the snapshot id). The five rows of the quick start are in the
snapshot: the refresh built every row up to its boundary, and with margin 0
that is every row written before it (see [Freshness](#freshness-explained)).

    SELECT id, del, ver IS NOT NULL AS has_ver, snapshot_id FROM app.docs_delta ORDER BY id;

     id | del | has_ver | snapshot_id
    ----+-----+---------+-------------
        |     | f       |         959
      2 | t   | t       |         959
      6 | f   | t       |         959

With `freshness='exact'`, id 6 is found and id 2 is gone; with the default
`snapshot` the result is the one of the last refresh:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, freshness='exact') OVER()
    FROM app.docs_delta;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

### Range search

The examples from here on use the index after one more refresh and three more
changes (the colours 7 and 8 are two shades of blue, 3 becomes a lighter
green), refreshed again:

    CALL vvector.refresh_index('docs');
    INSERT INTO app.docs (id, vec) VALUES (3, ARRAY[0.1, 0.9, 0.0]);
    INSERT INTO app.docs (id, vec) VALUES (7, ARRAY[0.2, 0.2, 0.9]);
    INSERT INTO app.docs (id, vec) VALUES (8, ARRAY[0.3, 0.1, 0.9]);
    COMMIT;
    CALL vvector.refresh_index('docs');

The live vectors are now 1, 3, 4, 5, 6, 7 and 8 (2 was deleted above).

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=10, radius=0.9, freshness='exact') OVER()
    FROM app.docs_delta;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2

At most k rows are returned. To get every vector within the radius, ask for
a large k; on an HNSW index that costs only as much as the vectors within the
radius, not k:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=16384, radius=0.9) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2

    SELECT vvector.vknn(ARRAY[0, 0.1, 1] USING PARAMETERS index_name='docs', k=16384, radius=0.95);

     id |       score       | rank
    ----+-------------------+------
      4 | 0.995037198066711 |    1
      7 | 0.970358312129974 |    2

How it works on HNSW: the graph search starts with the ef_search of the
precision level (fast: 32) and makes its candidate list four times longer,
up to k, as long as more than a quarter of the list is within the radius. A
narrow radius therefore stops after the first walk; a wide one grows to k.
On SIFT1M (1M vectors, k 16384) the search with a radius that holds about 10
vectors per query answers 2,674 queries per second against 485 when the list
is k long from the start, at recall 0.9999 (docs/design.md). Like every
graph search it is approximate: add `exact=true` for a complete answer.

### Filtered search

Only some vectors may be results: the documents of one customer, one
language, one category. Send their ids as allow-list rows (id set, vec and
del NULL) beside the query; vsearch returns only those ids. The examples
group the colours of the Quick start into families:

    CREATE TABLE app.families (id INT, family VARCHAR(20));
    INSERT INTO app.families VALUES (1, 'red'); INSERT INTO app.families VALUES (6, 'red');
    INSERT INTO app.families VALUES (5, 'warm'); INSERT INTO app.families VALUES (3, 'green');
    INSERT INTO app.families VALUES (4, 'blue'); INSERT INTO app.families VALUES (7, 'blue');
    INSERT INTO app.families VALUES (8, 'blue'); COMMIT;

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM app.families WHERE family = 'blue') x;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  8 |  0.32893693447113 |    1
       0 |  7 | 0.249459221959114 |    2
       0 |  4 |                 0 |    3

(Without the filter the same query returns 6, 1 and 5.) The filter applies to the journal rows too:
with `freshness='exact'` and the `_delta` view, a vector added since the
refresh is returned only when its id is allowed. Ids that are not in the
index are ignored.

A filter that matches nothing sends no allow-list row, and without any
allow-list row vsearch does not filter. When the filter may be empty, add
`filtered=true`: it returns nothing instead of the unfiltered answer.

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3, filtered=true) OVER()
    FROM (SELECT * FROM app.docs_snap
          UNION ALL SELECT NULL, NULL, id, NULL, NULL, NULL, NULL FROM app.families WHERE family = 'purple') x;

     qid | id | score | rank
    -----+----+-------+------
    (0 rows)

How it is searched: when fewer than max(10,000, sqrt(64 x ef_search x
vectors in the index)) allowed vectors remain (80,000 for 1M vectors at the
default precision), vsearch reads exactly those vectors: the answer is exact
and fast. With more, it walks the graph (or scans a flat index) and skips
the vectors outside the list. On SIFT1M a filter of 1% of the vectors
answers 75,700 queries per second in a batch (0.25 ms for a single query);
50% of the vectors 24,400 per second with recall 0.99 (docs/design.md).
The allow-list is part of the statement's input, so a list of millions of
ids costs the time Vertica needs to send them.

On a cluster, vsearch runs on the node that received the statement, and
allow-list rows from a segmented table must first be gathered from every
node. On the 4-node test cluster that costs about 24 ms whatever the list
size (one search, mixed, client median):

| allowed ids | from an UNSEGMENTED ALL NODES table | from a segmented table |
|---|---:|---:|
| no filter | 6.8 ms | 6.8 ms |
| 100 | 8.7 ms | 32.9 ms |
| 10,000 | 14.2 ms | 43.3 ms |
| 100,000 | 41.2 ms | 68.9 ms |

So keep the filter columns a search uses (id plus the category, tenant or
language) in a small table that is `UNSEGMENTED ALL NODES`, or give such a
table an unsegmented projection. Every node then has a full copy, and the
rows are read where vsearch runs. On one node it makes no difference.

For a filter that keeps most rows, searching without it and filtering the
results can be simpler; see [Recipes](#recipes).

### Join the results to your data

The payload stays in your own tables:

    CREATE TABLE app.titles (id INT, title VARCHAR(40));
    INSERT INTO app.titles VALUES (1, 'red'); INSERT INTO app.titles VALUES (2, 'dark red');
    INSERT INTO app.titles VALUES (3, 'green'); INSERT INTO app.titles VALUES (4, 'blue');
    INSERT INTO app.titles VALUES (5, 'yellow'); INSERT INTO app.titles VALUES (6, 'orange'); COMMIT;

    SELECT r.rank, r.id, t.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
          FROM app.docs_snap) r
    JOIN app.titles t ON t.id = r.id
    ORDER BY r.rank;

     rank | id |  title   |       score
    ------+----+----------+-------------------
        1 |  6 | orange   |  0.99886816740036
        2 |  1 | red      | 0.980580687522888
        3 |  5 | yellow   | 0.832050263881683

### Recipes

The examples use the live rows of the journal as a view:

    CREATE VIEW app.docs_live AS
    SELECT id, vec FROM (SELECT id, vec, del, ROW_NUMBER() OVER(PARTITION BY id ORDER BY ts DESC, del DESC) AS rn
                         FROM app.docs) j
    WHERE rn = 1 AND NOT del;

**Filter after the search** (post-filter). Ask for more neighbours than you
need, join, filter and keep the first rows. Simple, and fine when the filter
removes few rows; with a selective filter use [Filtered search](#filtered-search).

    SELECT r.id, f.family, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=30) OVER()
          FROM app.docs_snap) r
    JOIN app.families f ON f.id = r.id
    WHERE f.family <> 'red'
    ORDER BY r.rank LIMIT 3;

     id | family |       score
    ----+--------+-------------------
      5 | warm   | 0.832050263881683
      8 | blue   |  0.32893693447113
      3 | green  | 0.303203642368317

**Recommendation by centroids.** "More like these, less like that": the
query is the average of the liked vectors minus the average of the disliked
ones ([Vector functions](#vector-functions)).

    SELECT r.rank, r.id, t.title, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='docs', k=3) OVER()
          FROM (SELECT * FROM app.docs_snap
                UNION ALL
                SELECT 1, vvector.vector_sub(liked.vector_avg, disliked.vector_avg), NULL, NULL, NULL, NULL, NULL
                FROM (SELECT vvector.vector_avg(vec) OVER() FROM app.docs_live WHERE id IN (1, 5)) liked
                CROSS JOIN (SELECT vvector.vector_avg(vec) OVER() FROM app.docs_live WHERE id = 4) disliked) x) r
    LEFT JOIN app.titles t ON t.id = r.id ORDER BY r.rank;

     rank | id | title  |       score
    ------+----+--------+-------------------
        1 |  6 | orange | 0.618346929550171
        2 |  1 | red    | 0.588348388671875
        3 |  5 | yellow | 0.554700195789337

Add the liked ids as allow-list rows the other way round (or filter them
out afterwards) to leave them out of the answer.

**The best match per group.** Search once with a k large enough to reach
every group, then keep the first row of each group:

    SELECT family, id, score FROM (
      SELECT f.family, r.id, r.score, ROW_NUMBER() OVER(PARTITION BY f.family ORDER BY r.rank) AS n
      FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                   USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=100) OVER()
            FROM app.docs_snap) r
      JOIN app.families f ON f.id = r.id) g
    WHERE n = 1 ORDER BY score DESC;

     family | id |       score
    --------+----+-------------------
     red    |  6 |  0.99886816740036
     warm   |  5 | 0.832050263881683
     blue   |  8 |  0.32893693447113
     green  |  3 | 0.303203642368317

**Near-duplicates.** Every vector as a query, k 2 (the vector itself and its
nearest other one), pairs above a threshold:

    SELECT r.qid AS id, r.id AS near_id, r.score
    FROM (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                                 USING PARAMETERS index_name='docs', k=2) OVER()
          FROM (SELECT * FROM app.docs_snap
                UNION ALL SELECT id, vec, NULL, NULL, NULL, NULL, NULL FROM app.docs_live) x) r
    WHERE r.id <> r.qid AND r.score >= 0.97 AND r.qid < r.id
    ORDER BY r.score DESC;

     id | near_id |       score
    ----+---------+-------------------
      7 |       8 |  0.98894989490509
      1 |       6 | 0.970142483711243

For a large table run it in slices of query ids (`WHERE id % 10 = 0`, ...):
each statement searches all its query rows in one call.

## Index types and tuning

Two index types, chosen at registration (`register_index`, last argument) or
later with `set_index_options` (applies at the next refresh):

- **hnsw** (default): a graph over the vectors (HNSW, Malkov and Yashunin
  2018, built the way hnswlib builds it). A search walks the graph from an
  entry point towards the query and reads a few thousand vectors instead of
  all of them. It is approximate: it finds most of the true neighbours, and
  `precision` says how many. On 1M vectors of 128 dimensions one search
  takes 0.05 ms in the engine instead of 2.3 ms; a single-search statement is
  about 2.6 times faster (the rest is the cost of the statement), a batch of
  1000 queries 50 to 90 times.
- **flat**: every search reads every vector (with SIMD instructions and all
  cores). Always exact. Choose it for small indexes (up to about 100,000
  vectors a flat search takes well under a millisecond of engine time), when
  every answer must be exact, or when refreshes must be as fast as possible
  (no graph to build).

`precision` levels on SIFT1M (1M vectors of 128 dimensions, k = 10, 1000
queries, recall@10 = the share of the true 10 nearest neighbours found):

| precision | ef_search | recall@10 | 1000 queries in one statement | one search in the engine |
|---|---:|---:|---:|---:|
| fast | 2 x k, at least 32 | 0.892 | 14 ms | 0.05 ms |
| balanced (default) | 100 | 0.980 | 28 ms | 0.14 ms |
| best | 400 | 0.999 | not measured | 0.46 ms |
| exact | (every vector) | 0.999 (the rest are ties) | 1.1 s (measured on the flat index) | 9 ms (1 thread) |

A single statement costs about 1.5 ms more than the engine time (see
[Performance and results](#performance-and-results)), so for single searches
`balanced` costs little more than `fast` (0.1 ms), and it is the default. Use
`fast` for large batches where throughput counts more than the last 9% of
recall. Recall depends on the data: measure
it on your own vectors with `precision='exact'` as the reference. For 100
queries of a query table (here SIFT1M in schema VVBENCH):

    WITH q AS (SELECT * FROM VVBENCH.sift_hnsw_snap
               UNION ALL SELECT qid, qvec, NULL, NULL, NULL, NULL, NULL FROM VVBENCH.sift_query WHERE qid < 100),
         a AS (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10) OVER() FROM q),
         e AS (SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id USING PARAMETERS index_name='sift_hnsw', k=10, precision='exact') OVER() FROM q)
    SELECT (COUNT(a.id) / COUNT(*))::NUMERIC(5,3) AS recall
    FROM e LEFT JOIN a ON a.qid = e.qid AND a.id = e.id;

     recall
    --------
      0.990

Add the settings you want to try to the parameters of `a`, for example
`precision='fast'` or `ef_search=150`.

Build options of an HNSW index (`set_index_options`, then `refresh_index`):

| Option | Default | Effect |
|---|---:|---|
| m | 16 | links per vector (2 x m on the lowest level). Higher: better recall for high-dimensional data, more memory ((2m + 1) x 4 bytes per vector and more), slower build |
| ef_construction | 200 | the candidate list while building. Higher: a better graph and recall, slower build |

`ef_search` sets the candidate list of a search directly and overrides
`precision`; per query, per session (`ALTER SESSION SET UDPARAMETER FOR
vvector ef_search = '150'`) or per index (`set_index_options`).

#### Unreachable vectors

A search walks the graph along links from one entry point. A vector that no
chain of links leads to is never returned by an approximate search (an exact
search still finds it). The build links every vector it finds without a link
back into the graph, and then counts what is left: one walk over the lowest
level after the build, stored in the snapshot. `vinfo` shows it as
`unreachable`, the manifest keeps it, `status` prints it, and `refresh_note`
names it when it is above 0 or was not counted. The count is 0 on SIFT1M and
on the generated test sets. The option `reachability` (`set_index_options`)
chooses when to count: `auto` (the default) at every full build and at
incremental builds of graphs below 8 million vectors, `on` at every build,
`off` never. The walk reads the whole lowest level of the graph: 0.11 s for
1 million vectors on 8 cores, so about 10 s for 100 million when the graph
is in memory and minutes when it has to come from disk, against an
incremental refresh of seconds; that is why `auto` skips it on large
incremental builds. A count above 0 after an incremental refresh usually
goes back to 0 with `refresh_index(name, 'full')`.

| If you want | Set |
|---|---|
| the lowest latency for single queries | the `query` parameter, `FROM <index>_snap`, deploy with `FENCED=mixed`; `vknn` is a little faster still but has no stale check |
| higher recall | `precision='best'`, or a larger `ef_search`; for all queries of an index: `set_index_options` |
| more throughput in large batches | `quantization='sq8'` (same recall), or `precision='fast'` (recall 0.89 instead of 0.98 on SIFT1M) |
| exact answers on an HNSW index | `precision='exact'` or `exact=true` for that query |
| results that include every committed change | `freshness='exact'` and `FROM <index>_delta` (per query, session or index) |
| many queries at once | one vsearch statement with all query rows (a table), not one statement per query |
| fewer cores for one statement | `threads=N` |
| the same default for every user of an index | `set_index_options` |
| faster searches, same exact scores | `quantization='sq8'` (see below) |

### int8 quantisation (sq8)

With `quantization='sq8'` the snapshot also stores every element of every
vector as one byte: a code from 0 to 255 in one range for the whole index,
trained on a sample of the vectors at a full build. A search first ranks the
candidates by these bytes (a quarter of the memory to read, integer
arithmetic), then computes the exact scores of the best k x oversampling
candidates from the float vectors (rescoring) and returns the k best of them.
The scores you get are exact; only the choice of candidates is approximate.

    CALL vvector.set_index_options('docs', NULL, NULL, NULL, 'sq8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    CALL vvector.refresh_index('docs');

    NOTICE 2005:  vvector: index docs refreshed: snapshot 2016, full build (build options changed from hnsw cosine none m=16 ef_construction=200 to hnsw cosine sq8 m=16 ef_construction=200), 7 vectors of 3 dimensions, 0 tombstones, 0 MB, 0.307 seconds; journal digest taken in 0.015 seconds

The same search as before; the scores are exact (rescoring), and with
`precision='fast'` they come from the codes alone:

    SELECT vvector.vsearch(qid, qvec, id, vec, del, ver, snapshot_id
                           USING PARAMETERS index_name='docs', query='[1, 0.2, 0]', k=3) OVER()
    FROM app.docs_snap;

     qid | id |       score       | rank
    -----+----+-------------------+------
       0 |  6 |  0.99886816740036 |    1
       0 |  1 | 0.980580687522888 |    2
       0 |  5 | 0.832050263881683 |    3

    -- precision='fast':
       0 |  6 | 0.997308850288391 |    1
       0 |  1 | 0.980392277240753 |    2
       0 |  5 | 0.830449938774109 |    3

`vinfo` shows the codes on every node (column `quantization`: `sq8`), and
`memory_mode` can then be set:

    CALL vvector.set_index_options('docs', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'compact', NULL, NULL, NULL, NULL);
    -- and back without codes needs memory_mode ram in the same call:
    CALL vvector.set_index_options('docs', NULL, NULL, NULL, 'none', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    ERROR 2005:  vvector.set_index_options: memory_mode compact needs quantization sq8

Works with both index types, every metric, the journal (its rows are always
searched exactly) and incremental refresh (which keeps the range). The
`precision` levels on an index with sq8:

| precision | codes | rescoring |
|---|---|---|
| fast | graph walk or scan on the codes | none: the scores are approximate |
| balanced (default) | the same | 2 x k candidates rescored; 4 x k from 512 dimensions on |
| best | the same, ef_search 400 | 4 x k candidates rescored |
| exact | not used: every float vector is read | |

`rescore` and `oversampling` override the preset per query or per session
(not per index). SIFT1M (1M vectors of 128 dimensions, k = 10), engine alone:

| Search | recall@10 float | recall@10 sq8 | queries/s float | queries/s sq8 |
|---|---:|---:|---:|---:|
| HNSW, ef_search 100, rescoring 2 x k (balanced) | 0.983 | 0.983 | 48,900 / 35,500 | 72,600 / 60,300 |
| HNSW, ef_search 100, no rescoring | 0.983 | 0.970 | 48,900 / 35,500 | 73,500 / 61,500 |
| flat, rescoring 2 x k | 0.999 | 0.999 | 832 / 576 | 1,190 / 708 |

(queries/s: the test VM with 8 aarch64 cores / a 10-core x86_64 node, all
threads, 10,000 queries; `make test DATA_DIR=...`.) Through SQL, recall@10 of
1000 queries: fast 0.884, balanced 0.979, best 0.999, against 0.892, 0.980
and 0.999 without sq8.

On generated vectors of 768 and 1536 dimensions (metric cosine, 1M vectors)
sq8 lost more with 2 x k rescored: 0.933 and 0.932 against 0.973 and 0.962
without codes. With 4 x k rescored it gave 0.965 and 0.956 and was still 30%
faster than the float index, so from 512 dimensions on balanced rescores
4 x k; an `oversampling` you set (per query or per session) replaces
that preset. `precision='best'` gave 0.992 and 0.996 and was no slower than
the float index at balanced. Measure the recall on your own vectors (the query in
[Precision, ef_search and exact search](#precision-ef_search-and-exact-search-on-an-hnsw-index))
before you choose sq8 for high dimensions.

Use it when the index is large and searches read much memory (flat indexes,
large batches, high dimensions). The codes do not replace the float vectors,
they come in addition: 1M x 128 grows from 631 MB to 757 MB as an HNSW
index. With `memory_mode='compact'` a node keeps the codes, ids and graph in
memory and reads float vectors only for rescoring (see the option table).

Memory:

| What | How much |
|---|---|
| snapshot and cache file per node | 256 bytes + (4 x row_stride + 8) bytes per vector; row_stride = dims rounded up to a multiple of 16 |
| HNSW graph (in the snapshot) | about (2m + 1) x 4 + 5 + (m + 1) x 4 / (m - 1) bytes per vector: 141 bytes with m = 16 |
| sq8 codes (in the snapshot) | row_stride + 4 bytes per vector: 132 bytes with 128 dimensions |
| a refresh (on one node) | full build: about the snapshot size + 4 bytes per vector; HNSW adds 5 + 2 x cores bytes per vector. Incremental: the new snapshot + 4 x row_stride bytes per changed row + 2 x cores bytes per vector (HNSW); the base is read from the cache file. Fenced it counts against `FencedUDxMemoryLimitMB` (-1 = no limit) |
| a query | 4 x row_stride bytes per query and per journal row, plus 1 bit per vector when the journal has rows; HNSW: 2 bytes per vector per search thread for the visited marks, kept by the process between queries |

1,000,000 vectors of 128 dimensions take 496 MB as a flat index and 631 MB as
an HNSW index; of 768 dimensions, 2.9 GB and 3.0 GB. Queries read the cache
file through the operating system's page cache: keep all indexes of a node in
memory (`status` warns when they do not fit). `CALL vvector.sizing(...)`
estimates an index before you load it.

## Freshness explained

- A query with `freshness='snapshot'` (the default) sees the vectors of the
  last refresh.
- A query with `freshness='exact'` over the `_delta` view sees every
  committed change: the view returns the journal rows with a version after
  the boundary of the last refresh, and vsearch applies them on top of the
  snapshot (the latest row per id wins, a delete removes the id).
- The boundary is taken before the refresh reads the table: the earlier of
  "now" and the start of the oldest open transaction that writes to the
  table, minus the margin (default 60 s). The refresh builds the rows up to
  the boundary; the delta view returns the rows after it; the two never
  overlap. So the rows of the last margin seconds before a refresh stay in
  the delta (`status` counts them) and a query with `freshness='snapshot'`
  sees them after the next refresh. A refresh that finds no live vector up
  to the boundary, such as the first refresh right after a load, stops with
  an error that says so: refresh again when the margin has passed.
- An open transaction that writes to the table holds the boundary back:
  everything written after its start stays in the delta until it ends. When
  it is older than ten times the margin (at least 60 s), the refresh note
  says so ("the boundary lags N minutes behind ...").
- Why the snapshot takes nothing after the boundary: the journal digest
  (see [refresh_index](#refresh_index)) covers the rows up to the boundary, so a physical DELETE or UPDATE
  of any row in the snapshot is found at the next refresh.
- Every query over the delta pays for its rows: about 1.7 ms per 1000 rows
  of 128 numbers on one node. `status` warns above 100,000 rows or when
  reading the delta is slow; refresh more often then. On a cluster the delta
  read also costs a transfer between nodes (see [Operations](#operations)).
- The journal works the same with both index types: journal rows are searched
  exactly and merged with the result of the graph or flat search; ids changed
  or deleted in the journal are never returned from the snapshot. Tested on
  flat and HNSW indexes: with `precision='exact'` the result equals the full
  scan of the live rows, before and after the next refresh.
- An incremental refresh folds the delta into the snapshot: it reads the same
  rows the delta view shows (the journal after the previous boundary), so the
  refresh costs the reading of the delta plus the writing of the snapshot, not
  a pass over the whole table.
- If a node answers with `snapshot cache stale on <node>: run vload`, its
  cache is older than the view: `CALL vvector.load_all('<index>')`.
- Grant SELECT on `<index>_snap` and `<index>_delta` to the users who search.
  The delta view shows the journal rows themselves.

## Operations

- **Multi-node**: every refresh loads the snapshot on every node (`vload`
  through `vvector.probe`); every node answers from its own cache file.
  Tested on one node, on an Eon cluster (3 nodes and a secondary subcluster
  of 2) and on a 4-node Enterprise cluster. A search runs on the node
  that receives the statement; the index is not split over nodes.
- **Eon subclusters**: a statement runs on the nodes of the subcluster its
  session is connected to, never on another. So every subcluster keeps node
  caches of its own: a `refresh_index` builds the snapshot, stores it in
  `vvector.snapshot` (communal storage, visible to every subcluster) and loads
  the nodes of the subcluster it runs in. A scheduled refresh runs on
  Vertica's scheduler node, a primary node, whatever subcluster created the
  schedule, so it loads the primary subcluster. Every other subcluster that
  searches the index loads it with

      CALL vvector.load_all('docs');    -- in a session on that subcluster, after each refresh
      CALL vvector.load_all();          -- or every index at once: one cron line per subcluster

  from the application after a refresh, or from a cron job with vsql every
  minute: `load_all` first asks every node of the subcluster whether it has the
  active snapshot (one call, milliseconds) and loads only when one has not, so
  the repeated call costs nothing between refreshes. `refresh_index` prints a
  reminder when the database has more than one subcluster; `status` says how
  many nodes of the session's subcluster hold the active snapshot; `vinfo`
  lists the nodes of the session's subcluster. Nothing goes wrong without the
  load: `vsearch` on a subcluster that is behind answers
  `snapshot cache stale on <node>: run vload`, and with no cache at all
  `no snapshot cache for index ... run vload`. `vknn` has no stale check and
  answers from the snapshot that subcluster holds. A refresh can also run from
  a session on a secondary subcluster (it needs the base snapshot in that
  subcluster's cache for an incremental build, else it builds in full); then
  the primary is the one that runs `load_all`. Tested with
  `tests/sql/test_subcluster.sh --secondary='vsql -h <a node of the other subcluster> -X -A -t -q'`.
- **Journal replica on a cluster**: a statement over the `_delta` view
  reads the journal. With the journal segmented over the nodes, every node
  scans its part and sends the rows to the node that runs the search, even
  when no row qualifies: 15 to 17 ms per statement on the 3-node test
  cluster. vvector therefore keeps a replica of the journal: a projection
  `<schema>.<index>_journal_rep` of the id, vector, delete and version
  columns, sorted by the version column, `UNSEGMENTED ALL NODES`, with
  statistics on the version column. The planner then reads the delta on the
  node that runs the search.

  | Measured on the 3-node test cluster (100,000 x 128, mixed mode) | Without | With the replica |
  |---|---:|---:|
  | search over the `_delta` view, empty delta | 23.4 ms | 10.7 ms |
  | the same with 1000 journal rows | 31.3 ms | 14.6 ms |
  | storage per node | the node's share of the journal | a full copy of the journal (in Eon: in the depot of every node, one more copy in communal storage) |
  | bulk insert into the journal (20,000 rows) | 250 to 340 ms | 830 ms |
  | single-row insert and commit | about the same | about the same |
  | `refresh_index` | no difference | no difference |

  Because the cost grows with the journal, the default mode `auto` keeps the
  replica only where it is cheap: on a database with more than one node, when
  one copy of the journal takes at most 2048 MB and at most 10% of the
  smallest free disk space of a node. `register_index` and every
  `refresh_index` apply the rule: they make the replica (and refresh its
  statistics), or drop it when the journal has grown beyond the limit. On a
  single node there is nothing to gain and nothing is made. Indexes on the
  same journal columns share one replica; `unregister_index` drops it with
  the last of them. Change the mode per index:

      CALL vvector.set_journal_replica('docs', 'on');     -- always, whatever the size
      CALL vvector.set_journal_replica('docs', 'off');    -- never (drops it)
      CALL vvector.set_journal_replica('docs', 'auto');   -- the default rule

  `register_index`, `refresh_index`, `set_journal_replica` and `status` print
  what was done and why, for example
  `journal replica: created app.docs_journal_rep (journal 79 MB, one copy on each of 3 nodes)` or
  `journal replica: none: the journal takes 5120 MB, more than the limit of 2048 MB ...`.
  A new replica is not made while another transaction writes to the journal
  (Vertica's `CREATE PROJECTION` would wait for it to end): the message says
  "postponed" and the next refresh tries again, so a refresh never hangs
  behind a long load. Making the projection needs the right to create a
  projection on the journal table. Without it nothing fails: the message says "not created",
  gives the reason, and lists the three statements a DBA can run instead.
  vvector never drops a projection it did not make, and in `auto` mode it
  makes none when the table has an unsegmented projection already.
- **What each node has**:

      SELECT node_name, index_name, snapshot_id, vector_count, dims, metric, index_type, freshness_default, loaded
      FROM (SELECT vvector.vinfo(USING PARAMETERS index_name='docs') OVER(PARTITION NODES) FROM vvector.probe) i;

         node_name    | index_name | snapshot_id | vector_count | dims | metric | index_type | freshness_default | loaded
      ----------------+------------+-------------+--------------+------+--------+------------+-------------------+--------
       v_vdb_node0001 | docs       |         959 |            5 |    3 | cosine | hnsw       | exact             | t

  Without `index_name` it lists every index in the cache directory.
  Other columns: max_ver, quantization, graph_bytes, tombstones,
  base_snapshot, precision_default, ef_search_default, threads_default,
  cache_file (or why the cache cannot be read), resident_mb (how much of the
  cache file is in the node's memory now: what a query reads without going to
  disk; vinfo itself reads nothing ahead), capacity (the vectors the layout
  has room for: an incremental refresh appends into that room and sends only
  the changed bytes), file_bytes (the size of the cache file), unreachable
  (HNSW: the live vectors no search can reach, counted by the build; NULL
  when it did not count, see [Unreachable vectors](#unreachable-vectors)).
- **Cache directory**: `/tmp/vvector` by default. Per index (recommended):
  the option `cache_dir` of [set_index_options](#set_index_options); the
  procedures and every query then use it without further settings. Per
  session: `ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '/data/vvector';`
  (the refresh and every query must use the same one), or `cache_dir=` per
  call. The order: the function parameter, the session parameter, the index
  option, the default; inside the procedures the index option comes first.
  `/tmp` may be cleaned at reboot: that is safe (`load_all` restores it),
  but a node answers "run vload" until then. Use a directory on a data disk
  that belongs to the database's operating system user (vload creates it
  with mode 0700 if it is missing): a search reads only cache files and
  index directories of that user, and treats any other as "no cache" (any
  user who may search can set cache_dir, so a file someone else wrote is
  never read).
- **Memory**: a search reads the cache file through the page cache, and
  `vinfo` shows how much of it is in memory (`resident_mb`). Vertica's own
  scans go through the same cache, so a large scan can push index pages out;
  the next search then reads them from disk (on the test VM: one search of
  0.3 s for a 632 MB index, then normal speed). A new mapping asks the
  kernel to read ahead the ids, codes and graph before the float rows, at
  most 8 GB in all; the rest of a file that fits in the node's memory comes
  in as searches touch it, at the disk's read-around speed (on the 100M
  test the first search on a cold 63 GB HNSW file took 46 s, a cold batch of
  1000 queries about two minutes, and everything was fast after that); a
  file larger than the node's available memory is marked for random access
  beyond the 8 GB instead, so its searches read only the pages they need; a
  flat index reads all its rows at every search, so they are always read ahead;
  and `load_all` reads all of it. A search reads the file on one node only, in the tests the node the
  client was connected to: on the 4-node test cluster, after a benchmark
  whose sessions all used node 1, the indexes were fully resident there and
  100 to 200 MB of 630 MB on the other nodes. Spread
  the clients over the nodes you want warm, or expect the first searches on
  a cold node to read from disk.
  A small, hot index can be pinned in memory with `cache_dir` on a RAM
  file system, if it fits there with room to spare beside Vertica on every
  node: `/dev/shm` (no setup), or a tmpfs mounted with huge pages, which is
  also faster (a root step on every node, for example `mount -t tmpfs -o
  size=8G,huge=always,mode=0755 tmpfs /data/vvhot` and `chown` to the
  database user): on the test VM batches of 1000 searches took 22% less
  time and single searches 4 instead of 5 ms. Size it for the index files
  of every index placed there, plus one more snapshot during a refresh. The
  memory is taken for good, and after a reboot the cache is empty until
  `load_all`. Large indexes belong in a `cache_dir` on a disk: the page
  cache keeps what the searches use.
  A refresh whose build needs more than half of the smallest node's free
  memory (with the page cache; for the first build of an index estimated from
  the journal rows and the vector length) builds in a file in the index's cache
  directory instead (on the test VM 7% slower for flat, 13% for HNSW), and says so in its note; the kernel
  can then write the build out instead of running out of memory. Such a
  build sorts the rows into a second file, so the cache directory needs room
  for two more copies of the vectors during the build, and it keeps the HNSW graph
  in memory (the graph part of `vvector.sizing`), because graph links are
  written in random order and random writes to a file run at the speed of
  the disk. It does
  not get around `FencedUDxMemoryLimitMB`: Vertica applies it as the address
  space limit of the fenced process, which counts a file mapping too.
- **Backup**: the snapshots are rows of `vvector.snapshot` and the options
  are rows of `vvector.manifest`: a backup of the database contains them.
- **Disk space**: a refresh keeps the active chain (the last whole copy and
  the patches after it; a full build starts a new chain) in the
  table and in every cache. The table is segmented: a snapshot is stored once
  across the cluster (with K-safety 1, twice), compressed; every node's cache
  holds the whole snapshot. Every refresh that changes the index writes a
  whole new snapshot, also an incremental one (631 MB for 1M vectors of 128
  dimensions with HNSW; Vertica compresses it to about half). The older one is
  deleted, and its rows keep using space until Vertica's Tuple Mover purges
  them; on the test VM that happened by itself within minutes (100 refreshes
  in 15 minutes never used more than 7 GB above the start). To free it at once:
  `SELECT PURGE_TABLE('vvector.snapshot');`.
- **Library version**: `SELECT vvector.vversion() OVER();`

       library_version | format_version |                           build_flags
      -----------------+----------------+------------------------------------------------------------------
       0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 aarch64 g++-11.5.0 kernels=neon

  A new library that reads another snapshot format refuses the old cache
  files with "format version 1, this library reads version 2: refresh the
  index": run `refresh_index` for every index after such an upgrade.
  Upgrading from a version that had `vbuild`, `vload`, `vconfig` and `vnode`
  in schema `vvector`: `make deploy` drops the library with its functions and
  creates them again (Vertica cannot drop a single function with an ARRAY
  argument); searches running at that moment fail. The first refresh of every
  index after that upgrade is a full build (the digest is new).
  Upgrading from a version from before 2026-09-25, where `vvector.snapshot` was
  `UNSEGMENTED ALL NODES`: `make deploy` copies its rows once into a segmented
  table of the same name (a table cannot be resegmented in place). Deploy when
  no refresh runs; the copy takes about as long as writing the stored
  snapshots once (2.3 GB: a few seconds on the test VM). The copy does not
  touch the caches on the nodes, which stay valid.
  Upgrading from a version from before 2026-09-25, where searching was open to
  PUBLIC: `make deploy` moves the search functions to the role
  `vvector_search` and closes the manifest to PUBLIC. Grant the role to the
  users who search, or deploy with `make deploy SEARCH=public` to keep the
  old behaviour.
- **Monitoring**: the refresh labels its statements `vvector_verify` (count
  and digest of the journal recomputed), `vvector_digest` (carried forward from
  the new rows, or taken for a full build), `vvector_build` and `vvector_load`:
  `SELECT request_label, request_duration_ms FROM v_monitor.query_requests WHERE request_label LIKE 'vvector%' ORDER BY start_timestamp DESC;`
  Put your own `/*+LABEL(name)*/` in searches to find them the same way.

### Scripts

The same from the shell:

    scripts/register.sh --index=docs --table=app.docs --id=id --vec=vec --op=del --ver=ts --metric=cosine
    scripts/refresh.sh --index=docs                        # refresh_index
    scripts/refresh.sh --index=docs --mode=full            # refresh_index('docs', 'full')
    scripts/refresh.sh --index=docs --schedule='0 * * * *' # schedule_refresh
    scripts/refresh.sh --index=docs --status               # status
    scripts/refresh.sh --index=docs --load_only            # load_all
    scripts/refresh.sh --load_only                         # load_all() for every index

`scripts/demo.sh` walks through everything on a table of its own (schema
VVDEMO, removed at the end unless `--keep`): it loads generated vectors
(100,000 x 128 by default) or SIFT1M (`--dir=<directory with sift_base.fvecs>`),
registers and builds an HNSW index, searches one query with vvector and with
the built-in full scan, measures the recall of the three precision levels,
adds and deletes a vector without a refresh and shows the difference between
`freshness='snapshot'` and `'exact'`, refreshes incrementally and shows every
node's cache. On the test VM with SIFT1M: one query 17 ms against 6.2 s for
the full scan, recall@10 0.93 / 0.99 / 0.999 (fast / balanced / best).

### The manifest

`SELECT * FROM vvector.manifest;` shows one row per index: the source
(`source_table`, `id_col`, `vec_col`, `op_col`, `ver_col`, `ver_margin`,
`metric`), the options of `set_index_options`, and the state of the active
snapshot (`active_snapshot`, `active_max_ver`, `delta_from` = the boundary of
the delta view, `vector_count` (live vectors), `tombstones`, `base_snapshot`
(the snapshot an incremental build started from, 0 after a full build),
`dims`, `graph_bytes` (the HNSW graph), `index_bytes` (the whole snapshot),
`built_at`, `build_seconds`, `format_version`, `active_options` (the build
options of the active snapshot), `incremental_count` (incremental refreshes
since the last full build), `boundary_rows` and `boundary_digest` (the number
and the digest of the journal rows up to the boundary, carried forward and
verified), `verify_every`, `refreshes_since_verify` (refreshes since the last
verification or full build), `refresh_note` (what the last refresh did and why),
`refresh_started_at` and `refresh_started_by` (set while a refresh runs),
`snapshot_chain` (the snapshot ids `vvector.snapshot` holds: the last whole
copy and the patches after it, up to the active one), `chain_bytes` (the
patch bytes since that whole copy), `sent_bytes` and `transfer` (what the
last refresh stored and sent: `whole` or `patch`), `capacity` (the vectors
the active snapshot's layout has room for), `reachability` and `unreachable`
(the option, and the vectors of the active graph no search can reach; NULL
when not counted)), and
the journal replica (`journal_replica` = auto, on or off;
`replica_projection`, the projection vvector made; `replica_note`, what the
last check did and why).

    SELECT index_name, source_table, metric, index_type, active_snapshot, vector_count, dims, graph_bytes, index_bytes, build_seconds
    FROM vvector.manifest WHERE index_name = 'docs';

     index_name | source_table | metric | index_type | active_snapshot | vector_count | dims | graph_bytes | index_bytes | build_seconds
    ------------+--------------+--------+------------+-----------------+--------------+------+-------------+-------------+---------------
     docs       | app.docs     | cosine | hnsw       |             959 |            5 |    3 |         896 |        1536 |         0.282

### Troubleshooting

Every message starts with the function or procedure that raised it. `vknn`
raises the messages of `vsearch` (those about queries, parameters and the
cache), starting with `vknn:`.

| Message (shortened) | Cause | Fix |
|---|---|---|
| `vsearch: no snapshot cache for index 'x' in DIR: run vload` | the index was never refreshed, or this node's cache is missing, or the session uses another cache_dir, or (Eon) this subcluster was never loaded | `CALL vvector.load_all('x')` in a session on this subcluster; check cache_dir |
| `vsearch: snapshot cache stale on NODE: run vload` | this node missed the last vload (down during a refresh), or (Eon) the refresh ran on another subcluster | `CALL vvector.load_all('x')` in a session on this subcluster |
| `vsearch: snapshot cache of index 'x' is missing or damaged (...): run vload` | the cache file was deleted or changed | `CALL vvector.load_all('x')` |
| `... format version 1, this library reads version 2: refresh the index ...` | cache written by an older library | `CALL vvector.refresh_index('x')` |
| `vsearch: index 'x' has N dimensions, the query parameter has M` (or `query Q has M`, `the journal vector of id I has M`) | vectors of another length | use vectors of the index's length |
| `vsearch: no query: give the query parameter, or query rows (qid, qvec)` | the input has no query | add `query=` or query rows |
| `vsearch: query vector text: expected ',' or ']' at character N` (and similar) | malformed `query` text | write `'[1.5, 2, -3e-2]'` |
| `vsearch: query Q has no vector (qvec is NULL)`, `a query row has a vector (qvec) but no qid` | incomplete query row | set both qid and qvec |
| `vsearch: journal row with id I has no vector and is not a delete` | a journal row with a NULL vector and del false | fix the row, or set del |
| `vsearch: a journal row has no id (id is NULL but not its vector, delete flag or version)` | a hand-made input row with values but no id (the views leave such rows out) | set the id, or leave the row out |
| `vsearch: no snapshot cache for index 'x' in DIR: the directory is not owned by the database's operating system user` | the index directory under that cache_dir belongs to another operating system user; it is not trusted | use the index's own cache_dir; remove or `chown` the foreign directory |
| `vsearch: k must be 1 to 16384`, `precision must be ...`, `freshness must be ...`, `ef_search must be ...`, `oversampling must be 1 to 100`, `threads must be 0 (one per core) to 64`, `radius must be a finite number` | a parameter out of range | use a value from the parameter table |
| `vsearch: session parameter NAME = 'v' is not an integer` (or `index default ...`) | a bad session value or index default | `ALTER SESSION SET UDPARAMETER FOR vvector NAME = ...`, or `set_index_options` |
| `vsearch: the snapshot of index 'x' changed while the query ran: run it again` | a refresh changed the dimensions during the query | run the query again |
| `vbuild: id I appears twice` | a static index (no version column) with a repeated id (or the same id twice in the changes of a hand-made incremental build) | make the ids unique, or register a version column |
| `vbuild: index 'x': base snapshot S is not usable in the cache of NODE (...): refresh with mode full` | an incremental build found no valid active snapshot on the node that runs it (a refresh checks this first and builds in full; a hand-made build does not) | `CALL vvector.refresh_index('x', 'full')` |
| `vbuild: ... the base snapshot has metric M, not N`, `... is an HNSW index, the build is flat`, `... is a flat index, the build is hnsw`, `m is N, the base snapshot was built with m M`: `... a full build is needed` | a hand-made incremental build with other options than the base | build in full, or use the base's options |
| `vbuild: vector of id I has N elements, the index has M` | a change of another length than the index | one length per index |
| `vbuild: base_snapshot must be a snapshot id, or 0 for a full build` | a negative base_snapshot | a snapshot id from the manifest |
| `vbuild: the vector of id I is NULL (a delete needs del = true)`, `... has a NULL element`, `... element N is not a finite float32 value` | a missing vector, a NULL, NaN, Infinity or a value beyond +-3.4e38 | fix the row |
| `vbuild: vector of id I has N elements, the ones before have M` | vectors of different lengths | one length per index |
| `vbuild: vectors have N elements, at most 32768 are supported` | too many dimensions | reduce the dimensions |
| `vbuild: metric must be l2, cosine, dot or l1`, `index_type must be flat or hnsw`, `quantization ...`, `m must be 2 to 256`, `ef_construction must be 1 to 100000` | bad build parameter | see the parameter tables |
| `vbuild: ... too many vectors for an HNSW graph with m = N` | the upper levels of the graph would need more than 4,294,967,294 blocks | a larger `m`, or split the index |
| `vload: on NODE: bad snapshot graph: ... in FILE`, `vsearch: bad snapshot graph: ...` | the graph section of the snapshot or cache file is damaged (vload checks every link) | `CALL vvector.refresh_index('x')` |
| `vload: on NODE: bad snapshot sq8 section: ... in FILE`, `vsearch: bad snapshot sq8 section: ...` | the int8 codes of the snapshot or cache file are damaged (vload checks every row) | `CALL vvector.refresh_index('x', 'full')` |
| `vbuild: index 'x': the base snapshot has quantization sq8, not none: refresh with mode full` (or `none, not sq8`) | a hand-made incremental build with another quantization than the base (a refresh builds in full by itself when the option changed) | `CALL vvector.refresh_index('x', 'full')` |
| `vbuild: out of memory: cannot map N MB for the snapshot` | the node has too little memory for the build | `CALL vvector.sizing(...)`; a larger node or FencedUDxMemoryLimitMB |
| `vload: on NODE: pieces are missing or duplicated`, `bad snapshot: checksum mismatch`, `cannot write ...` | a damaged transfer or a full disk | check disk space of cache_dir; `load_all` |
| `vload`, `vinfo`, `vsearch`: `cache_dir '...' must be an absolute path`, `index name '...' is not valid` | bad cache_dir or index name | use `/path` and letters, digits, underscore |
| `vsearch: index 'x': OPTIONS file in the cache DIR: ...: run vvector.load_all` | the index defaults file of the node was changed by hand | `CALL vvector.load_all('x')` |
| `vvector.register_index: ...` (table, column, type, metric, margin, op_col needs ver_col, already registered) | a bad argument; the message names it | fix the argument |
| `vvector.register_index: index x: the views could not be made (...); the index is not registered` | the views `<schema>.x_delta` and `x_snap` could not be created: no CREATE on the schema of the table, or another object has the name | grant CREATE, or free the name; then register again |
| `vvector.refresh_index: index x: table T has no vectors, nothing to build` | the table has no live rows | insert rows first |
| `vvector.refresh_index: index x: no live vector up to the delta boundary B ...; N rows of T are newer` | every live row was written after the boundary (within the margin before the refresh, or after the start of an open writer), for example the first refresh right after a load | refresh again when the margin has passed; the rows are found meanwhile with `freshness='exact'`. On one node with `CLOCK_TIMESTAMP()` versions a margin of 0 is safe |
| `vvector.refresh_index: mode must be auto, incremental or full` | a bad second argument | `'auto'`, `'incremental'` or `'full'` |
| `vvector.refresh_index: index x is being refreshed since T UTC (by USER, session S). Two refreshes of one index cannot run at the same time ...` | another refresh of the index runs (by hand or by the schedule), or one was killed and left its mark (a superuser, or the same user, gets past a killed one at once) | wait until it ends; if none runs, `UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = 'x'; COMMIT;` (or wait for the hours the message names) |
| `Function vvector.vsearch(...) does not exist, or permission is denied for vvector.vsearch(...)` (or vknn, vinfo, a vector function) | the user lacks the role `vvector_search`, or has it but not enabled | `GRANT vvector_search TO someone; ALTER USER someone DEFAULT ROLE vvector_search;` (or `make deploy SEARCH=public`) |
| `Permission denied for schema vvector_admin` | a build or load function called without the role `vvector_admin` | `GRANT vvector_admin TO someone;` and enable it (default role or `SET ROLE`) |
| `Function vvector.refresh_index(unknown) does not exist, or permission is denied ...` (any procedure) | the caller lacks the role `vvector_admin`, or has it but not enabled in the session | grant it and enable it (default role or `SET ROLE vvector_admin`) |
| `Only a Super User can drop triggers` (from `schedule_refresh`) | `schedule_refresh` was called by a user who is not a superuser | a superuser runs `schedule_refresh` |
| `vvector.unregister_index: index x has a refresh schedule; only a superuser can remove it ...` | the index has a schedule and the caller is not a superuser; nothing was removed | a superuser runs `unregister_index` |
| `vvector.load_on_nodes: index x, snapshot S: loaded on N of M nodes` | a node could not load (disk, rights) | vinfo shows the cause per node; `load_all` |
| `vvector.push_options: index x: options written on N of M nodes` | a node could not write its cache directory | check the cache directory; `load_all` |
| `vvector.<procedure>: index x is not registered` | a wrong index name | `SELECT index_name FROM vvector.manifest` |
| `vvector.set_index_options: memory_mode compact needs quantization sq8` | `compact` on an index without sq8, or `quantization none` while `memory_mode` is `compact` | set sq8 first, or `memory_mode ram` in the same call |
| `vvector.set_index_options: ...` | a value out of range | see the option table |
| `vvector.schedule_refresh: cron_expr may hold digits, spaces and * / , - only` | a bad cron expression | e.g. `'0 * * * *'` |
| `vvector.set_journal_replica: mode must be auto, on or off` | a bad mode | `'auto'`, `'on'` or `'off'` |
| `vector_add: the vectors have different lengths: N and M elements` (any vector function, `vector_avg`, `vector_sum`) | two vectors of different lengths | vectors of one length |
| `vector_l1: element I of the second vector is NULL` (any vector function; `vector_avg: element I of a vector is NULL`) | a NULL inside an array | replace the NULL, or filter the row |
| `Function vvector.vector_hamming(array[numeric], array[numeric]) does not exist` (or `array[float]`) | Hamming and Jaccard take ARRAY[INT] (bits) | cast to `ARRAY[INT]` |
| `ERROR 2521: Cannot specify anything other than user defined transforms and partitioning expressions in the SELECT list` | `vector_sum` and `vector_avg` are transform functions: only PARTITION BY columns may stand beside them | put other expressions in an outer query (see [Vector functions](#vector-functions)) |
| `journal replica: not created: ... A DBA can run: CREATE PROJECTION ...` (a NOTICE) | the caller may not create a projection on the journal table, or the name is taken | run the three statements as the table owner, or ignore it (searches work, only the cluster delta read stays slower) |

Warnings of `status` and `sizing` are explained in their text.

## Performance and results

Measured on the test VM: Vertica 26.2.0-1, one node, aarch64, 8 cores, 34 GB,
g++ 11.5 (2026-09-24). Data: SIFT1M (1,000,000 vectors of 128
dimensions, 10,000 queries with ground truth, TEXMEX corpus), metric l2,
k = 10. HNSW with m = 16, ef_construction = 200. Three numbers, reported
separately. Repeated on 2026-09-25 (the engine tests on SIFT1M and the
definition-of-done benchmark): the same within the noise.

**Engine alone** (`make bench DATA_DIR=...`, no Vertica, all 10,000 queries):

| Measurement | Flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| build of the snapshot (8 threads) | 0.09 s | 35 s | 35 s (the codes: 0.08 s) |
| snapshot size | 496 MB | 631 MB | 757 MB |
| 1 query, 1 thread | 9.2 ms | 0.14 ms (balanced) | 0.09 ms (balanced) |
| 1 query, 8 threads | 2.3 ms (the machine's full memory bandwidth) | (one thread per query) | (one thread per query) |
| queries per second, 8 threads, balanced | 950 (batch of 1000; exact) | 48,000 | 72,000 |
| recall@10 fast / balanced / best | 0.999 (exact; the rest are ties) | 0.903 / 0.983 / 0.999 | 0.895 (no rescoring) / 0.983 / 0.999 |

HNSW against hnswlib (v0.10.0-rc.2, the reference implementation), same
machine, same data and parameters (`make bench HNSWLIB_DIR=...`):

| ef_search | recall@10 vvector | recall@10 hnswlib | queries/s, 1 thread, vvector / with sq8 | hnswlib | queries/s, 8 threads, vvector / with sq8 | hnswlib |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.903 | 0.904 | 19,188 / 28,973 | 18,019 | 125,119 / 179,532 | 106,828 |
| 100 | 0.983 | 0.983 | 7,839 / 11,242 | 7,151 | 48,326 / 72,396 | 43,060 |
| 400 | 0.999 | 0.999 | 2,281 / 3,384 | 2,176 | 14,137 / 20,842 | 13,221 |

Equal recall; the float index 5 to 17% more queries per second than
hnswlib, with sq8 (2 x k rescored) 55 to 68% more; build 35 s against 40 s.

**One statement** (`scripts/latency.sh`, median at the client, 200 runs, `query`
parameter):

| Statement | Fenced | Mixed (or unfenced) |
|---|---:|---:|
| `SELECT 1` (the floor of any statement) | 0.8 ms | 0.8 ms |
| HNSW (precision balanced): vsearch `FROM sift_hnsw_snap` | 7.6 ms | 1.8 ms |
| HNSW: the same over `sift_hnsw_delta`, `freshness='exact'`, empty delta | 8.6 ms | 2.9 ms |
| HNSW: `vknn ... FROM dual` | 8.0 ms | 1.5 ms |
| HNSW with sq8 (balanced): vsearch `FROM sift_sq8_snap` / `vknn` | 7.5 ms / 7.9 ms | 1.8 ms / 1.5 ms |
| flat: vsearch `FROM sift_snap` | 11.0 ms | 5.5 ms |
| flat: the same over `sift_delta`, empty delta | 12.2 ms | 6.7 ms |
| SQL full scan (`ORDER BY VECTOR_L2(vec, q) LIMIT 10`) | 7797 ms | |

A single search spends about 0.1 ms in the engine; the rest is the
statement, so sq8 changes little for single searches and much for batches.

**Throughput, recall and refresh** (one statement with 1000 queries):

| Measurement | Fenced | Mixed |
|---|---:|---:|
| vsearch HNSW, precision fast | 24 ms (42,000 queries/s) | 14 ms (71,000 queries/s) |
| vsearch HNSW, precision balanced | 38 ms (26,000 queries/s) | 28 ms (36,000 queries/s) |
| vsearch HNSW with sq8, precision fast (codes only) | 20 ms (50,000 queries/s) | 11 ms (91,000 queries/s) |
| vsearch HNSW with sq8, precision balanced | 32 ms (31,000 queries/s) | 20 ms (50,000 queries/s) |
| vknn HNSW, precision balanced, 1000 rows | 158 ms (6,300 queries/s) | 149 ms (6,700 queries/s) |
| vsearch flat | 1108 ms (900 queries/s) | 1108 ms (900 queries/s) |
| recall@10 against the ground truth: HNSW fast / balanced / best / exact; flat | 0.892 / 0.980 / 0.999 / 0.999; 0.999 | |
| the same with sq8: fast / balanced / best | 0.884 / 0.979 / 0.999 | |
| `refresh_index`, 1M vectors, full build: HNSW / HNSW with sq8 / flat | 47 s / 49 s / 12 s | |
| `refresh_index` after 1000 adds and 500 deletes, 900,000 vectors, incremental: HNSW / flat | 6.8 s / 4.9 s | |
| `refresh_index` with nothing changed (verify_every 1 / 0) | 1.0 s / 0.6 s | |

On the 4-node Enterprise test cluster (x86_64 with AVX-512, 10 cores and 78 GB
per node, Vertica 26.2.0-3, g++ 8.5; SIFT1M as above, loaded on all 4 nodes) a
statement costs more, the engine is slower per core and hnswlib and vvector
are equal there: `SELECT 1` 3.3 to 3.5 ms; HNSW `_snap` 13.8 ms fenced and 6.6
ms mixed (13.3 and 6.4 ms in an earlier run), with sq8 14.4 and 6.1 ms; `vknn`
14.2 and 6.0 ms; the first search of a new session 42 ms fenced and 13 ms
mixed; an ARRAY literal instead of the `query` parameter costs 23 ms more
there. One statement with
1000 queries at precision balanced: 49 ms mixed, 34 ms with sq8. Engine, 10
threads, ef_search 100: 35,400 queries/s (hnswlib 34,700), with sq8 59,600.
Recall through SQL as on the VM. A full refresh of 1M x 128 HNSW takes 80 s
(87 s with sq8), an incremental one of 900,000 vectors after 1000 adds and 500
deletes 12.7 s when every node still loaded the whole snapshot,
and 15 s (flat: 7 s) now that only the changed bytes travel: at this
size the fixed part of a refresh (4 to 5 s here) and the verification of the
patched file weigh as much as the whole load did; the gain shows from 10M on
(docs/design.md, "Incremental transfer"). A range
search (k 16384, the radius of the query's 10th neighbour) costs 15.6 ms
fenced and 7.1 ms mixed; filtered searches: see
[Filtered search](#filtered-search).

**10 million vectors** (the first 10M of BIGANN / SIFT1B, 128 dimensions, with
its ground truth for 10M; the 4-node cluster, 1000 queries;
fenced build in memory, cache files on a second data disk of each node
through the index option `cache_dir`):

| Measurement | Flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 4.8 GB | 6.2 GB | 7.4 GB |
| full build (`refresh_index`; vbuild / vload) | 132 s (65 / 55) | 899 s (828 / 65) | 944 s (843 / 94) |
| incremental refresh, 1000 adds and 500 deletes, when the whole snapshot was sent | 99 s | 144 s | 157 s |
| the same now (only the changed bytes travel; vbuild / vload) | 18.8 s (5.7 / 7.3) | 47.1 s (5.2 / 35.4) | 47.5 s (7.0 / 34.0) |
| recall@10 fast / balanced / best | 1.000 (exact) | 0.826 / 0.953 / 0.995 | 0.818 / 0.952 / 0.995 |
| one search, fenced / mixed (client ms) | 92 / 87 ms | 14.4 / 7.0 ms | 15.1 / 6.3 ms |
| 1000 queries in one statement, balanced, fenced / mixed | 17.7 / 17.6 s | 154 / 57 ms | 138 / 52 ms |

At 10M a larger `ef_search` keeps the recall of 1M: 200 gives 0.983 (1000
queries in 0.4 s). Before, an incremental refresh stored and loaded
the whole snapshot on every node (that disk writes 100 MB/s); now a
refresh sends under 1 MB (flat) or 8 MB (HNSW) of changed bytes, and what
remains of the HNSW time is the copy of the 6.7 GB base file on every node
plus its verification (docs/design.md, "Incremental transfer").
An exact search over the delta view read the 10M-row journal on every node
(33 ms instead of 7 ms): its rows were all loaded the same day, so
partitioning by date could not skip any, and at 2.7 GB it is above the size
the journal replica is made for; a journal partitioned by day reads only the
recent partitions.

**768 and 1536 dimensions** (1M generated vectors of Gaussian clusters, the
size of text embeddings, metric cosine, recall against the exact search; the
4-node cluster, same build and cache setup as the 10M test):

| Measurement | 768: HNSW | 768: HNSW with sq8 | 1536: HNSW | 1536: HNSW with sq8 |
|---|---:|---:|---:|---:|
| cache file per node (flat: 2.9 GB / 5.7 GB) | 3.0 GB | 3.7 GB | 5.9 GB | 7.3 GB |
| full build (`refresh_index`) | 197 s | 218 s | 360 s | 384 s |
| incremental refresh, 1000 adds and 500 deletes | 82 s | 89 s | 130 s | 156 s |
| recall@10 fast / balanced / best | 0.808 / 0.971 / 0.998 | 0.694 / 0.933 / 0.992 | 0.802 / 0.960 / 0.999 | 0.706 / 0.931 / 0.996 |
| one search, fenced / mixed (client ms) | 21.7 / 13.2 ms | 22.8 / 12.8 ms | 30.4 / 19.9 ms | 30.3 / 19.7 ms |
| 1000 queries, balanced, fenced / mixed | 159 / 110 ms | 111 / 61 ms | 254 / 167 ms | 173 / 103 ms |

A flat index answers one search in 76 ms (768) and 112 ms (1536) mixed: it
reads the whole file per query. sq8 halves batch times but loses more recall
here than on SIFT (see [int8 quantisation](#int8-quantisation-sq8)). The sq8
balanced column was measured with 2 x k rescored; with the preset of 4 x k
for 512 dimensions and more, balanced gives 0.965 (768) and 0.956 (1536), and
1000 queries take 115 and 180 ms fenced (docs/design.md).

**100 million vectors** (the first 100M of BIGANN / SIFT1B, 128 dimensions,
with its ground truth for 100M; the 4-node cluster, 78 GB per node, 1000
queries; one index built at a time, the build in a file, caches on the second
data disk):

| Measurement | Flat | HNSW | HNSW with sq8 |
|---|---:|---:|---:|
| snapshot, cache file per node | 48 GB | 62 GB | 74 GB |
| full build (`refresh_index`; vbuild / vload) | 1,628 s (1,162 / 109) | 12,872 s (12,056 / 178) | 13,641 s (12,586 / 260) |
| incremental refresh, 1000 adds and 500 deletes: whole snapshot sent / only the changed bytes | 1,156 s / 146 s | 1,739 s / 301 s | 2,766 s / 390 s |
| recall@10 fast / balanced / best | 1.000 (exact) | 0.748 / 0.903 / 0.981 | 0.741 / 0.903 / 0.981 |
| one search, fenced / mixed (client ms) | 779 / 786 ms | 13.6 / 7.6 ms | 15.5 / 6.6 ms |
| 1000 queries in one statement, balanced, fenced / mixed | 175 / 176 s | 822 / 452 ms | 1,143 / 918 ms |

A search costs the same as at 10M once the file is in memory (4.7 ms of server
time unfenced). Recall at a given `ef_search` falls with the size: balanced
(ef 100) gives 0.903 here against 0.953 at 10M; use a larger `ef_search`
(the index default `ef_search_default`) for a 100M index, `precision='best'`
gives 0.981. sq8 is slower than the float index at this size and gains no
recall: it saves nothing at 128 dimensions. The three files (184 GB) do not fit
one node's memory together; after another build had pushed the HNSW file out,
the first search took 45 s and the next ones 10 to 25 s while the file was
read back (docs/design.md, "100 million vectors"): a node needs the memory for
the indexes it serves beside Vertica. A refresh with nothing changed takes 18
to 82 s: the journal digest over 100M rows, from disk when the table is not in
the page cache (`verify_every`).

**A journal of one billion rows** (10M ids written 100 times, 8 numbers per
vector, partitioned by day; flat index; the 4-node cluster): full build 104 s
(the latest row of each id out of 1B rows), delta read after one day of
changes 75 ms, an exact search through it 274 ms, incremental refresh of that
day 31 s with the journal verified (15 s of it the verification over 1B rows)
and 20.5 s without, a refresh with no change 1.6 s unverified and 16 s
verified. On a large journal set `verify_every` to N so the full check runs at
every Nth refresh. Details: docs/design.md.

On the 3-node Eon test cluster (x86_64, 2 cores and 15 GB per node, Vertica
26.2.0-2; HNSW index of 100,000 random vectors of 128 dimensions) a statement
costs more: `SELECT 1` 2.7 ms, vsearch `_snap` 12.8 ms fenced and 7.6 ms mixed,
`vknn` 12.1 and 5.9 ms (precision fast). An exact search over the empty delta
takes 10.7 ms mixed with the journal replica, which is made there by default,
and 23.4 ms without it (see [Operations](#operations)). An incremental
refresh there of 900,000 vectors of 128 dimensions after 1000 adds and 500
deletes takes 6.4 s (flat) and 11.5 s (HNSW) since only the changed bytes of
the snapshot travel (19.6 s and 25.3 s when the whole snapshot did; 100
changes 5.9 and 8.3 s, 10,000 changes 6.7 and 12.4 s; docs/design.md
"Incremental transfer"). `vscan` over 1M vectors of 128 dimensions there
takes 0.6 to 0.7 s for one query and 0.6 s for ten queries in one statement,
against 19 s and 25 s for the built-in full scan; `vsearch` with precision
exact on a flat index of the same rows 80 ms and 165 ms (the same ids from
all three). On the 4-node cluster (10 cores per node) `vscan` takes 0.5 to
0.6 s at 1M rows, 3.6 s at 10M and 19 s at 100M for one query, against 14
to 20 s, 37 to 80 s and 364 s for the built-in scan, and ten queries in one
statement cost the same scan (0.6 s, 3.7 s, 20 s); `vsearch` with precision
exact on the flat index 32 ms, 132 ms and 1.25 s for one query
(docs/design.md, "Exact search without an index").

**Filtered and range search** (the VM, SIFT1M, one query, median at the
client): with an allow-list of 100 ids 8.6 ms fenced / 3.0 ms mixed, 10,000
ids 12.5 / 5.3 ms, 100,000 ids 29.8 / 13.9 ms (against 7.4 / 1.9 ms without a
filter); most of the added time is Vertica passing the allow-list rows.
Engine alone, 1000 queries: a filter of 1% of the vectors 75,700 queries/s
(exact), 50% 24,400 queries/s (recall 0.99). A range search with k 16384 and
a radius that holds about 10 vectors per query costs what a plain search
costs (1.9 ms mixed); in the engine it answers 2,674 queries/s against 485
with a candidate list of k. Details: docs/design.md.

The first search of a session costs more when vsearch is fenced, because the
session starts its own fenced process and maps the index: 13.3 ms instead of
7.6 ms (HNSW, 1M vectors); in mixed mode 2.4 ms instead of 2.1 ms. Keep
sessions open (a connection pool) for single searches.

Fenced mode adds about 6 ms per statement, more than the search itself; for
single searches `FENCED=mixed` gives the unfenced latency while the memory-
heavy build stays fenced. For batches the difference is smaller. Where each
millisecond goes is in [docs/design.md](docs/design.md).

To reproduce (loads SIFT1M into schema VVBENCH; about 40 minutes with the
hnswlib comparison):

    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
    make && make tools && make deploy
    scripts/benchmark.sh --data_dir=$PWD/sift [--hnswlib=<a clone of github.com/nmslib/hnswlib>]

Without `--data_dir` it generates 1M vectors of 128 dimensions and measures
recall against the exact search. Besides the tables above it measures
filtered search (allow-lists of 100 to 100,000 ids from an unsegmented and a
segmented table), range search, and how much of each cache file is in memory
on every node (`vinfo` `resident_mb`); `--parts` and `--modes` choose what to
run, `--help` lists everything.

**Scale tests.** `scripts/scale.sh` measures one data set end to end: loading,
full builds of a flat, an HNSW and an HNSW index with sq8 (`vvector.sizing`
first), vinfo on every node, recall@10 at every precision level, single
searches and batches fenced and mixed, and an incremental refresh. It is not
part of `make test`; `--help` lists the options. Examples:

    # the first 10M vectors of BIGANN (bigann_base.bvecs, bigann_query.bvecs and gnd/ of the TEXMEX corpus)
    scripts/scale.sh --dataset=bigann --dir=<dir> --rows=10000000 --gt=<dir>/gnd/idx_10M.ivecs
    # generated vectors (Gaussian clusters) of 768 numbers: for memory and speed; recall against exact search
    scripts/scale.sh --dataset=g768 --generate=768 --rows=1000000

Loading alone: `scripts/load_dataset.sh` takes the same `--dir`, `--rows`,
`--gt`, `--generate` and `--streams` options (parallel COPY streams, one per
two cores by default).

## Restrictions and not supported

Vertica and SDK:
- Vertica 26.2 has no VECTOR type and no vector index. vvector does not
  change `ORDER BY ... LIMIT`: searches are written with `vvector.vsearch`.
- The default array bound is 65000 bytes (8125 FLOAT elements); longer
  vectors need an explicit bound on the column (`ARRAY[FLOAT, 16384]`) and
  are untested.
- A transform function is not called on empty input: the views carry a
  sentinel row for that reason.
- Rights are granted per schema (Vertica cannot grant a single function with
  an ARRAY argument): the search functions in `vvector` need the role
  `vvector_search`, the build and load functions in `vvector_admin` the role
  `vvector_admin`.
- Searching cannot be limited per index: whoever has `vvector_search` can
  search every index by name (`vknn`, `vsearch ... FROM dual`), without a
  view. The views protect the journal rows, not the index.
- `schedule_refresh` needs a superuser (Vertica: only a superuser may create
  a trigger), and so does `unregister_index` of an index with a schedule
  (it drops the trigger).
- A UNION ALL of `ARRAY[INT]` and `ARRAY[NUMERIC]` columns fails inside
  Vertica 26.2 (INTERNAL 5445); cast to `ARRAY[FLOAT]` first.
- Fenced mode adds about 6 ms per statement, and about 6 ms more to the first
  search of every session; unfenced functions run inside the Vertica process,
  where a fault stops the node.
- On a multi-node cluster a statement over the `_delta` view without the
  journal replica reads the journal on every node and sends the rows to the
  node that runs the search: 15 to 17 ms more per statement on the 3-node test
  cluster. The replica (automatic up to 2048 MB of journal, see
  [Operations](#operations)) costs a full copy of the journal on every node
  and makes bulk loads into the journal about 2.5 times slower.
  Snapshot-only statements (`_snap` view, `FROM dual`, `vknn`) do not read
  the journal at all.
- An `ARRAY[...]` literal of many numbers is slow to parse (about 7 ms for
  128): use the `query` parameter.

Data model:
- Ids are INT and unique per index; every vector of an index has the same
  number of elements, at most 32768; NULL elements, NaN, Infinity and values
  beyond +-3.4e38 are refused.
- The table is a journal: adds and deletes are INSERTs. A physical DELETE or
  UPDATE (or a dropped partition) becomes visible at the next refresh, which
  notices any physical change to rows up to its boundary (count and digest of
  those rows) and rebuilds in full because of it. With `verify_every` N it
  notices it within N refreshes; with 0 not at all: run
  `refresh_index(name, 'full')` after such a change. The verification reads
  every journal row up to the boundary.
- Without a version column the index is static and every id must appear once.
- The version column must be filled by the database (`DEFAULT
  CLOCK_TIMESTAMP()`); versions set by an application can make results wrong
  (`status` warns about versions in the future). Plain TIMESTAMP versions need
  the same time zone in every session; use TIMESTAMPTZ.
- Two rows of one id with the same version: a delete wins; between two adds
  either may win.
- The metric is fixed at registration; changing it means unregister and
  register again.

Index and search:
- HNSW is approximate: recall depends on `precision` / `ef_search`, `m`,
  `ef_construction` and the data (0.89 / 0.98 / 0.999 for fast / balanced /
  best on SIFT1M). `precision='exact'` or a flat index gives exact answers at
  the cost of reading every vector: about 2 to 3 ms per million vectors of 128
  dimensions on 8 cores.
- Range search on an HNSW index is approximate like every graph search: a
  vector inside the radius can be missed (use `exact=true` for a complete
  answer). At most k rows (k up to 16384) are returned.
- Filtered search: the allow-list is sent as rows with every statement (one
  row per allowed id); vvector cannot read a filter column itself. Above the
  exact-search limit the graph walk passes through the vectors outside the
  list; with a very selective filter just above that limit the walk is slow
  (it visits about ef_search x vectors / allowed nodes). `vknn` has no
  allow-list. On a multi-node cluster, allow-list rows from a segmented
  table are gathered from every node: about 24 ms more per statement on the
  4-node test cluster; read them from an `UNSEGMENTED ALL NODES` table
  ([Filtered search](#filtered-search)).
- `vector_sum` and `vector_avg` are transform functions, not aggregates
  (Vertica 26.2 aggregates cannot take an ARRAY argument): use them with
  `OVER()` or `OVER(PARTITION BY ...)` and only partition columns beside
  them. Hamming and Jaccard work on bits of ARRAY[INT] (0/1 elements or
  packed 64-bit words), not on sets of values.
- `vknn` searches the snapshot only: no journal and no stale check (a node
  that missed a refresh answers from its old snapshot until `load_all`).
- `register_index` creates an HNSW index by default (since 2026-09-23; it
  was flat before). Existing indexes keep their type.
- At most 4,294,967,295 vectors per index; one snapshot per index, cached
  whole on every node: an index larger than one node's memory works from disk,
  slowly.
- k is at most 16384; a radius search returns at most k rows.
- Journal rows since the last refresh are searched at every exact query; a
  large delta (100,000 rows and more) slows every such query: refresh more
  often.
- Scores are 32-bit floats: they agree with the FLOAT built-ins to about
  1e-6 relative; nearly equal scores can be ranked differently than the
  built-ins. Cosine with a zero vector gives score 0, as the built-in.
- int8 quantisation (sq8) is lossy. With rescoring (the default of balanced
  and best) the returned scores are exact, but a true neighbour whose codes
  rank it outside the k x oversampling candidates is missed; with
  `rescore=false` (and `precision='fast'`) the scores are approximate. One
  code range holds for the whole index; it is trained at a full build and
  kept by incremental refreshes, so vectors added later with values outside
  it get clipped codes until the next full build. The codes come in
  addition to the float vectors (about a quarter more bytes for float
  data); `memory_mode compact` only changes what is read ahead, and it takes
  effect at the next refresh.
- Not supported: product quantisation, IVF, DiskANN, sparse vectors, several
  vectors per id, GPU, hybrid text and vector search, sharding one index over
  nodes, big-endian hosts, Windows.

Operations:
- A new index default is seen by queries within 200 ms. A new snapshot: at
  once by vsearch over a view (its snapshot_id makes the node look again),
  within 200 ms by vknn and by vsearch without a view.
- Two refreshes of one index at the same time are refused: the second one
  gets an error. A refresh whose session was killed leaves its mark in the
  manifest; the next refresh ignores it at once when it can see that the
  session is gone (a superuser, or the same user), else after 6 hours or 4
  times the last build time; the mark of a scheduled refresh counts by age
  only (or remove it by hand, see [refresh_index](#refresh_index)).
- Cache files stay on the nodes after `unregister_index`.
- A refresh builds on one node; its memory is the build memory above.
- A node that missed a refresh answers "snapshot cache stale ... run vload"
  until `load_all` runs.
- Eon: a refresh loads the subcluster it runs in (a scheduled one the
  primary subcluster); every other subcluster runs `load_all` (one index,
  or all without an argument) in a session of its own after each refresh,
  and `vknn` there has no stale check (see [Operations](#operations)).
- A new snapshot format needs a refresh of every index; the error says so.
- An incremental refresh reads only the changes and sends only the bytes that
  changed, but every node still reads the new snapshot file once (its
  verification, which is also the prewarming: a new file has no pages in
  memory), and the build compares the parts of the snapshot that can change
  (the graph, the id index) with the previous one. That read grows with the
  index: about 90 s for 66 GB from the cluster's disks, under a second at 1M.
  A full build of 1M x 128 takes 11 s as a flat index and 46 s as an HNSW
  index. When the room to grow of the layout is used up (5% of the vectors
  by default), or the patches since the last whole copy weigh more than the
  copy, one refresh sends the whole snapshot again, as a full build does.
- The first refresh after an upgrade from a version from before 2026-09-26
  sends the whole snapshot (the old layout has no room to grow).
- Tombstones (the old positions of changed and deleted vectors) stay in the
  snapshot until the next full build: they take memory, and an HNSW search
  passes through them. `refresh_mode auto` rebuilds in full at
  `tombstone_ratio`; with `refresh_mode incremental` schedule full refreshes
  yourself.
- A static index (no version column) is rebuilt in full at every refresh.
- The graph of a parallel build depends on the order in which threads insert:
  two builds of the same data give slightly different graphs (and recall);
  the results of a search on a given snapshot are always the same.

## Files

| File | What it does |
|---|---|
| `Makefile` | `make`, `make test`, `make bench`, `make tools`, `make deploy [FENCED=yes\|no\|mixed] [SEARCH=role\|public]`, `make undeploy` |
| `src/engine/` | pure C++17, no Vertica includes: snapshot format, incremental build (`delta.cpp`), node cache, distance kernels, flat search, HNSW (`hnsw.cpp`), int8 codes (`sq8.cpp`), the search with rescoring and filters (`search.cpp`), the arithmetic of the vector functions (`vecmath.cpp`), threads, query text |
| `src/udx/` | the Vertica adapters: one small file per SQL function (`vsearch.cpp`, `vknn.cpp`, `vscan.cpp`, `vbuild.cpp`, ...), and one per family of vector functions (`vector_functions.cpp`, `vector_aggregates.cpp`) |
| `sql/` | `install.sql`, `procedures.sql`, `uninstall.sql` |
| `scripts/` | `deploy.sh`, `register.sh`, `refresh.sh`, `load_dataset.sh`, `latency.sh`, `benchmark.sh`, `demo.sh`, `scale.sh` |
| `tools/fvecs.cpp` | converts `.fvecs`, `.ivecs`, `.bvecs` files (SIFT1M, BIGANN) to text for COPY, and generates random clustered vectors |
| `tests/engine/` | unit tests (`make test`, among them `test_hnsw.cpp`, `test_delta.cpp`, `test_sq8.cpp`, `test_filter.cpp` and `test_vecmath.cpp`) and the engine benchmarks (`make bench`: `bench_flat.cpp`, `bench_hnsw.cpp` (float and sq8), `bench_filter.cpp` (filtered and range search), and `bench_hnswlib.cpp` with `HNSWLIB_DIR=`) |
| `tests/sql/` | integration tests: `test_snapshot.sh`, `test_freshness.sh`, `test_search.sh`, `test_hnsw.sh`, `test_incremental.sh` (`--sift=SCHEMA` adds the 100-refresh test on SIFT1M), `test_rights.sh` (a user with only the documented rights; needs a superuser connection), `test_sq8.sh` (int8 quantisation; `--sift=SCHEMA` adds recall on SIFT1M), `test_filter.sh` (filtered and range search), `test_vscan.sh` (exact search without an index), `test_subcluster.sh` (Eon subclusters; needs `--secondary=<command for a vsql session on another subcluster>`, else skipped), `test_vector_functions.sh`, `test_journal_types.sh` (ARRAY[INT] and ARRAY[NUMERIC] vectors, INT delete flags, TIMESTAMP and INT versions, grants on the views); `run_all.sh` runs them all unfenced and a short set fenced (`--complete`: all in every mode) |
| `docs/` | `design.md` (decisions, measurements), `format.md` (snapshot format), `build-x86.md` (step by step on x86_64 and Eon), `VERTICA_NOTES.md` (verified Vertica behaviour) |

## License

MIT. Author: Mo (github.com/mogomo).

The snapshot, cache, load and freshness code comes from
[vertica-graph-udx](https://github.com/mogomo/vertica-graph-udx).

## Disclaimer

This repository is a demo. It is not a product of, and is not endorsed or
supported by, Rocket Software, Vertica or any other company. The software is
provided "as is", without warranty of any kind, as the MIT license says.
Please try it on your own systems and data before you rely on it, and take
special care with unfenced mode, where the code runs inside the Vertica process.

This repository was tested on several different clusters (one node, a 3-node
Eon cluster and a 4-node Enterprise cluster), with the software versions and
settings named in [Performance and results](#performance-and-results). The
benchmark results are not a statement about any product in general, and your
results will differ. The scripts are included so that you can repeat every
measurement yourself.

Vertica, Rocket Software and all other product and company names are
trademarks or registered trademarks of their respective owners.
