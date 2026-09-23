# Build and install on x86_64, step by step

These steps were run as written on:

| Item      | Tested with |
|-----------|-------------|
| Hardware  | x86_64 with AVX-512, 2 cores, 15 GB RAM per node |
| OS        | Red Hat Enterprise Linux 8.10 |
| Compiler  | g++ 8.5.0 (package `gcc-c++-8.5.0-28.el8_10`), GNU make 4.2.1 |
| Vertica   | 26.2.0-2, Eon mode, 3 nodes in one subcluster, with `/opt/vertica/sdk` |

The same steps work on a single-node database. They were also run on Rocky
Linux 9 on aarch64 with g++ 11.5 and Vertica 26.2.0-1, single node.

Any g++ with full C++17 support should do (g++ 8 or later). Older compilers
were not tested. The library needs no CPU flags at build time: on x86_64 the
distance kernels are compiled for several CPU levels (SSE2, AVX2, AVX-512)
and the best one is picked when the library is loaded. Every level gives
bit-identical scores, so nodes with different CPUs return the same results.

## 1. What you need

- A shell on **one** Vertica node, as the OS user that runs Vertica (usually
  `dbadmin`). You build and deploy from this one node only. `CREATE LIBRARY`
  copies the library to all other nodes.
- The Vertica C++ SDK on that node: `/opt/vertica/sdk/include/Vertica.h` must exist.
  It is installed with the Vertica server package.
- A database user that may create a schema, a library, functions and a role
  (dbadmin, or a user with those rights).
- Packages:

      sudo dnf install -y gcc-c++ make git

- On every node: a directory for the index cache that the Vertica OS user may
  create and write. The default is `/tmp/vvector`. Plan for two snapshots of
  every index: one million vectors of 128 dimensions take 496 MB as a flat
  index and 631 MB as an HNSW index (README, "Index types and tuning").

Check the tools:

    g++ --version        # 8 or later
    make --version
    ls /opt/vertica/sdk/include/Vertica.h

## 2. Get the code

    mkdir -p ~/vvector && cd ~/vvector
    git clone https://github.com/mogomo/vertica-vector-db.git
    cd vertica-vector-db

## 3. Connection settings

The scripts call `vsql` and take the connection from the environment. Nothing
is stored in the repository. On a node, host and port defaults are fine:

    export VSQL_USER=dbadmin
    export VSQL_PASSWORD='...'
    # only if needed: VSQL_HOST, VSQL_PORT, VSQL_DATABASE

    vsql -c "SELECT version();"

## 4. Build and run the unit tests

    make
    make test

`make` takes about 60 seconds on 2 cores and prints no warnings. The result is
`build/libvvector.so`. `make test` needs no database, takes about 2 minutes on
2 cores and ends with

    kernel fingerprint e7324c3e28cb2b21 (avx512f)
    ...
    All engine tests passed.

The fingerprint is a checksum of the kernel results on fixed data. It must be
`e7324c3e28cb2b21` on every machine and CPU level (the name in brackets is the
level in use); the test fails otherwise.

`make test DATA_DIR=<dir>` also measures HNSW recall on SIFT1M when the
directory holds `sift_base.fvecs`, `sift_query.fvecs` and
`sift_groundtruth.ivecs` (it needs about 2.5 GB of memory).

## 5. Deploy

    scripts/deploy.sh --echo_only     # shows what would run, changes nothing
    make deploy                       # fenced mode (default)

or, with the search functions inside the Vertica process (lower latency, see
README "Install"):

    make deploy FENCED=mixed
    make deploy FENCED=no

Deploy creates the library `vvector`, the schema `vvector` with the tables
`snapshot`, `manifest` and `probe`, the sequence `snapshot_seq`, the role
`vvector_admin`, the functions and the stored procedures. It can be run again
at any time; snapshots and the manifest are kept. The message
`ROLLBACK 5403: User/role "vvector_admin" already exists` on a second run is
expected.

At the end it prints the version and the mode of every function:

     library_version | format_version |                            build_flags
    -----------------+----------------+-------------------------------------------------------------------
     0.1.0           |              2 | -O3 -ffp-contract=off -std=c++17 x86_64 g++-8.5.0 kernels=avx512f

     function_name | is_fenced
    ---------------+-----------
     vbuild        | t
     vconfig       | t
     vinfo         | t
     vknn          | t
     vload         | t
     vnode         | t
     vsearch       | t
     vversion      | t

`kernels=` names the CPU level the library picked inside Vertica.

## 6. Check the installation

    tests/sql/run_all.sh

runs every integration test fenced, then unfenced, then mixed, and deploys
fenced again at the end. On the 3-node cluster above it takes about
12 minutes. It creates the schemas `VVTEST`, `VVSEARCH` and `VVHNSW`
and a few small test indexes; `VVSEARCH` and `VVHNSW` are dropped at the end,
`VVTEST` stays (`DROP SCHEMA VVTEST CASCADE;` when done). Every test ends with
a line like

    test_snapshot: OK

and the script ends with

    all integration tests passed in: yes,no,mixed

A single test in the current mode, for example the exactness test of the
search against Vertica's own `VECTOR_L2`, `COSINE_SIMILARITY` and
`DOT_PRODUCT`:

    tests/sql/test_search.sh

On a cluster the line `PASS  vload on every node` of `test_snapshot.sh` is the
important one: it compares the nodes that loaded the cache with the nodes that
are UP.

See what every node has cached:

    SELECT node_name, index_name, snapshot_id, vector_count, index_type, loaded
    FROM (SELECT vvector.vinfo() OVER(PARTITION NODES) FROM vvector.probe) i;

## 7. Remove

    make undeploy

removes the library and the functions. The schema `vvector` with its snapshots
stays. To remove everything:

    DROP SCHEMA vvector CASCADE;
    DROP ROLE vvector_admin;

and delete the cache directory (`/tmp/vvector` by default) on every node.

## Problems seen

- `make: g++: Command not found`: install `gcc-c++`.
- `Vertica.h: No such file or directory`: the SDK is somewhere else. Build with
  `make SDK_HOME=/path/to/sdk`.
- `deploy.sh: ... libvvector.so not found`: run `make` first.
- `ERROR ... Permission denied` from `CREATE LIBRARY`: the database user may not
  create libraries (not a superuser, no `UDXDEVELOPER` role).
- `vsearch: no snapshot cache for index '...' in /tmp/vvector: run vload`: this
  node has no cache file yet, or another `cache_dir` was used at load time.
  `CALL vvector.load_all('<index>');` loads it on every node.
- A node killed by the operating system during tests (out of memory): check
  the free memory of every node first
  (`SELECT host_name, total_memory_free_bytes // 1048576 AS free_mb FROM v_monitor.host_resources;`).
  The integration tests need little memory; a SIFT1M index needs about 0.7 GB
  of cache per node and about 0.8 GB on the node that builds it.
