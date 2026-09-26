-- vvector install. Run by scripts/deploy.sh, which sets four vsql variables:
--   libfile         quoted absolute path of libvvector.so on the initiator node
--   fenced_build    FENCED or NOT FENCED: vbuild, vload, vconfig, vnode (memory-heavy or writing)
--   fenced_search   FENCED or NOT FENCED: vsearch, vknn, vinfo, vversion (the query path)
--   search_grantees who may search: vvector_search, or "vvector_search, PUBLIC" (--search=public)
-- Safe to run again: tables and their data are kept; missing manifest columns are added.
\set ON_ERROR_STOP on

-- Upgrade from a version that had vbuild, vload, vconfig and vnode in schema vvector. They move to
-- schema vvector_admin. DROP FUNCTION cannot name a function with an ARRAY argument (Vertica 26.2:
-- syntax error at "ARRAY"), so the library is dropped with all its functions and made again below.
-- The tables, the procedures and the views do not depend on it.
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM v_catalog.user_functions WHERE schema_name = 'vvector' AND function_name = 'vbuild') > 0 THEN
        EXECUTE 'DROP LIBRARY vvector CASCADE';
    END IF;
END;
$$;

CREATE OR REPLACE LIBRARY vvector AS :libfile LANGUAGE 'C++';

-- Catalog.
CREATE SCHEMA IF NOT EXISTS vvector;
-- The functions that build and load snapshots (vbuild, vload, vconfig, vnode): role vvector_admin only.
CREATE SCHEMA IF NOT EXISTS vvector_admin;

-- Upgrade: until milestone M6 the table was UNSEGMENTED ALL NODES (a full copy on every node).
-- A table cannot be resegmented in place, so its rows move to a segmented copy that takes its name.
-- Run it when no refresh runs. Should it stop half way, the next install finishes it: the copy
-- vvector.snapshot_seg is renamed when vvector.snapshot is gone.
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM v_catalog.projections WHERE projection_schema = 'vvector'
          AND anchor_table_name = 'snapshot' AND NOT is_segmented) > 0 THEN
        EXECUTE 'DROP TABLE IF EXISTS vvector.snapshot_seg';
        EXECUTE 'CREATE TABLE vvector.snapshot_seg (index_name VARCHAR(64) NOT NULL, snapshot_id INT NOT NULL, '
             || 'byte_offset INT NOT NULL, chunk LONG VARBINARY(8388608) NOT NULL) '
             || 'ORDER BY index_name, snapshot_id, byte_offset SEGMENTED BY HASH(snapshot_id, byte_offset) ALL NODES';
        EXECUTE 'INSERT INTO vvector.snapshot_seg SELECT index_name, snapshot_id, byte_offset, chunk FROM vvector.snapshot';
        EXECUTE 'COMMIT';
        EXECUTE 'DROP TABLE vvector.snapshot';
    END IF;
    IF (SELECT COUNT(*) FROM v_catalog.tables WHERE table_schema = 'vvector' AND table_name = 'snapshot') = 0
       AND (SELECT COUNT(*) FROM v_catalog.tables WHERE table_schema = 'vvector' AND table_name = 'snapshot_seg') > 0 THEN
        EXECUTE 'ALTER TABLE vvector.snapshot_seg RENAME TO snapshot';
    END IF;
END;
$$;

-- The snapshots in 8 MB chunks. Segmented: a refresh writes one copy (plus the buddy copies of
-- K-safety), not one per node; vload broadcasts the chunks to every node (load_on_nodes). Backed up
-- and protected by K-safety like any other table.
CREATE TABLE IF NOT EXISTS vvector.snapshot (
    index_name   VARCHAR(64) NOT NULL,
    snapshot_id  INT NOT NULL,
    byte_offset  INT NOT NULL,               -- where the piece goes in the snapshot file
    chunk        LONG VARBINARY(8388608) NOT NULL,
    base_snapshot INT DEFAULT NULL           -- a patch: the snapshot these bytes are written over (milestone M7); NULL = a whole copy
) ORDER BY index_name, snapshot_id, byte_offset SEGMENTED BY HASH(snapshot_id, byte_offset) ALL NODES;
-- Upgrade from a version without patches: a whole copy has no base.
ALTER TABLE vvector.snapshot ADD COLUMN IF NOT EXISTS base_snapshot INT DEFAULT NULL;

CREATE TABLE IF NOT EXISTS vvector.manifest (
    index_name        VARCHAR(64) NOT NULL PRIMARY KEY,
    source_table      VARCHAR(256) NOT NULL,
    id_col            VARCHAR(128) NOT NULL,
    vec_col           VARCHAR(128) NOT NULL,  -- ARRAY[FLOAT], ARRAY[INT] or ARRAY[NUMERIC]
    op_col            VARCHAR(128),           -- BOOLEAN (true = deleted) or INT (+1 / -1)
    ver_col           VARCHAR(128),           -- TIMESTAMPTZ, TIMESTAMP or INT; orders the journal
    ver_margin        INT,                    -- overlap of the delta, in units of ver (microseconds for timestamps)
    metric            VARCHAR(16) NOT NULL,   -- l2, cosine, dot or l1
    index_type        VARCHAR(16) NOT NULL    -- flat or hnsw
) UNSEGMENTED ALL NODES;

-- Columns added after the first release are added here, so an existing manifest is upgraded in place.
-- Build options (set_index_options; the next refresh uses them):
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS hnsw_m INT DEFAULT 16;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS hnsw_ef_construction INT DEFAULT 200;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS quantization VARCHAR(16) DEFAULT 'none';
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS memory_mode VARCHAR(16) DEFAULT 'ram';
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS refresh_mode VARCHAR(16) DEFAULT 'auto';
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS tombstone_ratio FLOAT DEFAULT 0.2;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS rebuild_every INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS verify_every INT DEFAULT 1;               -- verify the journal digest every N refreshes, 0 = never
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS cache_dir VARCHAR(1024) DEFAULT NULL;     -- the node cache directory of the index, NULL = the default
-- Query defaults (set_index_options; NULL = the built-in default; the next query uses them):
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS precision_default VARCHAR(16) DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS freshness_default VARCHAR(16) DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS ef_search_default INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS threads_default INT DEFAULT NULL;
-- State of the active snapshot (refresh_index):
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS active_snapshot INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS active_max_ver INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS delta_from VARCHAR(64) DEFAULT NULL;   -- SQL literal: the delta view reads ver_col > delta_from
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS base_snapshot INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS vector_count INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS dims INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS tombstones INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS graph_bytes INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS index_bytes INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS built_at TIMESTAMPTZ DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS build_seconds FLOAT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS format_version INT DEFAULT NULL;
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS active_options VARCHAR(200) DEFAULT NULL;    -- build options of the active snapshot
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS incremental_count INT DEFAULT NULL;         -- incremental refreshes since the last full build
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS boundary_rows INT DEFAULT NULL;             -- journal rows with ver_col <= delta_from at the last refresh
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS boundary_digest NUMERIC(38,0) DEFAULT NULL; -- sum of HASH(id, vec, op, ver) over those rows
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS refreshes_since_verify INT DEFAULT NULL;    -- refreshes since the digest was last verified
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS refresh_started_at TIMESTAMPTZ DEFAULT NULL; -- set while a refresh runs (at most one at a time)
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS refresh_started_by VARCHAR(200) DEFAULT NULL; -- user and session of that refresh
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS refresh_note VARCHAR(1000) DEFAULT NULL;    -- what the last refresh did and why
-- Incremental transfer (milestone M7): vvector.snapshot holds the active chain, the last whole copy
-- and the patch sets after it (the changed bytes of each refresh), nothing else.
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS snapshot_chain VARCHAR(4000) DEFAULT NULL;  -- snapshot ids from the whole copy to the active one, comma separated
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS chain_bytes INT DEFAULT NULL;              -- patch bytes in the table since the whole copy
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS sent_bytes INT DEFAULT NULL;               -- bytes the last refresh stored and sent to the nodes
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS transfer VARCHAR(8) DEFAULT NULL;          -- how the last refresh sent its snapshot: whole or patch
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS capacity INT DEFAULT NULL;                 -- positions the active snapshot's layout has room for
-- Unreachable vectors of an HNSW graph (milestone M7): counted by the build, reported by vinfo.
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS reachability VARCHAR(8) DEFAULT 'auto';    -- auto (full builds, and incremental ones below 8M positions), on, off
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS unreachable INT DEFAULT NULL;              -- live vectors of the active graph no search can reach; NULL = not counted, or flat
-- Journal replica (set_journal_replica; kept up to date by register_index and refresh_index):
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS journal_replica VARCHAR(16) DEFAULT 'auto';     -- auto, on or off
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS replica_projection VARCHAR(256) DEFAULT NULL;   -- schema.projection made by vvector
ALTER TABLE vvector.manifest ADD COLUMN IF NOT EXISTS replica_note VARCHAR(1000) DEFAULT NULL;        -- what the last check did and why

-- Snapshot ids. Never reused, so an old cache file can never pass as a newer snapshot.
CREATE SEQUENCE IF NOT EXISTS vvector.snapshot_seq CACHE 1;

-- Rows on every node, so that functions with OVER(PARTITION NODES) run on
-- every node. It must be segmented: Vertica reads an unsegmented table on one
-- node only. 8192 rows: every node of a large cluster gets some. Made once and filled up to 8192
-- rows: an install never drops it, so a refresh or vinfo that runs meanwhile keeps working.
CREATE TABLE IF NOT EXISTS vvector.probe (k INT NOT NULL) SEGMENTED BY HASH(k) ALL NODES;
INSERT INTO vvector.probe
SELECT k FROM (SELECT ROW_NUMBER() OVER() AS k
               FROM (SELECT 1 FROM (SELECT '2000-01-01 00:00:00'::TIMESTAMP AS t UNION ALL SELECT '2000-01-01 02:16:31'::TIMESTAMP) b
                     TIMESERIES ts AS '1 second' OVER (ORDER BY t)) g) n
WHERE k NOT IN (SELECT k FROM vvector.probe);
COMMIT;

-- CREATE ROLE has no IF NOT EXISTS: on a second install the errors are expected.
\set ON_ERROR_STOP off
CREATE ROLE vvector_admin;
CREATE ROLE vvector_search;
\set ON_ERROR_STOP on

-- Search and information functions live in schema vvector: call them as vvector.vsearch(...) or put
-- vvector on the search path. The build and load functions live in schema vvector_admin.
CREATE OR REPLACE TRANSFORM FUNCTION vvector_admin.vbuild   AS LANGUAGE 'C++' NAME 'VBuildFactory'   LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector_admin.vload    AS LANGUAGE 'C++' NAME 'VLoadFactory'    LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector_admin.vconfig  AS LANGUAGE 'C++' NAME 'VConfigFactory'  LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector_admin.vnode    AS LANGUAGE 'C++' NAME 'VNodeFactory'    LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vsearch  AS LANGUAGE 'C++' NAME 'VSearchFactory'  LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vknn     AS LANGUAGE 'C++' NAME 'VKnnFactory'     LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vscan    AS LANGUAGE 'C++' NAME 'VScanFactory'    LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vinfo    AS LANGUAGE 'C++' NAME 'VInfoFactory'    LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vversion AS LANGUAGE 'C++' NAME 'VVersionFactory' LIBRARY vvector :fenced_search;

-- Vector functions (only those Vertica has no equivalent for), fenced like the search functions.
CREATE OR REPLACE FUNCTION vvector.vector_add        AS LANGUAGE 'C++' NAME 'VectorAddFactory'       LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_sub        AS LANGUAGE 'C++' NAME 'VectorSubFactory'       LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_mul        AS LANGUAGE 'C++' NAME 'VectorMulFactory'       LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.scalar_vector_mul AS LANGUAGE 'C++' NAME 'ScalarVectorMulFactory' LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_normalize  AS LANGUAGE 'C++' NAME 'VectorNormalizeFactory' LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_l1         AS LANGUAGE 'C++' NAME 'VectorL1Factory'        LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_l2sq       AS LANGUAGE 'C++' NAME 'VectorL2sqFactory'      LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_hamming    AS LANGUAGE 'C++' NAME 'VectorHammingFactory'   LIBRARY vvector :fenced_search;
CREATE OR REPLACE FUNCTION vvector.vector_jaccard    AS LANGUAGE 'C++' NAME 'VectorJaccardFactory'   LIBRARY vvector :fenced_search;
-- vector_sum and vector_avg are transform functions: a C++ aggregate cannot read an ARRAY argument.
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vector_sum AS LANGUAGE 'C++' NAME 'VectorSumFactory' LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vector_avg AS LANGUAGE 'C++' NAME 'VectorAvgFactory' LIBRARY vvector :fenced_search;

-- Rights. GRANT and REVOKE cannot name a function with an ARRAY argument (Vertica 26.2: syntax error
-- at "ARRAY"), so rights are given per schema:
--   vvector        vsearch, vknn, vinfo, vversion and the vector functions: role vvector_search
--                  (milestone M6; before, everyone), or also PUBLIC with deploy --search=public. (A role
--                  granted to PUBLIC is not enabled for anyone, so GRANT vvector_search TO PUBLIC would
--                  not do: VERTICA_NOTES.) This also reaches the stored procedures, so procedures.sql
--                  revokes them again and grants them to vvector_admin (sizing: PUBLIC).
--   vvector_admin  vbuild, vload, vconfig, vnode: role vvector_admin only. vload and vconfig write
--                  files on the nodes; vbuild with base_snapshot reads a whole snapshot from the node
--                  cache, so it must not be open to users who may not read the indexed tables.
-- vvector_admin holds vvector_search. The manifest (source tables, boundaries, who refreshes) is
-- read by vvector_admin only; queries never read it.
GRANT USAGE ON SCHEMA vvector TO PUBLIC;
-- Upgrade from the PUBLIC search of earlier versions (a NOTICE on a new install: nothing to revoke).
REVOKE SELECT ON vvector.manifest, vvector.probe FROM PUBLIC;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA vvector FROM PUBLIC;
GRANT SELECT ON vvector.probe TO :search_grantees;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vvector TO :search_grantees;
GRANT vvector_search TO vvector_admin;
GRANT USAGE ON SCHEMA vvector_admin TO vvector_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vvector_admin TO vvector_admin;
GRANT ALL ON vvector.snapshot, vvector.manifest TO vvector_admin;
GRANT SELECT ON SEQUENCE vvector.snapshot_seq TO vvector_admin;

-- Stored procedures: register_index, set_index_options, refresh_index, load_all (one index or all), status, sizing,
-- schedule_refresh, unregister_index.
\i sql/procedures.sql
