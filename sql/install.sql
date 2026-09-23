-- vvector install. Run by scripts/deploy.sh, which sets three vsql variables:
--   libfile         quoted absolute path of libvvector.so on the initiator node
--   fenced_build    FENCED or NOT FENCED: vbuild, vload, vconfig, vnode (memory-heavy or writing)
--   fenced_search   FENCED or NOT FENCED: vsearch, vinfo, vversion (the query path)
-- Safe to run again: tables and their data are kept; missing manifest columns are added.
\set ON_ERROR_STOP on

CREATE OR REPLACE LIBRARY vvector AS :libfile LANGUAGE 'C++';

-- Catalog. UNSEGMENTED ALL NODES: every node holds a full copy, so the
-- snapshot is backed up and replicated like any other table.
CREATE SCHEMA IF NOT EXISTS vvector;

CREATE TABLE IF NOT EXISTS vvector.snapshot (
    index_name   VARCHAR(64) NOT NULL,
    snapshot_id  INT NOT NULL,
    byte_offset  INT NOT NULL,               -- where the piece goes in the snapshot file
    chunk        LONG VARBINARY(8388608) NOT NULL
) ORDER BY index_name, snapshot_id, byte_offset UNSEGMENTED ALL NODES;

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

-- Snapshot ids. Never reused, so an old cache file can never pass as a newer snapshot.
CREATE SEQUENCE IF NOT EXISTS vvector.snapshot_seq CACHE 1;

-- Rows on every node, so that functions with OVER(PARTITION NODES) run on
-- every node. It must be segmented: Vertica reads an unsegmented table on one
-- node only. 8192 rows: every node of a large cluster gets some.
\set ON_ERROR_STOP off
DROP TABLE IF EXISTS vvector.probe CASCADE;
\set ON_ERROR_STOP on
CREATE TABLE vvector.probe (k INT NOT NULL) SEGMENTED BY HASH(k) ALL NODES;
INSERT INTO vvector.probe
SELECT ROW_NUMBER() OVER()
FROM (SELECT 1 FROM (SELECT '2000-01-01 00:00:00'::TIMESTAMP AS t UNION ALL SELECT '2000-01-01 02:16:31'::TIMESTAMP) b
      TIMESERIES ts AS '1 second' OVER (ORDER BY t)) g;
COMMIT;

-- CREATE ROLE has no IF NOT EXISTS: on a second install the error is expected.
\set ON_ERROR_STOP off
CREATE ROLE vvector_admin;
\set ON_ERROR_STOP on

-- Functions live in schema vvector. Call them as vvector.vsearch(...) or put vvector
-- on the search path.
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vbuild   AS LANGUAGE 'C++' NAME 'VBuildFactory'   LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vload    AS LANGUAGE 'C++' NAME 'VLoadFactory'    LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vconfig  AS LANGUAGE 'C++' NAME 'VConfigFactory'  LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vnode    AS LANGUAGE 'C++' NAME 'VNodeFactory'    LIBRARY vvector :fenced_build;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vsearch  AS LANGUAGE 'C++' NAME 'VSearchFactory'  LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vinfo    AS LANGUAGE 'C++' NAME 'VInfoFactory'    LIBRARY vvector :fenced_search;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vversion AS LANGUAGE 'C++' NAME 'VVersionFactory' LIBRARY vvector :fenced_search;

-- Rights. vload and vconfig write files on the nodes and the procedures change the catalog:
-- vvector_admin only. The other functions only read their input or the node cache: everyone.
-- GRANT and REVOKE cannot name a function with an ARRAY argument (Vertica 26.2: syntax error at
-- "ARRAY"). So all functions of the schema are granted to PUBLIC in one statement, which also
-- reaches stored procedures, and vload, vconfig and the procedures are revoked again (procedures.sql).
-- vbuild stays open: it writes nothing, it turns its input rows into bytes.
GRANT USAGE ON SCHEMA vvector TO PUBLIC;
GRANT SELECT ON vvector.manifest, vvector.probe TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vvector TO PUBLIC;
REVOKE EXECUTE ON TRANSFORM FUNCTION vvector.vload(INT, LONG VARBINARY) FROM PUBLIC;
GRANT EXECUTE ON TRANSFORM FUNCTION vvector.vload(INT, LONG VARBINARY) TO vvector_admin;
REVOKE EXECUTE ON TRANSFORM FUNCTION vvector.vconfig(INT) FROM PUBLIC;
GRANT EXECUTE ON TRANSFORM FUNCTION vvector.vconfig(INT) TO vvector_admin;
GRANT ALL ON vvector.snapshot, vvector.manifest TO vvector_admin;
GRANT SELECT ON SEQUENCE vvector.snapshot_seq TO vvector_admin;

-- Stored procedures: register_index, set_index_options, refresh_index, load_all, status, sizing,
-- schedule_refresh, unregister_index.
\i sql/procedures.sql
