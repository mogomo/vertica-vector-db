-- vvector install. Run by scripts/deploy.sh, which sets two vsql variables:
--   libfile  quoted absolute path of libvvector.so on the initiator node
--   fenced   FENCED or NOT FENCED
-- Safe to run again: tables and their data are kept.
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
    ver_col           VARCHAR(128),           -- TIMESTAMP or INT; orders the journal
    ver_margin        INT,                    -- overlap of the delta, in units of ver (microseconds for timestamps)
    metric            VARCHAR(16) NOT NULL,   -- l2, cosine or dot
    index_type        VARCHAR(16) NOT NULL,   -- flat or hnsw
    active_snapshot   INT,
    active_max_ver    INT,
    delta_from        VARCHAR(64),            -- SQL literal: the delta view reads ver_col > delta_from
    vector_count      INT,
    dims              INT,
    built_at          TIMESTAMPTZ,
    build_seconds     FLOAT,
    format_version    INT
) UNSEGMENTED ALL NODES;

-- Snapshot ids. Never reused, so an old cache file can never pass as a newer snapshot.
CREATE SEQUENCE IF NOT EXISTS vvector.snapshot_seq CACHE 1;

-- Rows on every node, so that functions with OVER(PARTITION NODES) run on
-- every node. It must be segmented: Vertica reads an unsegmented table on one
-- node only.
\set ON_ERROR_STOP off
DROP TABLE IF EXISTS vvector.probe CASCADE;
\set ON_ERROR_STOP on
CREATE TABLE vvector.probe (k INT NOT NULL) SEGMENTED BY HASH(k) ALL NODES;
INSERT INTO vvector.probe
SELECT ROW_NUMBER() OVER()
FROM (SELECT 1 FROM (SELECT '2000-01-01 00:00:00'::TIMESTAMP AS t UNION ALL SELECT '2000-01-01 00:17:03'::TIMESTAMP) b
      TIMESERIES ts AS '1 second' OVER (ORDER BY t)) g;
COMMIT;

-- CREATE ROLE has no IF NOT EXISTS: on a second install the error is expected.
\set ON_ERROR_STOP off
CREATE ROLE vvector_admin;
\set ON_ERROR_STOP on

-- Functions live in schema vvector. Call them as vvector.vsearch(...) or put vvector
-- on the search path.
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vversion AS LANGUAGE 'C++' NAME 'VVersionFactory' LIBRARY vvector :fenced;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vbuild   AS LANGUAGE 'C++' NAME 'VBuildFactory'   LIBRARY vvector :fenced;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vload    AS LANGUAGE 'C++' NAME 'VLoadFactory'    LIBRARY vvector :fenced;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vnode    AS LANGUAGE 'C++' NAME 'VNodeFactory'    LIBRARY vvector :fenced;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vinfo    AS LANGUAGE 'C++' NAME 'VInfoFactory'    LIBRARY vvector :fenced;
CREATE OR REPLACE TRANSFORM FUNCTION vvector.vsearch  AS LANGUAGE 'C++' NAME 'VSearchFactory'  LIBRARY vvector :fenced;

-- Rights. vload writes files on the nodes and the procedures change the catalog: vvector_admin only.
-- The other functions only read their input or the node cache: everyone.
-- GRANT and REVOKE cannot name a function with an ARRAY argument (Vertica 26.2: syntax error at
-- "ARRAY"). So all functions of the schema are granted to PUBLIC in one statement, which also
-- reaches stored procedures, and vload and the procedures are revoked again (procedures.sql).
-- vbuild stays open: it writes nothing, it turns its input rows into bytes.
GRANT USAGE ON SCHEMA vvector TO PUBLIC;
GRANT SELECT ON vvector.manifest, vvector.probe TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vvector TO PUBLIC;
REVOKE EXECUTE ON TRANSFORM FUNCTION vvector.vload(INT, LONG VARBINARY) FROM PUBLIC;
GRANT EXECUTE ON TRANSFORM FUNCTION vvector.vload(INT, LONG VARBINARY) TO vvector_admin;
GRANT ALL ON vvector.snapshot, vvector.manifest TO vvector_admin;
GRANT SELECT ON SEQUENCE vvector.snapshot_seq TO vvector_admin;

-- Stored procedures: register_index, refresh_index, load_all, status, schedule_refresh, unregister_index.
\i sql/procedures.sql
