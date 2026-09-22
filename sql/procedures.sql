-- vvector stored procedures (PL/vSQL). Installed by sql/install.sql.
--
--   vvector.register_index(index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin)
--   vvector.refresh_index(index_name)
--   vvector.load_all(index_name)
--   vvector.status(index_name)
--   vvector.schedule_refresh(index_name, cron_expr)
--   vvector.unregister_index(index_name)
--
-- Identifiers given by the caller are checked against a strict pattern before
-- they are put into dynamic SQL. Local variables never have the name of a
-- manifest column (PL/vSQL reports an ambiguous column otherwise).

-- The vector column reaches the functions as ARRAY[FLOAT]. A FLOAT array is used as it is: a cast
-- to ARRAY[FLOAT] would give it the default bound of 65000 bytes. INT and NUMERIC arrays are cast.

-- Builds <source schema>.<index_name>_delta from the manifest row.
-- The view returns the journal rows that may be newer than the active snapshot,
-- plus one sentinel row, so its result is never empty. The boundary is a
-- literal, so Vertica can prune partitions and storage containers.
CREATE OR REPLACE PROCEDURE vvector.make_delta_view(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); idc VARCHAR(128); vc VARCHAR(128); op VARCHAR(128); ver VARCHAR(128);
    op_type VARCHAR(128); ver_type VARCHAR(128); vc_type VARCHAR(128);
    sid INT; v_from VARCHAR(64);
    del_expr VARCHAR(400); ver_expr VARCHAR(400);
    view_name VARCHAR(400); stmt VARCHAR(8000); sid_text VARCHAR(32);
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.make_delta_view: index % is not registered', nm;
    END IF;
    idc := (SELECT id_col FROM vvector.manifest WHERE index_name = nm);
    vc := (SELECT vec_col FROM vvector.manifest WHERE index_name = nm);
    op := (SELECT op_col FROM vvector.manifest WHERE index_name = nm);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    sid := (SELECT active_snapshot FROM vvector.manifest WHERE index_name = nm);
    v_from := (SELECT m.delta_from FROM vvector.manifest m WHERE m.index_name = nm);
    view_name := SPLIT_PART(tab, '.', 1) || '.' || nm || '_delta';
    sid_text := COALESCE(sid::VARCHAR, 'NULL::INT');

    stmt := 'CREATE OR REPLACE VIEW ' || view_name || ' AS ';
    IF ver IS NOT NULL AND v_from IS NOT NULL THEN
        op_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                    AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE op);
        ver_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                     AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE ver);
        vc_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                    AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE vc);
        del_expr := CASE WHEN op IS NULL THEN 'FALSE'
                         WHEN op_type ILIKE 'bool%' THEN 'COALESCE(' || op || ', FALSE)'
                         ELSE '(COALESCE(' || op || ', 1) < 0)' END;
        ver_expr := CASE WHEN ver_type ILIKE 'int%' THEN ver || '::INT'
                         ELSE '(EXTRACT(EPOCH FROM ' || ver || ') * 1000000)::INT' END;
        stmt := stmt || 'SELECT NULL::INT AS qid, NULL::ARRAY[FLOAT] AS qvec, ' || idc || '::INT AS id, '
             || CASE WHEN vc_type ILIKE 'array[float8%' THEN vc ELSE vc || '::ARRAY[FLOAT]' END || ' AS vec, ' || del_expr || ' AS del, ' || ver_expr || ' AS ver, '
             || sid_text || ' AS snapshot_id FROM ' || tab || ' WHERE ' || ver || ' > ' || v_from || ' UNION ALL ';
    END IF;
    stmt := stmt || 'SELECT NULL::INT AS qid, NULL::ARRAY[FLOAT] AS qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, '
         || 'NULL::BOOLEAN AS del, NULL::INT AS ver, ' || sid_text || ' AS snapshot_id';
    EXECUTE stmt;
END;
$$;

-- vec_col: ARRAY[FLOAT] (recommended), ARRAY[INT] or ARRAY[NUMERIC]; every vector of the index has the same length.
-- op_col:  BOOLEAN (true = vector deleted) or INT (+1 added, -1 deleted), or NULL when rows are only added.
-- ver_col: TIMESTAMP, TIMESTAMPTZ or INT column that orders the journal (last row per id wins).
--          NULL = static index: queries see the snapshot only, changes show up at the next refresh.
-- metric:  l2, cosine or dot: the measure the index is built and searched for.
-- margin:  overlap of the delta. Timestamp ver_col: seconds (NULL = 60). Open transactions are found
--          through their locks, so it only covers clock differences and statement-start versions.
--          INT ver_col: units of that column; there it must also cover the longest write transaction.
CREATE OR REPLACE PROCEDURE vvector.register_index(nm VARCHAR, src_table VARCHAR, id_column VARCHAR, vec_column VARCHAR,
                                                   op_column VARCHAR, ver_column VARCHAR, measure VARCHAR, margin INT)
LANGUAGE PLvSQL AS $$
DECLARE
    ident VARCHAR(64) := '^[A-Za-z_][A-Za-z0-9_]*$';
    n INT; ver_type VARCHAR(128); op_type VARCHAR(128); vc_type VARCHAR(128); m INT;
BEGIN
    IF nm IS NULL OR NOT REGEXP_LIKE(nm, '^[A-Za-z0-9_]{1,64}$') THEN
        RAISE EXCEPTION 'vvector.register_index: index name must be 1 to 64 letters, digits or underscores';
    END IF;
    IF src_table IS NULL OR NOT REGEXP_LIKE(src_table, '^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$') THEN
        RAISE EXCEPTION 'vvector.register_index: source_table must be given as schema.table';
    END IF;
    IF id_column IS NULL OR vec_column IS NULL OR NOT REGEXP_LIKE(id_column, ident) OR NOT REGEXP_LIKE(vec_column, ident)
       OR NOT REGEXP_LIKE(COALESCE(op_column, 'x'), ident) OR NOT REGEXP_LIKE(COALESCE(ver_column, 'x'), ident) THEN
        RAISE EXCEPTION 'vvector.register_index: column names must be plain identifiers';
    END IF;
    IF measure IS NULL OR measure NOT IN ('l2', 'cosine', 'dot') THEN
        RAISE EXCEPTION 'vvector.register_index: metric must be l2, cosine or dot';
    END IF;
    n := (SELECT COUNT(*) FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(src_table, '.', 1)
          AND table_name ILIKE SPLIT_PART(src_table, '.', 2)
          AND (column_name ILIKE id_column OR column_name ILIKE vec_column OR column_name ILIKE op_column
               OR column_name ILIKE ver_column));
    IF n <> 2 + (op_column IS NOT NULL)::INT + (ver_column IS NOT NULL)::INT THEN
        RAISE EXCEPTION 'vvector.register_index: table % or one of the given columns does not exist', src_table;
    END IF;
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) > 0 THEN
        RAISE EXCEPTION 'vvector.register_index: index % is already registered', nm;
    END IF;
    vc_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(src_table, '.', 1)
                AND table_name ILIKE SPLIT_PART(src_table, '.', 2) AND column_name ILIKE vec_column);
    IF NOT (vc_type ILIKE 'array[float8%' OR vc_type ILIKE 'array[int8%' OR vc_type ILIKE 'array[numeric%') THEN
        RAISE EXCEPTION 'vvector.register_index: vec_col % must be ARRAY[FLOAT], ARRAY[INT] or ARRAY[NUMERIC], not %', vec_column, vc_type;
    END IF;
    IF op_column IS NOT NULL AND ver_column IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: op_col needs ver_col: deletes must be ordered against adds';
    END IF;
    m := NULL;
    IF ver_column IS NOT NULL THEN
        ver_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(src_table, '.', 1)
                     AND table_name ILIKE SPLIT_PART(src_table, '.', 2) AND column_name ILIKE ver_column);
        IF ver_type ILIKE 'timestamp%' THEN
            m := COALESCE(margin, 60) * 1000000;
        ELSIF ver_type ILIKE 'int%' THEN
            IF margin IS NULL THEN
                RAISE EXCEPTION 'vvector.register_index: an INT ver_col needs an explicit margin (units of %)', ver_column;
            END IF;
            m := margin;
        ELSE
            RAISE EXCEPTION 'vvector.register_index: ver_col % must be TIMESTAMP, TIMESTAMPTZ or INT, not %', ver_column, ver_type;
        END IF;
        IF m < 0 THEN
            RAISE EXCEPTION 'vvector.register_index: margin must not be negative';
        END IF;
    END IF;
    IF op_column IS NOT NULL THEN
        op_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(src_table, '.', 1)
                    AND table_name ILIKE SPLIT_PART(src_table, '.', 2) AND column_name ILIKE op_column);
        IF NOT (op_type ILIKE 'bool%' OR op_type ILIKE 'int%') THEN
            RAISE EXCEPTION 'vvector.register_index: op_col % must be BOOLEAN or INT, not %', op_column, op_type;
        END IF;
    END IF;

    PERFORM INSERT INTO vvector.manifest (index_name, source_table, id_col, vec_col, op_col, ver_col, ver_margin, metric, index_type)
            VALUES (nm, src_table, id_column, vec_column, op_column, ver_column, m, measure, 'flat');
    PERFORM COMMIT;
    PERFORM CALL vvector.make_delta_view(nm);
    RAISE NOTICE 'vvector: index % registered. Next: CALL vvector.refresh_index(''%''). Queries read %.%_delta; grant SELECT on it to the users who may search the index.',
                 nm, nm, SPLIT_PART(src_table, '.', 1), nm;
END;
$$;

-- vload of one snapshot on every node, and a check that every node loaded it.
CREATE OR REPLACE PROCEDURE vvector.load_on_nodes(nm VARCHAR, sid INT) LANGUAGE PLvSQL AS $$
DECLARE
    want INT; got INT;
BEGIN
    want := (SELECT COUNT(*) FROM (SELECT vvector.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n);
    got := EXECUTE 'SELECT /*+LABEL(vvector_load)*/ COUNT(DISTINCT node_name) FROM (SELECT vvector.vload(byte_offset, chunk USING PARAMETERS index_name='
        || QUOTE_LITERAL(nm) || ', snapshot_id=' || sid || ') OVER(PARTITION NODES) FROM (SELECT s.byte_offset, s.chunk '
        || 'FROM vvector.snapshot s CROSS JOIN vvector.probe p WHERE s.index_name=' || QUOTE_LITERAL(nm) || ' AND s.snapshot_id=' || sid
        || ' AND p.k IN (SELECT k FROM (SELECT vvector.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) l WHERE status = ''loaded''';
    IF got IS NULL OR got < want THEN
        RAISE EXCEPTION 'vvector.load_on_nodes: index %, snapshot %: loaded on % of % nodes', nm, sid, COALESCE(got, 0), want;
    END IF;
    RAISE NOTICE 'vvector: index %, snapshot % loaded on % nodes', nm, sid, got;
END;
$$;

-- Cache repair: loads the active snapshot again on every node. Safe at any time.
CREATE OR REPLACE PROCEDURE vvector.load_all(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    sid INT;
BEGIN
    sid := (SELECT MAX(active_snapshot) FROM vvector.manifest WHERE index_name = nm);
    IF sid IS NULL THEN
        RAISE EXCEPTION 'vvector.load_all: index % is not registered or has no snapshot yet: run vvector.refresh_index', nm;
    END IF;
    PERFORM CALL vvector.load_on_nodes(nm, sid);
    RAISE NOTICE 'vvector: index %: snapshot % loaded on all nodes', nm, sid;
END;
$$;

-- How many journal rows a query has to apply now, and how long it takes to read them.
CREATE OR REPLACE PROCEDURE vvector.status(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); ver VARCHAR(128); n INT; t0 TIMESTAMPTZ; ms INT;
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.status: index % is not registered', nm;
    END IF;
    t0 := (SELECT CLOCK_TIMESTAMP());
    n := EXECUTE 'SELECT COUNT(id) FROM (SELECT id, vec, del, ver FROM ' || SPLIT_PART(tab, '.', 1) || '.' || nm || '_delta) d';
    ms := (SELECT DATEDIFF('millisecond', t0, CLOCK_TIMESTAMP()));
    RAISE NOTICE 'vvector: index %: % journal rows in the delta, read in % ms', nm, n, ms;
    n := (SELECT COUNT(DISTINCT transaction_id) FROM v_monitor.locks WHERE LOWER(object_name) = LOWER('Table:' || tab)
          AND (lock_mode ILIKE '%I%' OR lock_mode = 'X'));
    IF n > 0 THEN
        ms := (SELECT DATEDIFF('second', MIN(request_timestamp), CLOCK_TIMESTAMP()) FROM v_monitor.locks
               WHERE LOWER(object_name) = LOWER('Table:' || tab) AND (lock_mode ILIKE '%I%' OR lock_mode = 'X'));
        RAISE NOTICE 'vvector: index %: % open transactions are writing to %, the oldest for % seconds. A refresh now keeps their rows in the delta.', nm, n, tab, ms;
    END IF;
    IF ms > 500 THEN
        RAISE WARNING 'vvector: index %: reading the delta is slow (% ms) and every query pays for it. Usual cause: the journal is not partitioned by the date of its version column, so new rows were merged into old storage. Fix: partition the journal by the version date, or refresh more often.', nm, ms;
    END IF;
    IF n > 100000 THEN
        RAISE WARNING 'vvector: index %: the delta holds % rows. Refresh more often.', nm, n;
    END IF;
    -- A version that lies in the future was not set by the database clock: some writer fills the
    -- version column itself. (A version set too far in the past cannot be recognised afterwards.)
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    IF ver IS NOT NULL AND (SELECT data_type ILIKE 'timestamp%' FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                            AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE ver) THEN
        n := EXECUTE 'SELECT COUNT(*) FROM ' || tab || ' WHERE ' || ver || ' > CLOCK_TIMESTAMP() + INTERVAL ''5 minutes''';
        IF n > 0 THEN
            RAISE WARNING 'vvector: index %: % rows of % have a version in the future. The version column must be filled by its default (CLOCK_TIMESTAMP), never by the application; results can be wrong.', nm, n, tab;
        END IF;
    END IF;
END;
$$;

-- boundary -> build -> insert chunks -> vload on all nodes -> update manifest and delta view -> delete older snapshots.
CREATE OR REPLACE PROCEDURE vvector.refresh_index(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); idc VARCHAR(128); vc VARCHAR(128); op VARCHAR(128); ver VARCHAR(128);
    measure VARCHAR(16); kind VARCHAR(16); margin INT; op_type VARCHAR(128); ver_type VARCHAR(128); vc_type VARCHAR(128);
    prev INT; sid INT; chunks INT; max_ver INT; v_from VARCHAR(64);
    source VARCHAR(4000); del_expr VARCHAR(400); v_expr VARCHAR(400);
    t0 TIMESTAMPTZ; cut TIMESTAMPTZ; secs FLOAT; n_vec INT; n_dims INT; fmt INT;
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.refresh_index: index % is not registered', nm;
    END IF;
    idc := (SELECT id_col FROM vvector.manifest WHERE index_name = nm);
    vc := (SELECT vec_col FROM vvector.manifest WHERE index_name = nm);
    op := (SELECT op_col FROM vvector.manifest WHERE index_name = nm);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    measure := (SELECT m.metric FROM vvector.manifest m WHERE m.index_name = nm);
    kind := (SELECT m.index_type FROM vvector.manifest m WHERE m.index_name = nm);
    margin := (SELECT ver_margin FROM vvector.manifest WHERE index_name = nm);
    prev := (SELECT active_snapshot FROM vvector.manifest WHERE index_name = nm);
    vc_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE vc);
    v_expr := CASE WHEN vc_type ILIKE 'array[float8%' THEN vc ELSE vc || '::ARRAY[FLOAT]' END;

    IF prev IS NOT NULL THEN
        PERFORM CALL vvector.status(nm);      -- reports a slow or large delta before it is folded into the new snapshot
    END IF;
    t0 := (SELECT CLOCK_TIMESTAMP());

    -- 1. Delta boundary, taken BEFORE the build reads the table.
    --    A row is missing from the snapshot only if it is committed after the build read starts.
    --    Timestamp version (insertion clock time): such a row was written either after now, or by a
    --    transaction that is open right now. An open writer holds an insert lock on the table, and
    --    v_monitor.locks shows when it asked for it. So:
    --        boundary = LEAST(now, earliest lock request of an open writer) - margin
    --    The margin only has to cover clock differences between nodes and versions taken at statement
    --    start (SYSDATE) instead of at write time (CLOCK_TIMESTAMP).
    --    INT version: highest version minus the margin; the margin has to cover open writers.
    --    Rows that are in both the snapshot and the delta are harmless.
    max_ver := 0;
    v_from := NULL;
    IF ver IS NOT NULL THEN
        ver_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                     AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE ver);
        IF ver_type ILIKE 'int%' THEN
            max_ver := EXECUTE 'SELECT MAX(' || ver || ')::INT FROM ' || tab;
            v_from := (max_ver - margin)::VARCHAR;
        ELSE
            max_ver := EXECUTE 'SELECT MAX((EXTRACT(EPOCH FROM ' || ver || ') * 1000000)::INT) FROM ' || tab;
            cut := (SELECT LEAST(CLOCK_TIMESTAMP(), COALESCE(MIN(request_timestamp), CLOCK_TIMESTAMP()))
                    FROM v_monitor.locks
                    WHERE LOWER(object_name) = LOWER('Table:' || tab)
                      AND (lock_mode ILIKE '%I%' OR lock_mode = 'X')
                      AND transaction_id <> (SELECT transaction_id FROM v_monitor.current_session));
            cut := (SELECT cut - (margin // 1000000) * INTERVAL '1 second');
            IF ver_type ILIKE 'timestamptz%' OR ver_type ILIKE '%with time zone%' THEN
                v_from := (SELECT 'TIMESTAMPTZ ''' || TO_CHAR(cut AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.US') || '+00''');
            ELSE
                -- Plain TIMESTAMP versions are local times of the writing session: all writers and this
                -- session must use the same time zone.
                v_from := (SELECT 'TIMESTAMP ''' || TO_CHAR(cut::TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS.US') || '''');
            END IF;
        END IF;
    END IF;

    -- 2. Consolidated vectors: the latest row of every id by version, kept if it is not a delete.
    --    Without a version column every id must appear once (vbuild refuses a repeated id).
    IF ver IS NULL THEN
        source := 'SELECT ' || idc || '::INT AS id, ' || v_expr || ' AS vec FROM ' || tab || ' WHERE ' || idc || ' IS NOT NULL';
    ELSE
        IF op IS NULL THEN
            del_expr := 'FALSE';
        ELSE
            op_type := (SELECT data_type FROM v_catalog.columns WHERE table_schema ILIKE SPLIT_PART(tab, '.', 1)
                        AND table_name ILIKE SPLIT_PART(tab, '.', 2) AND column_name ILIKE op);
            del_expr := CASE WHEN op_type ILIKE 'bool%' THEN 'COALESCE(' || op || ', FALSE)' ELSE '(COALESCE(' || op || ', 1) < 0)' END;
        END IF;
        source := 'SELECT id, vec FROM (SELECT ' || idc || '::INT AS id, ' || v_expr || ' AS vec, ' || del_expr || ' AS del, '
               || 'ROW_NUMBER() OVER(PARTITION BY ' || idc || ' ORDER BY ' || ver || ' DESC) AS rn FROM ' || tab
               || ' WHERE ' || idc || ' IS NOT NULL) j WHERE rn = 1 AND NOT del';
    END IF;

    -- 3. Build and store the chunks.
    --    Snapshot ids come from a sequence: they never repeat, also not after unregister and register,
    --    so a cache file left behind by an older index of the same name is always recognised as stale.
    sid := (SELECT NEXTVAL('vvector.snapshot_seq'));
    EXECUTE 'INSERT /*+LABEL(vvector_build)*/ INTO vvector.snapshot SELECT ' || QUOTE_LITERAL(nm) || ', ' || sid
         || ', byte_offset, chunk FROM (SELECT vvector.vbuild(id, vec USING PARAMETERS index_name=' || QUOTE_LITERAL(nm)
         || ', metric=' || QUOTE_LITERAL(measure) || ', index_type=' || QUOTE_LITERAL(kind) || ', max_ver=' || COALESCE(max_ver, 0)
         || ') OVER(ORDER BY id) FROM (' || source || ') e) b';
    PERFORM COMMIT;
    chunks := (SELECT COUNT(*) FROM vvector.snapshot WHERE index_name = nm AND snapshot_id = sid);
    IF chunks = 0 THEN
        RAISE EXCEPTION 'vvector.refresh_index: index %: table % has no vectors, nothing to build', nm, tab;
    END IF;

    -- 4. Load on every node. Until the manifest changes, queries keep using the previous snapshot's view.
    PERFORM CALL vvector.load_on_nodes(nm, sid);

    -- 5. Manifest and delta view.
    n_vec := EXECUTE 'SELECT MAX(vector_count) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
    n_dims := EXECUTE 'SELECT MAX(dims) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
    fmt := (SELECT format_version FROM (SELECT vvector.vversion() OVER()) v);
    secs := (SELECT DATEDIFF('millisecond', t0, CLOCK_TIMESTAMP()) / 1000.0);
    PERFORM UPDATE vvector.manifest SET active_snapshot = sid, active_max_ver = max_ver, delta_from = v_from,
                   vector_count = n_vec, dims = n_dims, built_at = CLOCK_TIMESTAMP(), build_seconds = secs, format_version = fmt
            WHERE index_name = nm;
    PERFORM COMMIT;
    PERFORM CALL vvector.make_delta_view(nm);

    -- 6. Keep the active and the previous snapshot.
    PERFORM DELETE FROM vvector.snapshot WHERE index_name = nm AND snapshot_id < COALESCE(prev, sid);
    PERFORM COMMIT;
    RAISE NOTICE 'vvector: index % refreshed: snapshot %, % vectors of % dimensions, % seconds', nm, sid, n_vec, n_dims, secs;
END;
$$;

-- Runs vvector.refresh_index(index_name) on a cron schedule, for example '*/15 * * * *'.
CREATE OR REPLACE PROCEDURE vvector.schedule_refresh(nm VARCHAR, cron_expr VARCHAR) LANGUAGE PLvSQL AS $$
BEGIN
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) = 0 THEN
        RAISE EXCEPTION 'vvector.schedule_refresh: index % is not registered', nm;
    END IF;
    IF cron_expr IS NULL OR NOT REGEXP_LIKE(cron_expr, '^[0-9*/, -]+$') THEN
        RAISE EXCEPTION 'vvector.schedule_refresh: cron_expr may hold digits, spaces and * / , - only';
    END IF;
    EXECUTE 'DROP TRIGGER IF EXISTS vvector.' || nm || '_refresh_trigger';
    EXECUTE 'DROP SCHEDULE IF EXISTS vvector.' || nm || '_refresh_schedule';
    EXECUTE 'CREATE SCHEDULE vvector.' || nm || '_refresh_schedule USING CRON ' || QUOTE_LITERAL(cron_expr);
    EXECUTE 'CREATE TRIGGER vvector.' || nm || '_refresh_trigger ON SCHEDULE vvector.' || nm
         || '_refresh_schedule EXECUTE PROCEDURE vvector.refresh_index(' || QUOTE_LITERAL(nm) || ') AS DEFINER';
    RAISE NOTICE 'vvector: index % is refreshed on schedule %', nm, cron_expr;
END;
$$;

-- Removes the schedule, the delta view, the snapshots and the manifest row.
-- Cache files stay on the nodes: no vvector function deletes paths on request. Remove <cache_dir>/<index_name> by hand.
CREATE OR REPLACE PROCEDURE vvector.unregister_index(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256);
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.unregister_index: index % is not registered', nm;
    END IF;
    EXECUTE 'DROP TRIGGER IF EXISTS vvector.' || nm || '_refresh_trigger';
    EXECUTE 'DROP SCHEDULE IF EXISTS vvector.' || nm || '_refresh_schedule';
    EXECUTE 'DROP VIEW IF EXISTS ' || SPLIT_PART(tab, '.', 1) || '.' || nm || '_delta';
    PERFORM DELETE FROM vvector.snapshot WHERE index_name = nm;
    PERFORM DELETE FROM vvector.manifest WHERE index_name = nm;
    PERFORM COMMIT;
    RAISE NOTICE 'vvector: index % unregistered. Cache files under <cache_dir>/% stay on the nodes.', nm, nm;
END;
$$;

-- install.sql grants all functions of the schema to PUBLIC; on a second install that reached the procedures too.
REVOKE EXECUTE ON PROCEDURE vvector.make_delta_view(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.load_on_nodes(VARCHAR, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.load_all(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.status(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.schedule_refresh(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.unregister_index(VARCHAR) FROM PUBLIC;

GRANT EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.load_all(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.status(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.schedule_refresh(VARCHAR, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.unregister_index(VARCHAR) TO vvector_admin;
