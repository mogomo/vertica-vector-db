-- vvector stored procedures (PL/vSQL). Installed by sql/install.sql.
--
--   vvector.register_index(index_name, source_table, id_col, vec_col, op_col, ver_col, metric, margin [, index_type])
--   vvector.set_index_options(index_name, index_type, m, ef_construction, quantization, refresh_mode,
--                             tombstone_ratio, rebuild_every, memory_mode, precision_default,
--                             freshness_default, ef_search_default, threads_default [, verify_every])
--   vvector.refresh_index(index_name [, mode])
--   vvector.set_journal_replica(index_name, auto | on | off)
--   vvector.load_all(index_name)
--   vvector.status(index_name)
--   vvector.sizing(vectors, dims, index_type, quantization)
--   vvector.schedule_refresh(index_name, cron_expr)
--   vvector.unregister_index(index_name)
--
-- Identifiers given by the caller are checked against a strict pattern before
-- they are put into dynamic SQL. Local variables and arguments never have the
-- name of a manifest column (PL/vSQL reports an ambiguous column otherwise).
-- Names are compared with LOWER(a) = LOWER(b), never with ILIKE, where '_' is a wildcard.
-- An IF whose condition is NULL is an error in PL/vSQL (not false): conditions on arguments that may
-- be NULL use COALESCE or IS NOT NULL.

-- The vector column reaches the functions as ARRAY[FLOAT]. A FLOAT array is used as it is: a cast
-- to ARRAY[FLOAT] would give it the default bound of 65000 bytes. INT and NUMERIC arrays are cast.

-- The views of an index, in the schema of its source table:
--   <index>_snap   one sentinel row that carries the active snapshot id: the input of a search that
--                  reads the snapshot only (the fastest statement that keeps the stale-cache check)
--   <index>_delta  (only with a version column) the journal rows that may be newer than the active
--                  snapshot, plus the sentinel row, so its result is never empty. The boundary is a
--                  literal, so Vertica can prune partitions and storage containers.
CREATE OR REPLACE PROCEDURE vvector.make_views(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); idc VARCHAR(128); vc VARCHAR(128); op VARCHAR(128); ver VARCHAR(128);
    op_type VARCHAR(128); ver_type VARCHAR(128); vc_type VARCHAR(128);
    sid INT; v_from VARCHAR(64); sch VARCHAR(128); tbl VARCHAR(128);
    del_expr VARCHAR(400); ver_expr VARCHAR(400); sentinel VARCHAR(1000);
    stmt VARCHAR(8000); sid_text VARCHAR(32);
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.make_views: index % is not registered', nm;
    END IF;
    sch := SPLIT_PART(tab, '.', 1);
    tbl := SPLIT_PART(tab, '.', 2);
    idc := (SELECT id_col FROM vvector.manifest WHERE index_name = nm);
    vc := (SELECT vec_col FROM vvector.manifest WHERE index_name = nm);
    op := (SELECT op_col FROM vvector.manifest WHERE index_name = nm);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    sid := (SELECT active_snapshot FROM vvector.manifest WHERE index_name = nm);
    v_from := (SELECT m.delta_from FROM vvector.manifest m WHERE m.index_name = nm);
    sid_text := COALESCE(sid::VARCHAR, 'NULL::INT');
    sentinel := 'SELECT NULL::INT AS qid, NULL::ARRAY[FLOAT] AS qvec, NULL::INT AS id, NULL::ARRAY[FLOAT] AS vec, '
             || 'NULL::BOOLEAN AS del, NULL::INT AS ver, ' || sid_text || ' AS snapshot_id';

    EXECUTE 'CREATE OR REPLACE VIEW ' || sch || '.' || nm || '_snap AS ' || sentinel;
    IF ver IS NULL THEN
        RETURN;
    END IF;

    stmt := 'CREATE OR REPLACE VIEW ' || sch || '.' || nm || '_delta AS ';
    IF v_from IS NOT NULL THEN
        op_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                    AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(op));
        ver_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                     AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(ver));
        vc_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                    AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(vc));
        del_expr := CASE WHEN op IS NULL THEN 'FALSE'
                         WHEN op_type ILIKE 'bool%' THEN 'COALESCE(' || op || ', FALSE)'
                         ELSE '(COALESCE(' || op || ', 1) < 0)' END;
        ver_expr := CASE WHEN ver_type ILIKE 'int%' THEN ver || '::INT'
                         ELSE '(EXTRACT(EPOCH FROM ' || ver || ') * 1000000)::INT' END;
        stmt := stmt || 'SELECT NULL::INT AS qid, NULL::ARRAY[FLOAT] AS qvec, ' || idc || '::INT AS id, '
             || CASE WHEN vc_type ILIKE 'array[float8%' THEN vc ELSE vc || '::ARRAY[FLOAT]' END || ' AS vec, ' || del_expr || ' AS del, '
             || ver_expr || ' AS ver, ' || sid_text || ' AS snapshot_id FROM ' || tab || ' WHERE ' || ver || ' > ' || v_from
             || ' UNION ALL ';
    END IF;
    EXECUTE stmt || sentinel;
END;
$$;

-- Writes the query defaults of the manifest (precision, freshness, ef_search, threads) into the
-- cache of every node, where vsearch reads them, and checks that every node wrote them.
CREATE OR REPLACE PROCEDURE vvector.push_options(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    opts VARCHAR(400); want INT; got INT;
BEGIN
    opts := (SELECT MAX('precision=' || COALESCE(precision_default, '') || ',freshness=' || COALESCE(freshness_default, '')
                        || ',ef_search=' || COALESCE(ef_search_default::VARCHAR, '') || ',threads=' || COALESCE(threads_default::VARCHAR, '')
                        || ',memory_mode=' || COALESCE(memory_mode, ''))
             FROM vvector.manifest WHERE index_name = nm);
    IF opts IS NULL THEN
        RAISE EXCEPTION 'vvector.push_options: index % is not registered', nm;
    END IF;
    want := (SELECT COUNT(*) FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n);
    got := EXECUTE 'SELECT COUNT(DISTINCT node_name) FROM (SELECT vvector_admin.vconfig(k USING PARAMETERS index_name='
        || QUOTE_LITERAL(nm) || ', options=' || QUOTE_LITERAL(opts) || ') OVER(PARTITION NODES) FROM vvector.probe) c WHERE status = ''written''';
    IF got IS NULL OR got < want THEN
        RAISE EXCEPTION 'vvector.push_options: index %: options written on % of % nodes', nm, COALESCE(got, 0), want;
    END IF;
END;
$$;

-- The journal replica: an UNSEGMENTED ALL NODES projection of the journal, sorted by the version
-- column. On a cluster a query over the delta view otherwise reads the journal on every node and sends
-- the rows to the node that runs the search, even when no row qualifies (15 to 17 ms per statement on
-- the 3-node test cluster; with the replica 4 ms). The planner uses the replica once it has statistics
-- on the version column, so they are taken when it is made and at every refresh.
-- Cost (measured, docs/design.md): a full copy of the journal on every node (in Eon: one more copy in
-- communal storage plus the depot of every node), bulk loads into the journal about 2.5 times slower.
-- Mode (manifest journal_replica, vvector.set_journal_replica):
--   auto  kept while the database has more than one node and one copy of the journal is at most
--         2048 MB and at most 10% of the smallest free disk of a node; made or dropped by
--         register_index and by every refresh_index as the journal grows
--   on    always kept (also on one node, where it only costs)
--   off   never kept
-- CREATE PROJECTION and REFRESH wait for open writers of the table, so the replica is not made while
-- another transaction writes to it (postponed to the next check). Only a projection vvector made itself is ever dropped. Indexes on the same journal columns share
-- one replica. A failure (for example no right to create a projection on the table) is not an error:
-- replica_note says what happened and gives the statements for a DBA.
-- The outcome goes to manifest replica_note: NOTICEs of a nested CALL do not reach the caller.
CREATE OR REPLACE PROCEDURE vvector.apply_replica(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); sch VARCHAR(128); tbl VARCHAR(128); idc VARCHAR(128); vc VARCHAR(128); op VARCHAR(128); ver VARCHAR(128);
    rmode VARCHAR(16); proj VARCHAR(256); other VARCHAR(256); foreign_proj VARCHAR(256); note VARCHAR(1000);
    nodes INT; journal_mb INT; free_mb INT; limit_mb INT; keep BOOLEAN; users INT; ddl VARCHAR(2000); r VARCHAR(1000);
    writers INT;
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.apply_replica: index % is not registered', nm;
    END IF;
    sch := SPLIT_PART(tab, '.', 1);
    tbl := SPLIT_PART(tab, '.', 2);
    idc := (SELECT id_col FROM vvector.manifest WHERE index_name = nm);
    vc := (SELECT vec_col FROM vvector.manifest WHERE index_name = nm);
    op := (SELECT op_col FROM vvector.manifest WHERE index_name = nm);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    rmode := (SELECT COALESCE(MAX(journal_replica), 'auto') FROM vvector.manifest WHERE index_name = nm);
    proj := (SELECT MAX(replica_projection) FROM vvector.manifest WHERE index_name = nm);

    nodes := (SELECT COUNT(*) FROM v_catalog.nodes);
    -- One copy of the journal: the largest segmented projection of the table, summed over the nodes.
    journal_mb := (SELECT COALESCE(MAX(b), 0) FROM (
                     SELECT SUM(used_bytes) // 1048576 AS b FROM v_monitor.projection_storage
                     WHERE LOWER(anchor_table_schema) = LOWER(sch) AND LOWER(anchor_table_name) = LOWER(tbl)
                       AND projection_name IN (SELECT projection_name FROM v_catalog.projections
                                               WHERE LOWER(projection_schema) = LOWER(sch) AND LOWER(anchor_table_name) = LOWER(tbl)
                                                 AND is_segmented)
                     GROUP BY projection_name) s);
    free_mb := (SELECT COALESCE(MIN(disk_space_free_mb), 0) FROM v_monitor.disk_storage
                WHERE storage_usage ILIKE '%DATA%' OR storage_usage ILIKE '%DEPOT%');
    limit_mb := LEAST(2048, free_mb // 10);
    -- An unsegmented projection of the table that vvector did not make: the reads are local already.
    foreign_proj := (SELECT MAX(projection_schema || '.' || projection_name) FROM v_catalog.projections
                     WHERE LOWER(projection_schema) = LOWER(sch) AND LOWER(anchor_table_name) = LOWER(tbl) AND NOT is_segmented
                       AND LOWER(projection_schema || '.' || projection_name) <> LOWER(COALESCE(proj, ''))
                       AND LOWER(projection_schema || '.' || projection_name) NOT IN
                           (SELECT LOWER(replica_projection) FROM vvector.manifest WHERE replica_projection IS NOT NULL));

    keep := ver IS NOT NULL AND (rmode = 'on' OR (rmode = 'auto' AND nodes > 1 AND journal_mb <= limit_mb AND foreign_proj IS NULL));
    note := '';
    IF ver IS NULL THEN
        note := 'none: a static index (no version column) has no delta view';
    ELSIF rmode = 'off' THEN
        note := 'none: journal_replica is off';
    ELSIF rmode = 'auto' AND nodes = 1 THEN
        note := 'none: a single node reads the delta locally already';
    ELSIF rmode = 'auto' AND foreign_proj IS NOT NULL THEN
        note := 'none: the table has an unsegmented projection already (' || foreign_proj || ')';
    ELSIF rmode = 'auto' AND journal_mb > limit_mb THEN
        note := 'none: the journal takes ' || journal_mb || ' MB, more than the limit of ' || limit_mb
             || ' MB for a copy on every node (2048 MB, and at most 10% of the smallest free disk). CALL vvector.set_journal_replica('''
             || nm || ''', ''on'') makes it anyway';
    END IF;

    writers := 0;
    IF keep THEN
        IF proj IS NOT NULL AND (SELECT COUNT(*) FROM v_catalog.projections
                                 WHERE LOWER(projection_schema || '.' || projection_name) = LOWER(proj)) = 0 THEN
            proj := NULL;                        -- dropped by someone else: make it again
        END IF;
        other := NULL;
        IF proj IS NULL THEN
            other := (SELECT MAX(replica_projection) FROM vvector.manifest
                      WHERE LOWER(source_table) = LOWER(tab) AND index_name <> nm AND replica_projection IS NOT NULL
                        AND LOWER(id_col) = LOWER(idc) AND LOWER(vec_col) = LOWER(vc) AND LOWER(ver_col) = LOWER(ver)
                        AND LOWER(COALESCE(op_col, '')) = LOWER(COALESCE(op, '')));
        END IF;
        IF proj IS NOT NULL THEN
            note := 'kept ' || proj;
        ELSIF other IS NOT NULL THEN
            proj := other;
            note := 'shared ' || proj;
        ELSE
            -- CREATE PROJECTION and REFRESH wait until every open transaction that writes to the table
            -- has ended. A refresh must not hang behind a long load: try again at the next refresh.
            writers := (SELECT COUNT(DISTINCT transaction_id) FROM v_monitor.locks
                        WHERE LOWER(object_name) = LOWER('Table:' || tab) AND (lock_mode ILIKE '%I%' OR lock_mode = 'X')
                          AND transaction_id <> (SELECT transaction_id FROM v_monitor.current_session));
        END IF;
        IF proj IS NULL AND other IS NULL AND writers > 0 THEN
            note := 'postponed: ' || writers || ' open transactions write to ' || tab
                 || ' and a new projection would wait for them; the next refresh_index (or set_journal_replica) tries again';
        ELSIF proj IS NULL AND other IS NULL THEN
            ddl := 'CREATE PROJECTION ' || sch || '.' || nm || '_journal_rep AS SELECT ' || idc || ', ' || vc
                || CASE WHEN op IS NULL THEN '' ELSE ', ' || op END || ', ' || ver || ' FROM ' || tab
                || ' ORDER BY ' || ver || ' UNSEGMENTED ALL NODES';
            BEGIN
                EXECUTE ddl;
                r := EXECUTE 'SELECT REFRESH(' || QUOTE_LITERAL(tab) || ')';
                proj := sch || '.' || nm || '_journal_rep';
                note := 'created ' || proj || ' (journal ' || journal_mb || ' MB, one copy on each of ' || nodes || ' nodes)';
            EXCEPTION WHEN OTHERS THEN
                proj := NULL;
                note := 'not created: ' || LEFT(SQLERRM, 300) || '. A DBA can run: ' || ddl || '; SELECT REFRESH('
                     || QUOTE_LITERAL(tab) || '); SELECT ANALYZE_STATISTICS(' || QUOTE_LITERAL(tab || '.' || ver) || ');';
            END;
        END IF;
        IF proj IS NOT NULL THEN
            BEGIN
                r := EXECUTE 'SELECT ANALYZE_STATISTICS(' || QUOTE_LITERAL(tab || '.' || ver) || ')';
            EXCEPTION WHEN OTHERS THEN
                note := note || '; statistics on ' || ver || ' failed (' || LEFT(SQLERRM, 200) || '): the planner may not use it';
            END;
            IF nodes = 1 THEN
                note := note || '; a single node gains nothing from it';
            END IF;
        END IF;
    ELSIF proj IS NOT NULL THEN
        users := (SELECT COUNT(*) FROM vvector.manifest WHERE LOWER(replica_projection) = LOWER(proj) AND index_name <> nm);
        IF users = 0 THEN
            BEGIN
                EXECUTE 'DROP PROJECTION IF EXISTS ' || proj;
                note := note || '; dropped ' || proj;
            EXCEPTION WHEN OTHERS THEN
                note := note || '; could not drop ' || proj || ': ' || LEFT(SQLERRM, 200);
            END;
        END IF;
        proj := NULL;
    END IF;
    PERFORM UPDATE vvector.manifest SET replica_projection = proj, replica_note = note WHERE index_name = nm;
    PERFORM COMMIT;
END;
$$;

-- journal_replica mode of an index: auto (the default), on or off (see apply_replica). Applies at once.
CREATE OR REPLACE PROCEDURE vvector.set_journal_replica(nm VARCHAR, x_mode VARCHAR) LANGUAGE PLvSQL AS $$
BEGIN
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) = 0 THEN
        RAISE EXCEPTION 'vvector.set_journal_replica: index % is not registered', nm;
    END IF;
    IF x_mode IS NULL OR x_mode NOT IN ('auto', 'on', 'off') THEN
        RAISE EXCEPTION 'vvector.set_journal_replica: mode must be auto, on or off';
    END IF;
    PERFORM UPDATE vvector.manifest SET journal_replica = x_mode WHERE index_name = nm;
    PERFORM COMMIT;
    PERFORM CALL vvector.apply_replica(nm);
    RAISE NOTICE 'vvector: index %: journal replica %: %', nm, x_mode,
                 (SELECT MAX(replica_note) FROM vvector.manifest WHERE index_name = nm);
END;
$$;

-- vec_col:    ARRAY[FLOAT] (recommended), ARRAY[INT] or ARRAY[NUMERIC]; every vector of the index has the same length.
-- op_col:     BOOLEAN (true = vector deleted) or INT (+1 added, -1 deleted), or NULL when rows are only added.
-- ver_col:    TIMESTAMPTZ (recommended), TIMESTAMP or INT column that orders the journal (last row per id wins).
--             NULL = static index: queries see the snapshot only, changes show up at the next refresh.
-- metric:     l2, cosine, dot or l1: the measure the index is built and searched for.
-- margin:     overlap of the delta. Timestamp ver_col: seconds (NULL = 60). Open transactions are found
--             through their locks, so it only covers clock differences and statement-start versions.
--             INT ver_col: units of that column; there it must also cover the longest write transaction.
-- index_type: hnsw (the default: approximate, fast) or flat (every query scans every vector).
-- register_index_core does the work; the two forms of register_index call it and print the result
-- themselves, because NOTICEs of a nested CALL do not reach the caller.
CREATE OR REPLACE PROCEDURE vvector.register_index_core(nm VARCHAR, src_table VARCHAR, id_column VARCHAR, vec_column VARCHAR,
                                                        op_column VARCHAR, ver_column VARCHAR, measure VARCHAR, margin INT,
                                                        kind VARCHAR)
LANGUAGE PLvSQL AS $$
DECLARE
    ident VARCHAR(64) := '^[A-Za-z_][A-Za-z0-9_]*$';
    sch VARCHAR(128); tbl VARCHAR(128);
    ver_type VARCHAR(128); op_type VARCHAR(128); vc_type VARCHAR(128); id_type VARCHAR(128); m INT;
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
    IF measure IS NULL OR measure NOT IN ('l2', 'cosine', 'dot', 'l1') THEN
        RAISE EXCEPTION 'vvector.register_index: metric must be l2, cosine, dot or l1';
    END IF;
    IF kind IS NULL OR kind NOT IN ('flat', 'hnsw') THEN
        RAISE EXCEPTION 'vvector.register_index: index_type must be flat or hnsw';
    END IF;
    sch := SPLIT_PART(src_table, '.', 1);
    tbl := SPLIT_PART(src_table, '.', 2);
    IF (SELECT COUNT(*) FROM v_catalog.tables WHERE LOWER(table_schema) = LOWER(sch) AND LOWER(table_name) = LOWER(tbl)) = 0 THEN
        RAISE EXCEPTION 'vvector.register_index: table % does not exist', src_table;
    END IF;
    id_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(id_column));
    vc_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(vec_column));
    op_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(op_column));
    ver_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                 AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(ver_column));
    IF id_type IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: column % does not exist in %', id_column, src_table;
    END IF;
    IF vc_type IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: column % does not exist in %', vec_column, src_table;
    END IF;
    IF op_column IS NOT NULL AND op_type IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: column % does not exist in %', op_column, src_table;
    END IF;
    IF ver_column IS NOT NULL AND ver_type IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: column % does not exist in %', ver_column, src_table;
    END IF;
    IF NOT id_type ILIKE 'int%' THEN
        RAISE EXCEPTION 'vvector.register_index: id_col % must be INT, not %', id_column, id_type;
    END IF;
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) > 0 THEN
        RAISE EXCEPTION 'vvector.register_index: index % is already registered', nm;
    END IF;
    IF NOT (vc_type ILIKE 'array[float8%' OR vc_type ILIKE 'array[int8%' OR vc_type ILIKE 'array[numeric%') THEN
        RAISE EXCEPTION 'vvector.register_index: vec_col % must be ARRAY[FLOAT], ARRAY[INT] or ARRAY[NUMERIC], not %', vec_column, vc_type;
    END IF;
    IF op_column IS NOT NULL AND ver_column IS NULL THEN
        RAISE EXCEPTION 'vvector.register_index: op_col needs ver_col: deletes must be ordered against adds';
    END IF;
    m := NULL;
    IF ver_column IS NOT NULL THEN
        IF ver_type ILIKE 'timestamp%' THEN
            m := COALESCE(margin, 60) * 1000000;
        ELSIF ver_type ILIKE 'int%' THEN
            IF margin IS NULL THEN
                RAISE EXCEPTION 'vvector.register_index: an INT ver_col needs an explicit margin (units of %)', ver_column;
            END IF;
            m := margin;
        ELSE
            RAISE EXCEPTION 'vvector.register_index: ver_col % must be TIMESTAMPTZ, TIMESTAMP or INT, not %', ver_column, ver_type;
        END IF;
        IF m < 0 THEN
            RAISE EXCEPTION 'vvector.register_index: margin must not be negative';
        END IF;
    END IF;
    IF op_column IS NOT NULL AND NOT (op_type ILIKE 'bool%' OR op_type ILIKE 'int%') THEN
        RAISE EXCEPTION 'vvector.register_index: op_col % must be BOOLEAN or INT, not %', op_column, op_type;
    END IF;

    PERFORM INSERT INTO vvector.manifest (index_name, source_table, id_col, vec_col, op_col, ver_col, ver_margin, metric, index_type)
            VALUES (nm, src_table, id_column, vec_column, op_column, ver_column, m, measure, kind);
    PERFORM COMMIT;
    PERFORM CALL vvector.make_views(nm);
    PERFORM CALL vvector.apply_replica(nm);
END;
$$;

CREATE OR REPLACE PROCEDURE vvector.register_index(nm VARCHAR, src_table VARCHAR, id_column VARCHAR, vec_column VARCHAR,
                                                   op_column VARCHAR, ver_column VARCHAR, measure VARCHAR, margin INT,
                                                   kind VARCHAR)
LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.register_index_core(nm, src_table, id_column, vec_column, op_column, ver_column, measure, margin, kind);
    IF ver_column IS NULL THEN
        RAISE NOTICE 'vvector: index % registered (static: no version column). Next: CALL vvector.refresh_index(''%''). Queries read %_snap in schema %; grant SELECT on it to the users who may search the index.',
                     nm, nm, nm, SPLIT_PART(src_table, '.', 1);
    ELSE
        RAISE NOTICE 'vvector: index % registered. Next: CALL vvector.refresh_index(''%''). Queries read %_snap (snapshot only) or %_delta (with the changes since the refresh) in schema %; grant SELECT on them to the users who may search the index.',
                     nm, nm, nm, nm, SPLIT_PART(src_table, '.', 1);
        RAISE NOTICE 'vvector: index %: journal replica: %', nm, (SELECT MAX(replica_note) FROM vvector.manifest WHERE index_name = nm);
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE vvector.register_index(nm VARCHAR, src_table VARCHAR, id_column VARCHAR, vec_column VARCHAR,
                                                   op_column VARCHAR, ver_column VARCHAR, measure VARCHAR, margin INT)
LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.register_index_core(nm, src_table, id_column, vec_column, op_column, ver_column, measure, margin, 'hnsw');
    IF ver_column IS NULL THEN
        RAISE NOTICE 'vvector: index % registered (static: no version column). Next: CALL vvector.refresh_index(''%''). Queries read %_snap in schema %; grant SELECT on it to the users who may search the index.',
                     nm, nm, nm, SPLIT_PART(src_table, '.', 1);
    ELSE
        RAISE NOTICE 'vvector: index % registered. Next: CALL vvector.refresh_index(''%''). Queries read %_snap (snapshot only) or %_delta (with the changes since the refresh) in schema %; grant SELECT on them to the users who may search the index.',
                     nm, nm, nm, nm, SPLIT_PART(src_table, '.', 1);
        RAISE NOTICE 'vvector: index %: journal replica: %', nm, (SELECT MAX(replica_note) FROM vvector.manifest WHERE index_name = nm);
    END IF;
END;
$$;

-- Changes the options of an index. NULL keeps a value. Build options (x_type, x_m, x_efc, x_quant) take
-- effect at the next refresh; query defaults (x_prec, x_fresh, x_ef, x_thr) at once, on every node;
-- memory_mode (x_memory: ram, or compact with quantization sq8) with the next snapshot a node maps.
-- 'default' (text) or 0 (numbers) sets a query default back to the built-in default.
-- x_verify (verify_every, the last argument of the 14-argument form): verify the journal digest at every
-- refresh (1, the default), every N refreshes (N), or never (0).
-- set_index_options_core does the work; both forms print the NOTICE (NOTICEs of a nested CALL do not
-- reach the caller).
CREATE OR REPLACE PROCEDURE vvector.set_index_options_core(nm VARCHAR, x_type VARCHAR, x_m INT, x_efc INT, x_quant VARCHAR,
                                                           x_refresh VARCHAR, x_ratio FLOAT, x_rebuild INT, x_memory VARCHAR,
                                                           x_prec VARCHAR, x_fresh VARCHAR, x_ef INT, x_thr INT, x_verify INT)
LANGUAGE PLvSQL AS $$
BEGIN
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) = 0 THEN
        RAISE EXCEPTION 'vvector.set_index_options: index % is not registered', nm;
    END IF;
    IF x_type IS NOT NULL AND x_type NOT IN ('flat', 'hnsw') THEN
        RAISE EXCEPTION 'vvector.set_index_options: index_type must be flat or hnsw';
    END IF;
    IF x_m IS NOT NULL AND (x_m < 2 OR x_m > 256) THEN
        RAISE EXCEPTION 'vvector.set_index_options: m must be 2 to 256';
    END IF;
    IF x_efc IS NOT NULL AND (x_efc < 1 OR x_efc > 100000) THEN
        RAISE EXCEPTION 'vvector.set_index_options: ef_construction must be 1 to 100000';
    END IF;
    IF x_quant IS NOT NULL AND x_quant NOT IN ('none', 'sq8') THEN
        RAISE EXCEPTION 'vvector.set_index_options: quantization must be none or sq8';
    END IF;
    IF x_refresh IS NOT NULL AND x_refresh NOT IN ('auto', 'incremental', 'full') THEN
        RAISE EXCEPTION 'vvector.set_index_options: refresh_mode must be auto, incremental or full';
    END IF;
    IF x_ratio IS NOT NULL AND NOT (x_ratio > 0 AND x_ratio <= 1) THEN
        RAISE EXCEPTION 'vvector.set_index_options: tombstone_ratio must be above 0 and at most 1';
    END IF;
    IF x_rebuild IS NOT NULL AND x_rebuild < 0 THEN
        RAISE EXCEPTION 'vvector.set_index_options: rebuild_every must be 0 (never) or more';
    END IF;
    IF x_memory IS NOT NULL AND x_memory NOT IN ('ram', 'compact') THEN
        RAISE EXCEPTION 'vvector.set_index_options: memory_mode must be ram or compact';
    END IF;
    IF COALESCE(x_memory, (SELECT MAX(memory_mode) FROM vvector.manifest WHERE index_name = nm)) = 'compact'
       AND COALESCE(x_quant, (SELECT MAX(quantization) FROM vvector.manifest WHERE index_name = nm), 'none') <> 'sq8' THEN
        RAISE EXCEPTION 'vvector.set_index_options: memory_mode compact needs quantization sq8';
    END IF;
    IF x_prec IS NOT NULL AND x_prec NOT IN ('fast', 'balanced', 'best', 'exact', 'default') THEN
        RAISE EXCEPTION 'vvector.set_index_options: precision_default must be fast, balanced, best, exact or default';
    END IF;
    IF x_fresh IS NOT NULL AND x_fresh NOT IN ('snapshot', 'exact', 'default') THEN
        RAISE EXCEPTION 'vvector.set_index_options: freshness_default must be snapshot, exact or default';
    END IF;
    IF x_ef IS NOT NULL AND (x_ef < 0 OR x_ef > 100000) THEN
        RAISE EXCEPTION 'vvector.set_index_options: ef_search_default must be 0 (built-in) to 100000';
    END IF;
    IF x_thr IS NOT NULL AND (x_thr < 0 OR x_thr > 64) THEN
        RAISE EXCEPTION 'vvector.set_index_options: threads_default must be 0 (one per core) to 64';
    END IF;
    IF x_verify IS NOT NULL AND x_verify < 0 THEN
        RAISE EXCEPTION 'vvector.set_index_options: verify_every must be 0 (never), 1 (every refresh) or more';
    END IF;

    PERFORM UPDATE vvector.manifest SET
        index_type = COALESCE(x_type, index_type),
        hnsw_m = COALESCE(x_m, hnsw_m),
        hnsw_ef_construction = COALESCE(x_efc, hnsw_ef_construction),
        quantization = COALESCE(x_quant, quantization),
        refresh_mode = COALESCE(x_refresh, refresh_mode),
        tombstone_ratio = COALESCE(x_ratio, tombstone_ratio),
        rebuild_every = CASE WHEN x_rebuild IS NULL THEN rebuild_every WHEN x_rebuild = 0 THEN NULL ELSE x_rebuild END,
        memory_mode = COALESCE(x_memory, memory_mode),
        precision_default = CASE WHEN x_prec IS NULL THEN precision_default WHEN x_prec = 'default' THEN NULL ELSE x_prec END,
        freshness_default = CASE WHEN x_fresh IS NULL THEN freshness_default WHEN x_fresh = 'default' THEN NULL ELSE x_fresh END,
        ef_search_default = CASE WHEN x_ef IS NULL THEN ef_search_default WHEN x_ef = 0 THEN NULL ELSE x_ef END,
        threads_default = CASE WHEN x_thr IS NULL THEN threads_default WHEN x_thr = 0 THEN NULL ELSE x_thr END,
        verify_every = COALESCE(x_verify, verify_every)
        WHERE index_name = nm;
    PERFORM COMMIT;
    PERFORM CALL vvector.push_options(nm);
END;
$$;

CREATE OR REPLACE PROCEDURE vvector.set_index_options(nm VARCHAR, x_type VARCHAR, x_m INT, x_efc INT, x_quant VARCHAR,
                                                      x_refresh VARCHAR, x_ratio FLOAT, x_rebuild INT, x_memory VARCHAR,
                                                      x_prec VARCHAR, x_fresh VARCHAR, x_ef INT, x_thr INT, x_verify INT)
LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.set_index_options_core(nm, x_type, x_m, x_efc, x_quant, x_refresh, x_ratio, x_rebuild, x_memory,
                                                x_prec, x_fresh, x_ef, x_thr, x_verify);
    RAISE NOTICE 'vvector: index % options changed. Build options apply at the next refresh; query defaults apply now.', nm;
END;
$$;

-- The 13-argument form of the first releases: verify_every stays as it is.
CREATE OR REPLACE PROCEDURE vvector.set_index_options(nm VARCHAR, x_type VARCHAR, x_m INT, x_efc INT, x_quant VARCHAR,
                                                      x_refresh VARCHAR, x_ratio FLOAT, x_rebuild INT, x_memory VARCHAR,
                                                      x_prec VARCHAR, x_fresh VARCHAR, x_ef INT, x_thr INT)
LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.set_index_options_core(nm, x_type, x_m, x_efc, x_quant, x_refresh, x_ratio, x_rebuild, x_memory,
                                                x_prec, x_fresh, x_ef, x_thr, NULL);
    RAISE NOTICE 'vvector: index % options changed. Build options apply at the next refresh; query defaults apply now.', nm;
END;
$$;

-- vload of one snapshot on every node, and a check that every node loaded it.
CREATE OR REPLACE PROCEDURE vvector.load_on_nodes(nm VARCHAR, sid INT) LANGUAGE PLvSQL AS $$
DECLARE
    want INT; got INT; hint VARCHAR(40); dist VARCHAR(40);
BEGIN
    want := (SELECT COUNT(*) FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n);
    -- One probe row per node, joined to every chunk: each node's vload gets the whole snapshot. The
    -- snapshot table is segmented (one copy written per refresh, not one per node), so on more than
    -- one node the chunks are broadcast to the probe rows (DISTRIB(L,B); join hints need
    -- SYNTACTIC_JOIN). On one node the hint would only give "not feasible" warnings.
    IF want > 1 THEN
        hint := '/*+SYNTACTIC_JOIN*/ '; dist := '/*+DISTRIB(L,B)*/ ';
    ELSE
        hint := ''; dist := '';
    END IF;
    got := EXECUTE 'SELECT /*+LABEL(vvector_load)*/ COUNT(DISTINCT node_name) FROM (SELECT vvector_admin.vload(byte_offset, chunk USING PARAMETERS index_name='
        || QUOTE_LITERAL(nm) || ', snapshot_id=' || sid || ') OVER(PARTITION NODES) FROM (SELECT ' || hint || 's.byte_offset, s.chunk '
        || 'FROM vvector.probe p JOIN ' || dist || 'vvector.snapshot s ON TRUE WHERE s.index_name=' || QUOTE_LITERAL(nm) || ' AND s.snapshot_id=' || sid
        || ' AND p.k IN (SELECT k FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n)) c) l WHERE status = ''loaded''';
    IF got IS NULL OR got < want THEN
        RAISE EXCEPTION 'vvector.load_on_nodes: index %, snapshot %: loaded on % of % nodes', nm, sid, COALESCE(got, 0), want;
    END IF;
END;
$$;

-- Cache repair: loads the active snapshot and the index defaults again on every node. Safe at any time.
CREATE OR REPLACE PROCEDURE vvector.load_all(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    sid INT;
BEGIN
    sid := (SELECT MAX(active_snapshot) FROM vvector.manifest WHERE index_name = nm);
    IF sid IS NULL THEN
        RAISE EXCEPTION 'vvector.load_all: index % is not registered or has no snapshot yet: run vvector.refresh_index', nm;
    END IF;
    PERFORM CALL vvector.load_on_nodes(nm, sid);
    PERFORM CALL vvector.push_options(nm);
    RAISE NOTICE 'vvector: index %: snapshot % loaded on all nodes', nm, sid;
END;
$$;

-- Memory estimate for an index before its table is loaded. Prints one line per figure; changes nothing.
-- index_type flat or hnsw (with m = 16), quantization none or sq8.
CREATE OR REPLACE PROCEDURE vvector.sizing(n_vectors INT, n_dims INT, kind VARCHAR, quant VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    stride INT; vec_mb FLOAT; ids_mb FLOAT; graph_mb FLOAT; sq8_mb FLOAT; total_mb FLOAT; build_mb FLOAT; mem_gb FLOAT;
BEGIN
    IF n_vectors IS NULL OR n_vectors < 1 OR n_vectors > 4294967295 THEN
        RAISE EXCEPTION 'vvector.sizing: vectors must be 1 to 4294967295';
    END IF;
    IF n_dims IS NULL OR n_dims < 1 OR n_dims > 32768 THEN
        RAISE EXCEPTION 'vvector.sizing: dims must be 1 to 32768';
    END IF;
    IF COALESCE(kind, 'flat') NOT IN ('flat', 'hnsw') OR COALESCE(quant, 'none') NOT IN ('none', 'sq8') THEN
        RAISE EXCEPTION 'vvector.sizing: index_type must be flat or hnsw, quantization none or sq8';
    END IF;
    stride := (n_dims + 15) // 16 * 16;
    vec_mb := n_vectors * stride * 4 / 1048576.0;
    ids_mb := n_vectors * 8 / 1048576.0;
    -- HNSW with m = 16: level 0 holds 2m + 1 uint32 per vector, plus a level byte and an index entry
    -- (5 bytes); a vector has 1/(m - 1) upper blocks of m + 1 uint32 on average.
    graph_mb := CASE WHEN kind = 'hnsw' THEN n_vectors * (33 * 4 + 5 + 17 * 4 / 15.0) / 1048576.0 ELSE 0 END;
    sq8_mb := CASE WHEN quant = 'sq8' THEN n_vectors * (stride + 4) / 1048576.0 ELSE 0 END;     -- a byte per element, a sum per vector
    total_mb := vec_mb + ids_mb + graph_mb + sq8_mb;
    mem_gb := (SELECT MIN(total_memory_bytes) FROM v_monitor.host_resources) / 1073741824.0;
    -- Build: the snapshot, 4 bytes per vector to sort; HNSW adds 2 bytes per vector per build thread
    -- (one per core) and 5 bytes per vector while the levels are laid out.
    build_mb := total_mb + n_vectors * 4 / 1048576.0
                + CASE WHEN kind = 'hnsw' THEN n_vectors * (5 + 2 * (SELECT MIN(processor_core_count) FROM v_monitor.host_resources)) / 1048576.0 ELSE 0 END;
    RAISE NOTICE 'vvector.sizing: % vectors of % dimensions (% floats per row): vectors % MB, ids % MB, graph % MB, sq8 codes % MB',
                 n_vectors, n_dims, stride, vec_mb::NUMERIC(18,1), ids_mb::NUMERIC(18,1), graph_mb::NUMERIC(18,1), sq8_mb::NUMERIC(18,1);
    RAISE NOTICE 'vvector.sizing: snapshot and cache file % MB per node; build memory about % MB on the refreshing node (fenced: counts against FencedUDxMemoryLimitMB)',
                 total_mb::NUMERIC(18,1), build_mb::NUMERIC(18,1);
    RAISE NOTICE 'vvector.sizing: queries read the cache file through the page cache: keep it in memory. Smallest node here: % GB of memory',
                 mem_gb::NUMERIC(18,1);
    IF total_mb / 1024.0 > 0.5 * mem_gb THEN
        RAISE WARNING 'vvector.sizing: the index needs more than half of the memory of the smallest node. Use quantization sq8 with memory_mode compact, or larger nodes.';
    END IF;
END;
$$;

-- How many journal rows a query has to apply now and how long it takes to read them, open writers,
-- versions in the future, and the sizing check (index bytes against node memory, build memory against
-- FencedUDxMemoryLimitMB and free memory, threads against cores, all indexes against the page cache).
-- It recommends; it never changes anything.
CREATE OR REPLACE PROCEDURE vvector.status(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); ver VARCHAR(128); n INT; t0 TIMESTAMPTZ; ms INT; sch VARCHAR(128); tbl VARCHAR(128);
    bytes INT; all_bytes INT; vectors INT; mem INT; free_mem INT; cores INT; fenced_mb INT; thr INT; build INT;
    rmode VARCHAR(16); tomb INT; kind VARCHAR(16); ver_type VARCHAR(128); live INT; ratio FLOAT; since INT; every INT;
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.status: index % is not registered', nm;
    END IF;
    sch := SPLIT_PART(tab, '.', 1);
    tbl := SPLIT_PART(tab, '.', 2);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    kind := (SELECT MAX(index_type) FROM vvector.manifest WHERE index_name = nm);
    rmode := (SELECT MAX(refresh_mode) FROM vvector.manifest WHERE index_name = nm);
    tomb := (SELECT COALESCE(MAX(tombstones), 0) FROM vvector.manifest WHERE index_name = nm);
    live := (SELECT COALESCE(MAX(vector_count), 0) FROM vvector.manifest WHERE index_name = nm);
    ratio := (SELECT COALESCE(MAX(tombstone_ratio), 0.2) FROM vvector.manifest WHERE index_name = nm);
    since := (SELECT COALESCE(MAX(incremental_count), 0) FROM vvector.manifest WHERE index_name = nm);
    every := (SELECT MAX(rebuild_every) FROM vvector.manifest WHERE index_name = nm);
    RAISE NOTICE 'vvector: index %: % index, % live vectors, % tombstones (tombstone_ratio %), refresh_mode %, % incremental refreshes since the last full build (rebuild_every %)',
                 nm, kind, live, tomb, ratio, COALESCE(rmode, 'auto'), since, COALESCE(every::VARCHAR, 'never');
    IF ver IS NOT NULL THEN
        RAISE NOTICE 'vvector: index %: journal digest verified %; % refreshes since the last verification or full build', nm,
                     (SELECT CASE WHEN COALESCE(MAX(verify_every), 1) = 0 THEN 'never (verify_every 0): after a physical UPDATE, DELETE or dropped partition run refresh_index(name, ''full'')'
                                  WHEN COALESCE(MAX(verify_every), 1) = 1 THEN 'at every refresh (verify_every 1)'
                                  ELSE 'every ' || MAX(verify_every) || ' refreshes (verify_every ' || MAX(verify_every) || ')' END
                      FROM vvector.manifest WHERE index_name = nm),
                     (SELECT COALESCE(MAX(refreshes_since_verify), 0) FROM vvector.manifest WHERE index_name = nm);
    END IF;
    RAISE NOTICE 'vvector: index %: last refresh: %', nm,
                 (SELECT COALESCE(MAX(refresh_note), 'none yet') FROM vvector.manifest WHERE index_name = nm);
    IF (SELECT COUNT(refresh_started_at) FROM vvector.manifest WHERE index_name = nm) > 0 THEN
        RAISE NOTICE 'vvector: index %: a refresh is running since % (by %)%', nm,
                     (SELECT TO_CHAR(MAX(refresh_started_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') || ' UTC' FROM vvector.manifest WHERE index_name = nm),
                     (SELECT MAX(refresh_started_by) FROM vvector.manifest WHERE index_name = nm),
                     (SELECT CASE WHEN MAX(refresh_started_at) < CLOCK_TIMESTAMP() - INTERVAL '6 hours'
                                  THEN ': its mark is older than 6 hours and the next refresh ignores it' ELSE '' END
                      FROM vvector.manifest WHERE index_name = nm);
    END IF;
    IF tomb > ratio * (live + tomb) AND COALESCE(rmode, 'auto') = 'incremental' THEN
        RAISE WARNING 'vvector: index %: % of % positions are tombstones: searches pass through them. refresh_mode incremental never rebuilds by itself: run CALL vvector.refresh_index(''%'', ''full'') when the node is quiet.', nm, tomb, live + tomb, nm;
    END IF;
    n := 0;
    IF ver IS NOT NULL THEN
        t0 := (SELECT CLOCK_TIMESTAMP());
        n := EXECUTE 'SELECT COUNT(id) FROM (SELECT id, vec, del, ver FROM ' || sch || '.' || nm || '_delta) d';
        ms := (SELECT DATEDIFF('millisecond', t0, CLOCK_TIMESTAMP()));
        RAISE NOTICE 'vvector: index %: % journal rows in the delta, read in % ms', nm, n, ms;
        IF ms > 500 THEN
            RAISE WARNING 'vvector: index %: reading the delta is slow (% ms) and every exact query pays for it. Usual cause: the journal is not partitioned by the date of its version column, so new rows were merged into old storage. Fix: partition the journal by the version date, or refresh more often.', nm, ms;
        END IF;
        IF n > 100000 THEN
            RAISE WARNING 'vvector: index %: the delta holds % rows. Every exact query searches them: refresh more often.', nm, n;
        END IF;
        RAISE NOTICE 'vvector: index %: journal replica %: %', nm,
                     (SELECT COALESCE(MAX(journal_replica), 'auto') FROM vvector.manifest WHERE index_name = nm),
                     (SELECT COALESCE(MAX(replica_note), 'not checked yet (the next refresh does it)') FROM vvector.manifest WHERE index_name = nm);
    END IF;
    n := (SELECT COUNT(DISTINCT transaction_id) FROM v_monitor.locks WHERE LOWER(object_name) = LOWER('Table:' || tab)
          AND (lock_mode ILIKE '%I%' OR lock_mode = 'X'));
    IF n > 0 THEN
        ms := (SELECT DATEDIFF('second', MIN(request_timestamp), CLOCK_TIMESTAMP()) FROM v_monitor.locks
               WHERE LOWER(object_name) = LOWER('Table:' || tab) AND (lock_mode ILIKE '%I%' OR lock_mode = 'X'));
        RAISE NOTICE 'vvector: index %: % open transactions are writing to %, the oldest for % seconds. A refresh now keeps their rows in the delta.', nm, n, tab, ms;
    END IF;
    -- A version that lies in the future was not set by the database clock: some writer fills the
    -- version column itself. (A version set too far in the past cannot be recognised afterwards.)
    ver_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                 AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(ver));
    IF ver IS NOT NULL AND ver_type ILIKE 'timestamp%' THEN
        n := EXECUTE 'SELECT COUNT(*) FROM ' || tab || ' WHERE ' || ver || ' > CLOCK_TIMESTAMP() + INTERVAL ''5 minutes''';
        IF n > 0 THEN
            RAISE WARNING 'vvector: index %: % rows of % have a version in the future. The version column must be filled by its default (CLOCK_TIMESTAMP), never by the application; results can be wrong.', nm, n, tab;
        END IF;
    END IF;

    -- Sizing check.
    bytes := (SELECT MAX(index_bytes) FROM vvector.manifest WHERE index_name = nm);
    vectors := (SELECT COALESCE(MAX(vector_count), 0) + COALESCE(MAX(tombstones), 0) FROM vvector.manifest WHERE index_name = nm);
    IF bytes IS NULL THEN
        RAISE NOTICE 'vvector: index %: no snapshot yet, no sizing check. For an estimate: CALL vvector.sizing(vectors, dims, index_type, quantization)', nm;
        RETURN;
    END IF;
    all_bytes := (SELECT SUM(index_bytes) FROM vvector.manifest);
    mem := (SELECT MIN(total_memory_bytes) FROM v_monitor.host_resources);
    free_mem := (SELECT MIN(total_memory_free_bytes + total_memory_cache_bytes) FROM v_monitor.host_resources);
    cores := (SELECT MIN(processor_core_count) FROM v_monitor.host_resources);
    fenced_mb := (SELECT MAX(current_value::INT) FROM v_monitor.configuration_parameters WHERE parameter_name = 'FencedUDxMemoryLimitMB');
    thr := (SELECT MAX(threads_default) FROM vvector.manifest WHERE index_name = nm);
    build := bytes + 4 * vectors + CASE WHEN kind = 'hnsw' THEN vectors * (5 + 2 * cores) ELSE 0 END;
    RAISE NOTICE 'vvector: index %: sizing: index % MB, all indexes % MB, build about % MB, smallest node % MB of memory (% MB free or cache), % cores',
                 nm, bytes // 1048576, all_bytes // 1048576, build // 1048576, mem // 1048576, free_mem // 1048576, cores;
    IF bytes > mem // 2 THEN
        RAISE WARNING 'vvector: index %: the index takes more than half of the memory of the smallest node. Use quantization sq8 with memory_mode compact, or larger nodes.', nm;
    ELSIF all_bytes > mem * 7 // 10 THEN
        RAISE WARNING 'vvector: index %: all indexes together take % MB, more than 70%% of the memory of the smallest node: they will not stay in the page cache. Use quantization sq8 with memory_mode compact, fewer indexes, or larger nodes.', nm, all_bytes // 1048576;
    END IF;
    IF fenced_mb > 0 AND build > fenced_mb * 1048576 THEN
        RAISE WARNING 'vvector: index %: a refresh needs about % MB in the fenced process, FencedUDxMemoryLimitMB is %: raise FencedUDxMemoryLimitMB (vbuild runs fenced with FENCED=yes and FENCED=mixed).', nm, build // 1048576, fenced_mb;
    END IF;
    IF build > free_mem THEN
        RAISE WARNING 'vvector: index %: a refresh needs about % MB, the node has % MB free or in the page cache. Refresh when the node is quiet, or use larger nodes.', nm, build // 1048576, free_mem // 1048576;
    END IF;
    IF thr IS NOT NULL AND thr > cores THEN
        RAISE WARNING 'vvector: index %: threads_default % is more than the % cores of the smallest node: set it to 0 (one per core) or lower.', nm, thr, cores;
    END IF;
END;
$$;

-- Refresh order, never changed: boundary -> build -> insert chunks -> vload on all nodes -> update
-- manifest and views -> delete older snapshots.
-- Mode (x_mode, else the manifest's refresh_mode, else auto):
--   full         builds from every row of the journal
--   incremental  builds from the active snapshot in the node cache and the journal rows after the
--                previous delta boundary; full only when it cannot be done (below)
--   auto         incremental, and full as well when the tombstones pass tombstone_ratio or after
--                rebuild_every incremental refreshes
-- A full build is always made for the first build, a static index (no version column), changed
-- build options (index_type, metric, m, ef_construction, quantization) or library format, a node
-- whose cache does not hold the active snapshot, and when the rows up to the previous boundary are
-- not the rows the last refresh saw (a physical DELETE or UPDATE, dropped partitions, versions set
-- by hand): their count and digest, carried forward from refresh to refresh, are recomputed and
-- compared every verify_every refreshes.
-- The outcome goes to manifest refresh_note: NOTICEs of a nested CALL do not reach the caller.
-- refresh_index_core runs this with the guard against a second refresh of the same index.
CREATE OR REPLACE PROCEDURE vvector.refresh_index_run(nm VARCHAR, x_mode VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256); idc VARCHAR(128); vc VARCHAR(128); op VARCHAR(128); ver VARCHAR(128); sch VARCHAR(128); tbl VARCHAR(128);
    measure VARCHAR(16); kind VARCHAR(16); quant VARCHAR(16); hm INT; hefc INT; margin INT;
    op_type VARCHAR(128); ver_type VARCHAR(128); vc_type VARCHAR(128);
    prev INT; sid INT; chunks INT; max_ver INT; v_from VARCHAR(64); prev_from VARCHAR(64);
    source VARCHAR(4000); del_expr VARCHAR(400); v_expr VARCHAR(400);
    t0 TIMESTAMPTZ; cut TIMESTAMPTZ; secs FLOAT; n_vec INT; n_dims INT; fmt INT; n_bytes INT; n_graph INT; n_tomb INT;
    rmode VARCHAR(16); ratio FLOAT; every INT; since INT; opts VARCHAR(200); prev_opts VARCHAR(200); prev_fmt INT;
    prev_vec INT; prev_tomb INT; prev_max INT; rows_then INT; rows_now INT; rows_next INT; want INT; got INT; counts VARCHAR(200);
    digest_then VARCHAR(64); digest_now VARCHAR(64); digest_next VARCHAR(64); h_expr VARCHAR(600);
    v_every INT; v_since INT; verify BOOLEAN; t_scan TIMESTAMPTZ; scan_secs FLOAT; j_note VARCHAR(300);
    why VARCHAR(600); r_note VARCHAR(2400); build_opts VARCHAR(400); w_since TIMESTAMPTZ; lag_note VARCHAR(600); young INT;
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.refresh_index: index % is not registered', nm;
    END IF;
    IF x_mode IS NOT NULL AND x_mode NOT IN ('auto', 'incremental', 'full') THEN
        RAISE EXCEPTION 'vvector.refresh_index: mode must be auto, incremental or full';
    END IF;
    sch := SPLIT_PART(tab, '.', 1);
    tbl := SPLIT_PART(tab, '.', 2);
    idc := (SELECT id_col FROM vvector.manifest WHERE index_name = nm);
    vc := (SELECT vec_col FROM vvector.manifest WHERE index_name = nm);
    op := (SELECT op_col FROM vvector.manifest WHERE index_name = nm);
    ver := (SELECT ver_col FROM vvector.manifest WHERE index_name = nm);
    measure := (SELECT m.metric FROM vvector.manifest m WHERE m.index_name = nm);
    kind := (SELECT m.index_type FROM vvector.manifest m WHERE m.index_name = nm);
    quant := (SELECT COALESCE(m.quantization, 'none') FROM vvector.manifest m WHERE m.index_name = nm);
    hm := (SELECT COALESCE(m.hnsw_m, 16) FROM vvector.manifest m WHERE m.index_name = nm);
    hefc := (SELECT COALESCE(m.hnsw_ef_construction, 200) FROM vvector.manifest m WHERE m.index_name = nm);
    margin := (SELECT ver_margin FROM vvector.manifest WHERE index_name = nm);
    prev := (SELECT active_snapshot FROM vvector.manifest WHERE index_name = nm);
    prev_from := (SELECT m.delta_from FROM vvector.manifest m WHERE m.index_name = nm);
    rmode := (SELECT COALESCE(x_mode, MAX(m.refresh_mode), 'auto') FROM vvector.manifest m WHERE m.index_name = nm);
    ratio := (SELECT COALESCE(MAX(m.tombstone_ratio), 0.2) FROM vvector.manifest m WHERE m.index_name = nm);
    every := (SELECT MAX(m.rebuild_every) FROM vvector.manifest m WHERE m.index_name = nm);
    since := (SELECT COALESCE(MAX(m.incremental_count), 0) FROM vvector.manifest m WHERE m.index_name = nm);
    prev_opts := (SELECT MAX(m.active_options) FROM vvector.manifest m WHERE m.index_name = nm);
    prev_fmt := (SELECT MAX(m.format_version) FROM vvector.manifest m WHERE m.index_name = nm);
    prev_vec := (SELECT COALESCE(MAX(m.vector_count), 0) FROM vvector.manifest m WHERE m.index_name = nm);
    prev_tomb := (SELECT COALESCE(MAX(m.tombstones), 0) FROM vvector.manifest m WHERE m.index_name = nm);
    rows_then := (SELECT MAX(m.boundary_rows) FROM vvector.manifest m WHERE m.index_name = nm);
    digest_then := (SELECT MAX(m.boundary_digest)::VARCHAR FROM vvector.manifest m WHERE m.index_name = nm);   -- digests are compared as text: PL/vSQL declares no NUMERIC(38,0)
    prev_max := (SELECT MAX(m.active_max_ver) FROM vvector.manifest m WHERE m.index_name = nm);
    v_every := (SELECT COALESCE(MAX(m.verify_every), 1) FROM vvector.manifest m WHERE m.index_name = nm);
    v_since := (SELECT COALESCE(MAX(m.refreshes_since_verify), 0) FROM vvector.manifest m WHERE m.index_name = nm);
    fmt := (SELECT format_version FROM (SELECT vvector.vversion() OVER()) v);
    -- The options a snapshot is built with. A snapshot can only be continued with the same ones.
    opts := kind || ' ' || measure || ' ' || quant || CASE WHEN kind = 'hnsw' THEN ' m=' || hm || ' ef_construction=' || hefc ELSE '' END;
    vc_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(vc));
    v_expr := CASE WHEN vc_type ILIKE 'array[float8%' THEN vc ELSE vc || '::ARRAY[FLOAT]' END;
    IF op IS NULL THEN
        del_expr := 'FALSE';
    ELSE
        op_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                    AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(op));
        del_expr := CASE WHEN op_type ILIKE 'bool%' THEN 'COALESCE(' || op || ', FALSE)' ELSE '(COALESCE(' || op || ', 1) < 0)' END;
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
    --    The snapshot holds exactly the rows up to the boundary (the rows the digest below covers);
    --    the rows after it are served by the delta view until the next refresh.
    --    The highest version (the watermark in the manifest and the snapshot header; for an INT version
    --    also the boundary) is read from the rows after the previous boundary only: the rows before it
    --    were there at the last refresh, which recorded their highest version.
    max_ver := 0;
    v_from := NULL;
    IF ver IS NOT NULL THEN
        ver_type := (SELECT MAX(data_type) FROM v_catalog.columns WHERE LOWER(table_schema) = LOWER(sch)
                     AND LOWER(table_name) = LOWER(tbl) AND LOWER(column_name) = LOWER(ver));
        IF ver_type ILIKE 'int%' THEN
            max_ver := EXECUTE 'SELECT MAX(' || ver || ')::INT FROM ' || tab
                || CASE WHEN prev_from IS NULL OR prev_max IS NULL THEN '' ELSE ' WHERE ' || ver || ' > ' || prev_from END;
            max_ver := GREATEST(COALESCE(max_ver, 0), COALESCE(prev_max, 0));
            v_from := (max_ver - margin)::VARCHAR;
        ELSE
            max_ver := EXECUTE 'SELECT MAX((EXTRACT(EPOCH FROM ' || ver || ') * 1000000)::INT) FROM ' || tab
                || CASE WHEN prev_from IS NULL OR prev_max IS NULL THEN '' ELSE ' WHERE ' || ver || ' > ' || prev_from END;
            max_ver := GREATEST(COALESCE(max_ver, 0), COALESCE(prev_max, 0));
            w_since := (SELECT MIN(request_timestamp) FROM v_monitor.locks
                        WHERE LOWER(object_name) = LOWER('Table:' || tab)
                          AND (lock_mode ILIKE '%I%' OR lock_mode = 'X')
                          AND transaction_id <> (SELECT transaction_id FROM v_monitor.current_session));
            cut := (SELECT LEAST(CLOCK_TIMESTAMP(), COALESCE(w_since, CLOCK_TIMESTAMP())) - (margin // 1000000) * INTERVAL '1 second');
            -- An open writer holds the boundary back: everything after its start stays in the delta.
            IF w_since IS NOT NULL AND DATEDIFF('second', w_since, CLOCK_TIMESTAMP()) > GREATEST(10 * (margin // 1000000), 60) THEN
                lag_note := 'the boundary lags ' || DATEDIFF('minute', w_since, CLOCK_TIMESTAMP()) || ' minutes behind: a transaction has been writing to '
                         || tab || ' since ' || TO_CHAR(w_since AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') || ' UTC (v_monitor.locks); its rows and all later rows stay in the delta until it ends';
            END IF;
            IF ver_type ILIKE 'timestamptz%' OR ver_type ILIKE '%with time zone%' THEN
                v_from := (SELECT 'TIMESTAMPTZ ''' || TO_CHAR(cut AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.US') || '+00''');
            ELSE
                -- Plain TIMESTAMP versions are local times of the writing session: all writers and this
                -- session must use the same time zone.
                v_from := (SELECT 'TIMESTAMP ''' || TO_CHAR(cut::TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS.US') || '''');
            END IF;
        END IF;
        max_ver := COALESCE(max_ver, 0);
    END IF;

    -- 2. Full or incremental. why = the reason for a full build.
    why := NULL;
    IF rmode = 'full' THEN
        why := 'mode full';
    ELSIF prev IS NULL THEN
        why := 'first build';
    ELSIF ver IS NULL THEN
        why := 'a static index (no version column) is always built in full';
    ELSIF prev_from IS NULL OR rows_then IS NULL OR prev_opts IS NULL THEN
        why := 'the active snapshot was made by an older vvector version';
    ELSIF prev_opts <> opts THEN
        why := 'build options changed from ' || prev_opts || ' to ' || opts;
    ELSIF COALESCE(prev_fmt, 0) <> fmt THEN
        why := 'the library writes format version ' || fmt || ', the active snapshot has ' || COALESCE(prev_fmt, 0);
    ELSIF rmode = 'auto' AND prev_tomb > ratio * (prev_vec + prev_tomb) THEN
        why := prev_tomb || ' tombstones in ' || (prev_vec + prev_tomb) || ' positions, more than tombstone_ratio ' || ratio;
    ELSIF rmode = 'auto' AND every IS NOT NULL AND since >= every THEN
        why := since || ' incremental refreshes since the last full build (rebuild_every ' || every || ')';
    END IF;
    IF why IS NULL THEN
        want := (SELECT COUNT(*) FROM (SELECT vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe) n);
        got := EXECUTE 'SELECT COUNT(DISTINCT node_name) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm)
            || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE loaded AND snapshot_id = ' || prev;
        IF COALESCE(got, 0) < want THEN
            why := 'the active snapshot ' || prev || ' is in the cache of ' || COALESCE(got, 0) || ' of ' || want || ' nodes';
        END IF;
    END IF;
    -- The journal rows up to the previous boundary must be the rows the last refresh saw: a physical
    -- DELETE or UPDATE, a dropped partition, or a row written with an old version by hand changes
    -- them, and then the build must be full. The manifest keeps their count and digest (the sum of
    -- HASH over the columns vvector reads: id, vector, delete flag, version). They are carried forward
    -- from the journal rows between the previous and the new boundary (all rows, before the
    -- consolidation per id), which a refresh reads anyway. The verification recomputes both over every
    -- row up to the previous boundary, the vectors included, and compares: at every refresh with
    -- verify_every 1 (the default), every N refreshes with N, never with 0. A full build stores exact
    -- values (one scan up to the new boundary). An index without a digest yet gets it by one scan,
    -- which also checks the count.
    rows_next := NULL;
    digest_next := NULL;
    j_note := NULL;
    verify := FALSE;
    IF ver IS NOT NULL THEN
        h_expr := 'HASH(' || idc || ', ' || vc || CASE WHEN op IS NULL THEN '' ELSE ', ' || op END || ', ' || ver || ')::NUMERIC(38,0)';
    END IF;
    IF why IS NULL THEN
        verify := digest_then IS NULL OR (v_every > 0 AND v_since + 1 >= v_every);
        t_scan := (SELECT CLOCK_TIMESTAMP());
        IF verify THEN
            counts := EXECUTE 'SELECT /*+LABEL(vvector_verify)*/ COUNT(CASE WHEN jv <= ' || prev_from || ' THEN 1 END) || '' '' || COALESCE(SUM(CASE WHEN jv <= '
                || prev_from || ' THEN jh END), 0) || '' '' || COUNT(CASE WHEN jv <= ' || v_from || ' THEN 1 END) || '' '' || COALESCE(SUM(CASE WHEN jv <= '
                || v_from || ' THEN jh END), 0) FROM (SELECT ' || ver || ' AS jv, ' || h_expr || ' AS jh FROM ' || tab
                || ' WHERE ' || ver || ' <= ' || prev_from || ' OR ' || ver || ' <= ' || v_from || ') j';
            rows_now := SPLIT_PART(counts, ' ', 1)::INT;
            digest_now := SPLIT_PART(counts, ' ', 2);
            rows_next := SPLIT_PART(counts, ' ', 3)::INT;
            digest_next := SPLIT_PART(counts, ' ', 4);
        ELSE
            counts := EXECUTE 'SELECT /*+LABEL(vvector_digest)*/ (' || rows_then || ' + COUNT(*)) || '' '' || (' || digest_then
                || '::NUMERIC(38,0) + COALESCE(SUM(' || h_expr || '), 0)) FROM ' || tab
                || ' WHERE ' || ver || ' > ' || prev_from || ' AND ' || ver || ' <= ' || v_from;
            rows_next := SPLIT_PART(counts, ' ', 1)::INT;
            digest_next := SPLIT_PART(counts, ' ', 2);
        END IF;
        scan_secs := (SELECT DATEDIFF('millisecond', t_scan, CLOCK_TIMESTAMP()) / 1000.0);
        IF verify AND rows_now <> rows_then THEN
            why := 'the journal has ' || rows_now || ' rows up to the previous boundary, the last refresh counted ' || rows_then
                || ' (a physical DELETE or UPDATE, dropped partitions, or versions set by hand)';
        ELSIF verify AND digest_then IS NOT NULL AND digest_now <> COALESCE(digest_then, '') THEN
            why := 'the journal rows up to the previous boundary changed since the last refresh: same count (' || rows_now
                || '), other digest (a physical UPDATE, or rows replaced by hand)';
        END IF;
        IF verify AND digest_then IS NULL THEN
            j_note := 'journal digest taken for the first time in ' || scan_secs || ' seconds (row count verified)';
        ELSIF verify THEN
            j_note := 'journal verified in ' || scan_secs || ' seconds';
        ELSIF v_every = 0 THEN
            j_note := 'journal not verified (verify_every 0)';
        ELSE
            j_note := 'journal not verified (verify_every ' || v_every || ', last verified ' || (v_since + 1) || ' refreshes ago)';
        END IF;
    ELSIF ver IS NOT NULL THEN
        t_scan := (SELECT CLOCK_TIMESTAMP());
        counts := EXECUTE 'SELECT /*+LABEL(vvector_digest)*/ COUNT(*) || '' '' || COALESCE(SUM(' || h_expr || '), 0) FROM ' || tab
            || ' WHERE ' || ver || ' <= ' || v_from;
        rows_next := SPLIT_PART(counts, ' ', 1)::INT;
        digest_next := SPLIT_PART(counts, ' ', 2);
        verify := TRUE;
        scan_secs := (SELECT DATEDIFF('millisecond', t_scan, CLOCK_TIMESTAMP()) / 1000.0);
        j_note := 'journal digest taken in ' || scan_secs || ' seconds';
    END IF;

    -- 3. The rows to build from.
    --    Full: the latest row of every id by version, kept if it is not a delete. Two rows of one id
    --    with the same version: the delete wins. Without a version column every id must appear once
    --    (vbuild refuses a repeated id).
    --    Incremental: the latest row of every id after the previous boundary, deletes included.
    --    Both read the rows up to the new boundary only: the snapshot holds exactly the rows the digest
    --    covers, so a physical DELETE or UPDATE of any row in it is found by the next verification.
    IF why IS NULL THEN
        source := 'SELECT id, vec, del FROM (SELECT ' || idc || '::INT AS id, ' || v_expr || ' AS vec, ' || del_expr || ' AS del, '
               || 'ROW_NUMBER() OVER(PARTITION BY ' || idc || ' ORDER BY ' || ver || ' DESC, ' || del_expr || ' DESC) AS rn FROM ' || tab
               || ' WHERE ' || idc || ' IS NOT NULL AND ' || ver || ' > ' || prev_from || ' AND ' || ver || ' <= ' || v_from || ') j WHERE rn = 1';
    ELSIF ver IS NULL THEN
        source := 'SELECT ' || idc || '::INT AS id, ' || v_expr || ' AS vec, FALSE AS del FROM ' || tab || ' WHERE ' || idc || ' IS NOT NULL';
    ELSE
        source := 'SELECT id, vec, FALSE AS del FROM (SELECT ' || idc || '::INT AS id, ' || v_expr || ' AS vec, ' || del_expr || ' AS del, '
               || 'ROW_NUMBER() OVER(PARTITION BY ' || idc || ' ORDER BY ' || ver || ' DESC, ' || del_expr || ' DESC) AS rn FROM ' || tab
               || ' WHERE ' || idc || ' IS NOT NULL AND ' || ver || ' <= ' || v_from || ') j WHERE rn = 1 AND NOT del';
    END IF;

    -- 4. Build and store the chunks. vbuild sorts by id itself: no ORDER BY, no sort of the table.
    --    Snapshot ids come from a sequence: they never repeat, also not after unregister and register,
    --    so a cache file left behind by an older index of the same name is always recognised as stale.
    sid := (SELECT NEXTVAL('vvector.snapshot_seq'));
    build_opts := 'index_name=' || QUOTE_LITERAL(nm) || ', metric=' || QUOTE_LITERAL(measure) || ', index_type=' || QUOTE_LITERAL(kind)
               || ', quantization=' || QUOTE_LITERAL(quant) || ', m=' || hm || ', ef_construction=' || hefc || ', max_ver=' || max_ver
               || CASE WHEN why IS NULL THEN ', base_snapshot=' || prev ELSE '' END;
    EXECUTE 'INSERT /*+LABEL(vvector_build)*/ INTO vvector.snapshot SELECT ' || QUOTE_LITERAL(nm) || ', ' || sid
         || ', byte_offset, chunk FROM (SELECT vvector_admin.vbuild(id, vec, del USING PARAMETERS ' || build_opts
         || ') OVER() FROM (' || source || ') e) b';
    PERFORM COMMIT;
    chunks := (SELECT COUNT(*) FROM vvector.snapshot WHERE index_name = nm AND snapshot_id = sid);
    IF chunks = 0 AND why IS NOT NULL AND ver IS NOT NULL THEN
        young := EXECUTE 'SELECT COUNT(*) FROM ' || tab || ' WHERE ' || idc || ' IS NOT NULL AND ' || ver || ' > ' || v_from;
        IF young > 0 THEN
            RAISE EXCEPTION 'vvector.refresh_index: index %: no live vector up to the delta boundary % (the margin before now or before the oldest open writer); % rows of % are newer: queries with freshness=''exact'' find them through the delta view. Refresh again when the margin has passed, or register the index with a smaller margin',
                            nm, v_from, young, tab;
        END IF;
    END IF;
    IF chunks = 0 AND why IS NOT NULL THEN
        RAISE EXCEPTION 'vvector.refresh_index: index %: table % has no vectors, nothing to build', nm, tab;
    END IF;

    IF chunks = 0 THEN
        -- Incremental and no vector changed: the active snapshot holds the live vectors up to the new
        -- boundary. Only the boundary moves; nothing is loaded.
        secs := (SELECT DATEDIFF('millisecond', t0, CLOCK_TIMESTAMP()) / 1000.0);
        r_note := 'refreshed: snapshot ' || prev || ' kept, no vector changed since it was built; the delta starts at the new boundary; '
               || secs || ' seconds' || COALESCE('; ' || j_note, '') || COALESCE('; ' || lag_note, '');
        PERFORM UPDATE vvector.manifest SET active_max_ver = max_ver, delta_from = v_from, boundary_rows = rows_next,
                       boundary_digest = digest_next::NUMERIC(38,0), refreshes_since_verify = CASE WHEN verify THEN 0 ELSE v_since + 1 END, built_at = CLOCK_TIMESTAMP(), build_seconds = secs, refresh_note = LEFT(r_note, 1000)
                WHERE index_name = nm;
        PERFORM COMMIT;
        PERFORM CALL vvector.make_views(nm);
    ELSE
        -- 5. Load on every node, with the index defaults. Until the manifest and the views change,
        --    queries keep using the previous snapshot's views.
        PERFORM CALL vvector.load_on_nodes(nm, sid);
        PERFORM CALL vvector.push_options(nm);

        -- 6. Manifest and views.
        n_vec := EXECUTE 'SELECT MAX(vector_count) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
        n_dims := EXECUTE 'SELECT MAX(dims) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
        n_graph := EXECUTE 'SELECT MAX(graph_bytes) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
        n_tomb := EXECUTE 'SELECT MAX(tombstones) FROM (SELECT vvector.vinfo(USING PARAMETERS index_name=' || QUOTE_LITERAL(nm) || ') OVER(PARTITION NODES) FROM vvector.probe) i WHERE snapshot_id = ' || sid;
        -- The size: offset and length of the last chunk (reading every chunk would take seconds).
        n_bytes := (SELECT MAX(byte_offset) FROM vvector.snapshot WHERE index_name = nm AND snapshot_id = sid);
        n_bytes := n_bytes + (SELECT MAX(OCTET_LENGTH(chunk)) FROM vvector.snapshot WHERE index_name = nm AND snapshot_id = sid AND byte_offset = n_bytes);
        secs := (SELECT DATEDIFF('millisecond', t0, CLOCK_TIMESTAMP()) / 1000.0);
        IF why IS NULL THEN
            r_note := 'refreshed: snapshot ' || sid || ', incremental from snapshot ' || prev || ' (' || ((n_vec + n_tomb) - (prev_vec + prev_tomb))
                   || ' vectors appended, ' || (n_tomb - prev_tomb) || ' tombstoned), ';
        ELSE
            r_note := 'refreshed: snapshot ' || sid || ', full build (' || why || '), ';
        END IF;
        r_note := r_note || n_vec || ' vectors of ' || n_dims || ' dimensions, ' || n_tomb || ' tombstones, '
               || n_bytes // 1048576 || ' MB, ' || secs || ' seconds' || COALESCE('; ' || j_note, '') || COALESCE('; ' || lag_note, '');
        PERFORM UPDATE vvector.manifest SET active_snapshot = sid, active_max_ver = max_ver, delta_from = v_from,
                       base_snapshot = CASE WHEN why IS NULL THEN prev ELSE 0 END, vector_count = n_vec, dims = n_dims, tombstones = n_tomb,
                       graph_bytes = n_graph, index_bytes = n_bytes, built_at = CLOCK_TIMESTAMP(), build_seconds = secs, format_version = fmt,
                       active_options = opts, incremental_count = CASE WHEN why IS NULL THEN since + 1 ELSE 0 END,
                       boundary_rows = rows_next, boundary_digest = digest_next::NUMERIC(38,0),
                       refreshes_since_verify = CASE WHEN verify OR why IS NOT NULL THEN 0 ELSE v_since + 1 END, refresh_note = LEFT(r_note, 1000)
                WHERE index_name = nm;
        PERFORM COMMIT;
        PERFORM CALL vvector.make_views(nm);

        -- 7. Keep the active and the previous snapshot.
        PERFORM DELETE FROM vvector.snapshot WHERE index_name = nm AND snapshot_id < COALESCE(prev, sid);
        PERFORM COMMIT;
    END IF;

    -- 8. Journal replica: made, kept (with fresh statistics) or dropped as the journal grows.
    IF ver IS NOT NULL THEN
        PERFORM CALL vvector.apply_replica(nm);
    END IF;
END;
$$;

-- At most one refresh of an index at a time (two would load, record and delete each other's
-- snapshots). A refresh marks the manifest row (refresh_started_at, refresh_started_by) with one
-- conditional UPDATE: the UPDATE of a second refresh waits for the lock on the manifest, then finds
-- the mark and changes nothing, and that refresh stops with an error. The mark is removed when the
-- refresh ends, also when it fails. A mark left behind by a refresh that could not remove it (its
-- session was killed, its node went down) is ignored at once when its session is gone from
-- v_monitor.sessions, which a superuser sees completely and another user for its own sessions only
-- (so that check covers the marks of the caller's own sessions). A refresh run by a schedule trigger
-- has a session that v_monitor.sessions never shows (VERTICA_NOTES): its mark says "scheduled" and is
-- never judged by its session. Otherwise a mark is ignored after 6 hours or 4 times the index's last
-- build time, whichever is longer, or removed by hand (the error message gives the statement).
CREATE OR REPLACE PROCEDURE vvector.refresh_index_core(nm VARCHAR, x_mode VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    me VARCHAR(200); t_start TIMESTAMPTZ; holder VARCHAR(200); since TIMESTAMPTZ; gone BOOLEAN; keep_s INT; my_session VARCHAR(200);
BEGIN
    IF (SELECT COUNT(*) FROM vvector.manifest WHERE index_name = nm) = 0 THEN
        RAISE EXCEPTION 'vvector.refresh_index: index % is not registered', nm;
    END IF;
    IF x_mode IS NOT NULL AND x_mode NOT IN ('auto', 'incremental', 'full') THEN
        RAISE EXCEPTION 'vvector.refresh_index: mode must be auto, incremental or full';
    END IF;
    my_session := (SELECT session_id FROM v_monitor.current_session);
    IF (SELECT COUNT(*) FROM v_monitor.sessions WHERE session_id = my_session) > 0 THEN
        me := CURRENT_USER() || ', session ' || my_session;
    ELSE
        me := CURRENT_USER() || ', scheduled, internal session ' || my_session;
    END IF;
    t_start := (SELECT CLOCK_TIMESTAMP());
    holder := (SELECT MAX(refresh_started_by) FROM vvector.manifest WHERE index_name = nm AND refresh_started_at IS NOT NULL);
    gone := FALSE;
    IF holder IS NOT NULL AND POSITION(', session ' IN holder) > 0 THEN
        IF (SELECT COUNT(*) FROM v_catalog.users WHERE user_name = CURRENT_USER() AND is_super_user) > 0
           OR SPLIT_PART(holder, ', session ', 1) = CURRENT_USER() THEN
            gone := (SELECT COUNT(*) FROM v_monitor.sessions WHERE session_id = SPLIT_PART(holder, ', session ', 2)) = 0;
        END IF;
    END IF;
    keep_s := (SELECT GREATEST(21600, 4 * COALESCE(MAX(build_seconds), 0))::INT FROM vvector.manifest WHERE index_name = nm);
    PERFORM UPDATE vvector.manifest SET refresh_started_at = t_start, refresh_started_by = me
            WHERE index_name = nm AND (refresh_started_at IS NULL OR refresh_started_at < t_start - keep_s * INTERVAL '1 second'
                                       OR (gone AND refresh_started_by = holder));
    PERFORM COMMIT;
    since := (SELECT MAX(refresh_started_at) FROM vvector.manifest WHERE index_name = nm);
    holder := (SELECT MAX(refresh_started_by) FROM vvector.manifest WHERE index_name = nm);
    IF since IS NULL OR since <> t_start OR COALESCE(holder, '') <> me THEN
        RAISE EXCEPTION 'vvector.refresh_index: index % is being refreshed since % (by %). Two refreshes of one index cannot run at the same time: wait until it ends. If it no longer runs (its session was killed or its node went down), the mark is ignored as soon as its session is gone (seen by a superuser, or by the same user), else % hours after its start (6, or 4 times the last build time), or a vvector_admin removes it: UPDATE vvector.manifest SET refresh_started_at = NULL WHERE index_name = ''%''; COMMIT;',
                        nm, COALESCE(TO_CHAR(since AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') || ' UTC', '?'), COALESCE(holder, '?'),
                        ROUND(keep_s / 3600.0)::INT, nm;
    END IF;
    BEGIN
        PERFORM CALL vvector.refresh_index_run(nm, x_mode);
    EXCEPTION WHEN OTHERS THEN
        PERFORM UPDATE vvector.manifest SET refresh_started_at = NULL, refresh_started_by = NULL
                WHERE index_name = nm AND refresh_started_at = t_start;
        PERFORM COMMIT;
        RAISE EXCEPTION '%', SQLERRM;
    END;
    PERFORM UPDATE vvector.manifest SET refresh_started_at = NULL, refresh_started_by = NULL
            WHERE index_name = nm AND refresh_started_at = t_start;
    PERFORM COMMIT;
END;
$$;

-- refresh_index(index_name [, mode]): mode auto, incremental or full; NULL or left out = the index's
-- refresh_mode (set_index_options). Prints what was done and why.
CREATE OR REPLACE PROCEDURE vvector.refresh_index(nm VARCHAR, x_mode VARCHAR) LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.refresh_index_core(nm, x_mode);
    RAISE NOTICE 'vvector: index % %', nm, (SELECT MAX(refresh_note) FROM vvector.manifest WHERE index_name = nm);
    IF (SELECT COUNT(ver_col) FROM vvector.manifest WHERE index_name = nm) > 0 THEN
        RAISE NOTICE 'vvector: index %: journal replica: %', nm, (SELECT MAX(replica_note) FROM vvector.manifest WHERE index_name = nm);
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE vvector.refresh_index(nm VARCHAR) LANGUAGE PLvSQL AS $$
BEGIN
    PERFORM CALL vvector.refresh_index_core(nm, NULL);
    RAISE NOTICE 'vvector: index % %', nm, (SELECT MAX(refresh_note) FROM vvector.manifest WHERE index_name = nm);
    IF (SELECT COUNT(ver_col) FROM vvector.manifest WHERE index_name = nm) > 0 THEN
        RAISE NOTICE 'vvector: index %: journal replica: %', nm, (SELECT MAX(replica_note) FROM vvector.manifest WHERE index_name = nm);
    END IF;
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

-- Removes the schedule, the journal replica (unless another index shares it), the views, the snapshots
-- and the manifest row.
-- Cache files stay on the nodes: no vvector function deletes paths on request. Remove <cache_dir>/<index_name> by hand.
CREATE OR REPLACE PROCEDURE vvector.unregister_index(nm VARCHAR) LANGUAGE PLvSQL AS $$
DECLARE
    tab VARCHAR(256);
BEGIN
    tab := (SELECT MAX(source_table) FROM vvector.manifest WHERE index_name = nm);
    IF tab IS NULL THEN
        RAISE EXCEPTION 'vvector.unregister_index: index % is not registered', nm;
    END IF;
    -- Only a superuser may drop a trigger, even with IF EXISTS: drop them only when schedule_refresh made them.
    IF (SELECT COUNT(*) FROM v_catalog.stored_proc_triggers WHERE LOWER(schema_name) = 'vvector'
                                                          AND LOWER(trigger_name) = LOWER(nm || '_refresh_trigger')) > 0 THEN
        EXECUTE 'DROP TRIGGER IF EXISTS vvector.' || nm || '_refresh_trigger';
    END IF;
    IF (SELECT COUNT(*) FROM v_catalog.user_schedules WHERE LOWER(schema_name) = 'vvector'
                                                      AND LOWER(schedule_name) = LOWER(nm || '_refresh_schedule')) > 0 THEN
        EXECUTE 'DROP SCHEDULE IF EXISTS vvector.' || nm || '_refresh_schedule';
    END IF;
    PERFORM UPDATE vvector.manifest SET journal_replica = 'off' WHERE index_name = nm;
    PERFORM COMMIT;
    PERFORM CALL vvector.apply_replica(nm);     -- drops the replica unless another index shares it
    EXECUTE 'DROP VIEW IF EXISTS ' || SPLIT_PART(tab, '.', 1) || '.' || nm || '_delta';
    EXECUTE 'DROP VIEW IF EXISTS ' || SPLIT_PART(tab, '.', 1) || '.' || nm || '_snap';
    PERFORM DELETE FROM vvector.snapshot WHERE index_name = nm;
    PERFORM DELETE FROM vvector.manifest WHERE index_name = nm;
    PERFORM COMMIT;
    RAISE NOTICE 'vvector: index % unregistered. Cache files under <cache_dir>/% stay on the nodes.', nm, nm;
END;
$$;

-- The procedure of milestone M0 that only made the delta view. make_views replaces it.
DROP PROCEDURE IF EXISTS vvector.make_delta_view(VARCHAR);

-- install.sql grants all functions of the schema to PUBLIC; that reaches the procedures too.
-- Only sizing (an estimate from its arguments and the node memory) stays open to everyone.
REVOKE EXECUTE ON PROCEDURE vvector.make_views(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.push_options(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.apply_replica(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.set_journal_replica(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.register_index_core(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.set_index_options(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.set_index_options(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.set_index_options_core(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.load_on_nodes(VARCHAR, INT) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.refresh_index_core(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.refresh_index_run(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.load_all(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.status(VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.schedule_refresh(VARCHAR, VARCHAR) FROM PUBLIC;
REVOKE EXECUTE ON PROCEDURE vvector.unregister_index(VARCHAR) FROM PUBLIC;

GRANT EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.register_index(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.set_index_options(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.set_index_options(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.set_index_options_core(VARCHAR, VARCHAR, INT, INT, VARCHAR, VARCHAR, FLOAT, INT, VARCHAR, VARCHAR, VARCHAR, INT, INT, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.refresh_index(VARCHAR, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.set_journal_replica(VARCHAR, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.load_all(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.status(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.schedule_refresh(VARCHAR, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.unregister_index(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.sizing(INT, INT, VARCHAR, VARCHAR) TO PUBLIC;
-- The procedures above call these; Vertica checks the caller's right on every nested CALL.
GRANT EXECUTE ON PROCEDURE vvector.make_views(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.push_options(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.apply_replica(VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.register_index_core(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, INT, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.load_on_nodes(VARCHAR, INT) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.refresh_index_core(VARCHAR, VARCHAR) TO vvector_admin;
GRANT EXECUTE ON PROCEDURE vvector.refresh_index_run(VARCHAR, VARCHAR) TO vvector_admin;
