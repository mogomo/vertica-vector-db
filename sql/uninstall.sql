-- vvector uninstall. Dropping the library with CASCADE drops its functions.
\set ON_ERROR_STOP on

DROP LIBRARY IF EXISTS vvector CASCADE;

-- The snapshots, the manifest and the procedures are kept. To remove them as well:
--   DROP SCHEMA vvector CASCADE;
--   DROP ROLE vvector_admin;
-- Cache files are under <cache_dir>/<index_name>/ on every node (default /tmp/vvector).
