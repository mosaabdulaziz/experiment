#!/usr/bin/env bash
# The platform's database (TimescaleDB in the `scale` project's ui-postgres container): dump, restore, count.
# Used on the server (export of a day-run), on the laptop (backup, import) and on the server again (to give it the
# laptop's site map, stops and users before a run). Runs where docker runs; the container must be up.
#
#   scale_db.sh dump FILE            custom-format dump with owners and grants (the app connects as iris_app)
#   scale_db.sh restore FILE         replace the database with the dump (Timescale's pre/post restore procedure)
#   scale_db.sh counts               row counts of the tables a run fills, one "table count" per line
# DB=name restores into another database (to test a restore without touching the real one).
set -euo pipefail
PG="${PG_CONTAINER:-scale-ui-postgres-1}"
DB="${DB:-iris}"
psql_() { docker exec -i "$PG" psql -U iris -v ON_ERROR_STOP=1 "$@"; }

case "${1:-}" in
    dump)
        out="${2:?usage: scale_db.sh dump FILE}"
        docker exec "$PG" pg_dump -U iris -d "$DB" -Fc -Z 6 > "$out.part"
        mv "$out.part" "$out"
        echo "dumped $DB -> $out ($(du -h "$out" | cut -f1))" ;;
    restore)
        in="${2:?usage: scale_db.sh restore FILE}"
        [ -s "$in" ] || { echo "no such dump: $in" >&2; exit 1; }
        # nothing may write while the database is replaced
        docker stop "${UI_CONTAINER:-scale-ui-1}" >/dev/null 2>&1 || true
        psql_ -d postgres -qAt -c "select pg_terminate_backend(pid) from pg_stat_activity where datname='$DB' and pid<>pg_backend_pid()" >/dev/null
        psql_ -d postgres -q -c "drop database if exists \"$DB\"" -c "create database \"$DB\" owner iris"
        psql_ -d "$DB" -q -c "create extension if not exists timescaledb" -c "select timescaledb_pre_restore()" >/dev/null
        # errors about extensions that already exist are expected; anything else is shown
        docker exec -i "$PG" pg_restore -U iris -d "$DB" --no-owner --role=iris < "$in" 2> /tmp/scale_db_restore.err || true
        grep -v "already exists" /tmp/scale_db_restore.err | grep -E "error|ERROR" | head -20 || true
        psql_ -d "$DB" -q -c "select timescaledb_post_restore()" -c "analyze" >/dev/null
        echo "restored $in -> $DB" ;;
    counts)
        psql_ -d "$DB" -qAt -F' ' <<'SQL'
select 'trucks', count(*) from trucks
union all select 'journeys', count(*) from journeys
union all select 'visits', count(*) from visits
union all select 'sightings', count(*) from sightings
union all select 'alerts', count(*) from alerts
union all select 'feed_events_raw', count(*) from feed_events_raw
union all select 'stops', count(*) from stops
union all select 'cameras', count(*) from cameras
union all select 'users', count(*) from users
union all select 'site_maps', count(*) from site_maps
order by 1;
SQL
        ;;
    *) echo "usage: $0 dump FILE | restore FILE | counts"; exit 2 ;;
esac
