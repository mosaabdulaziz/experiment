#!/usr/bin/env bash
# Named versions of the platform's day data on the laptop: save what is in the database now, list them, bring one back.
#
#   scale_day_version.sh save NAME      ~/URCA/IRIS/versions/NAME/: platform.dump, removed_trucks.json, selection files,
#                                       counts.json (day trucks), manifest.txt (gids, commits), the atlas page if built
#   scale_day_version.sh list           the saved versions with their counts
#   scale_day_version.sh restore NAME   replace the platform database with that version (the UI container is restarted),
#                                       then open https://localhost:35444/dashboard?day=2026-10-06
#
# A version holds the platform database only (trucks, journeys, visits, sightings, ...). The camera system's files and
# the original export archive (~/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL, read-only) are not copied.
# "Day trucks" are the ones created by the import (before DAY_CUTOFF); trucks a later live run added are in the dump
# too, but not in counts.json / the gid list. Restore never touches an existing version.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${VERSIONS:-$HOME/URCA/IRIS/versions}"
REVIEW="$HOME/URCA/IRIS/review"
PG="${PG_CONTAINER:-scale-ui-postgres-1}"
DAY_CUTOFF="${DAY_CUTOFF:-2026-10-07 13:00}"
q() { docker exec -i "$PG" psql -U iris -d iris -qAt -F' ' "$@"; }

counts() {
    q <<SQL
select 'trucks', count(*) from trucks where created_at < '$DAY_CUTOFF'
union all select 'journeys', count(*) from journeys where truck_id in (select id from trucks where created_at < '$DAY_CUTOFF')
union all select 'visits', count(*) from visits where truck_id in (select id from trucks where created_at < '$DAY_CUTOFF')
union all select 'sightings', count(*) from sightings where truck_id in (select id from trucks where created_at < '$DAY_CUTOFF')
union all select 'trucks_all', count(*) from trucks;
SQL
}

case "${1:-}" in
    save)
        name="${2:?usage: scale_day_version.sh save NAME}"
        d="$ROOT/$name"
        [ ! -e "$d" ] || { echo "version $name already exists ($d); pick another name" >&2; exit 1; }
        mkdir -p "$d"
        "$REPO/scripts/scale_db.sh" dump "$d/platform.dump"
        [ -f "$REVIEW/removed_trucks.json" ] && cp "$REVIEW/removed_trucks.json" "$d/"
        for f in "$REPO"/selection*.json; do [ -f "$f" ] && cp "$f" "$d/"; done
        [ -d "$REVIEW/atlas-day-v1" ] && cp -r "$REVIEW/atlas-day-v1" "$d/atlas"
        [ -d "$REVIEW/offline-direction" ] && cp -r "$REVIEW/offline-direction" "$d/offline-direction"
        counts | tee "$d/counts.txt"
        python3 - "$d" <<'PY'
import json, sys
d = sys.argv[1]
c = {k: int(v) for k, v in (l.split() for l in open(d + "/counts.txt"))}
json.dump(c, open(d + "/counts.json", "w"), indent=1)
PY
        rm "$d/counts.txt"
        {
            echo "version $name saved $(date '+%F %T')"
            echo "camera repo $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" branch --show-current))"
            echo "ui repo $(git -C "$REPO/../truck-tracking" rev-parse --short HEAD 2>/dev/null || echo unknown)"
            echo "day trucks (created before $DAY_CUTOFF): gids"
            q -c "select regexp_replace(external_vehicle_id, '^.*-', '')::int from trucks where created_at < '$DAY_CUTOFF' order by 1" | tr '\n' ' '
            echo
        } > "$d/manifest.txt"
        echo "saved version $name -> $d ($(du -sh "$d" | cut -f1))" ;;
    list)
        [ -d "$ROOT" ] || { echo "no versions yet ($ROOT)"; exit 0; }
        for d in "$ROOT"/*/; do
            [ -f "$d/counts.json" ] || continue
            echo "$(basename "$d"): $(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(', '.join(f'{k} {v}' for k, v in c.items()))" "$d/counts.json")  [$(sed -n 1p "$d/manifest.txt")]"
        done ;;
    restore)
        name="${2:?usage: scale_day_version.sh restore NAME}"
        d="$ROOT/$name"
        [ -s "$d/platform.dump" ] || { echo "no such version: $name" >&2; exit 1; }
        "$REPO/scripts/scale_db.sh" restore "$d/platform.dump"
        q -c "refresh materialized view visits_hourly" -c "refresh materialized view journeys_hourly" >/dev/null
        docker start "${UI_CONTAINER:-scale-ui-1}" >/dev/null
        counts
        echo "restored version $name: https://localhost:35444/dashboard?day=2026-10-06" ;;
    *) echo "usage: $0 save NAME | list | restore NAME"; exit 2 ;;
esac
