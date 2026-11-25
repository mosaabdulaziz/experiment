#!/usr/bin/env python3
"""Remove false-positive trucks from the platform's data, so they vanish from the dashboard and every count.

  scale_remove_trucks.py 13 42 351                 what removing these trucks (by gid) would delete: DRY RUN, nothing changes
  scale_remove_trucks.py selection.json            the file the review page downloads ({"gids": [13, 42]})
  scale_remove_trucks.py selection.json --apply    back the database up, delete, refresh the hourly views

A truck's gid N is the platform truck `ااا-NNNN` (external_vehicle_id). Deleting the truck row removes its journeys, visits,
sightings, paths, loops, transitions, aliases and schedules (foreign keys cascade); alerts whose context names the truck
are deleted too. The dry run does the real delete inside a transaction and rolls it back, so its counts are exact.
Every removal is added to ~/URCA/IRIS/review/removed_trucks.json; `just demo-v1` (a clean re-import of the archived day)
applies that list again, so the removals survive a re-import. The archived original export is never touched.
Not touched: the camera system's own files (crop images, stays, plate readings, VLM log) and Grafana history.
"""
import argparse, datetime as dt, json, os, subprocess, sys

PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
RECORD = os.environ.get("REMOVED_TRUCKS", os.path.expanduser("~/URCA/IRIS/review/removed_trucks.json"))
BACKUPS = os.path.expanduser("~/URCA/IRIS/backups")
PREFIX = "ااا-"
HERE = os.path.dirname(os.path.abspath(__file__))


def psql(sql):
    r = subprocess.run(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", "iris", "-v", "ON_ERROR_STOP=1", "-At", "-F", "|"],
                       input=sql, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"database error: {r.stderr[-600:]}")
    return r.stdout


def parse(args):
    gids = set()
    for a in args:
        if os.path.exists(a):
            d = json.load(open(a))
            gids |= {int(g) for g in (d["gids"] if isinstance(d, dict) else d)}
        else:
            gids.add(int(a.replace(PREFIX, "")))
    return sorted(gids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="*", help="gids, or a selection.json")
    ap.add_argument("--apply", action="store_true", help="really delete (default: dry run)")
    ap.add_argument("--from-record", action="store_true", help="apply the list in removed_trucks.json (after a re-import)")
    a = ap.parse_args()
    gids = parse(a.items)
    if a.from_record:
        gids = sorted(set(gids) | set(json.load(open(RECORD))["gids"])) if os.path.exists(RECORD) else gids
    if not gids:
        sys.exit("nothing to remove (give gids or a selection file)")
    ext = ", ".join(f"'{PREFIX}{g:04d}'" for g in gids)
    found = psql(f"select external_vehicle_id from trucks where external_vehicle_id in ({ext})").split()
    if not found:
        print("none of these trucks is in the database (already removed?)"); return
    missing = [f"{PREFIX}{g:04d}" for g in gids if f"{PREFIX}{g:04d}" not in found]
    ids = "select id from trucks where external_vehicle_id in (" + ext + ")"
    tables = psql("select c.table_name from information_schema.columns c join information_schema.tables t using (table_schema, table_name) "
                  "where c.table_schema='public' and c.column_name='truck_id' and t.table_type='BASE TABLE' order by 1").split()
    counts = " union all ".join(f"select '{t}', count(*) from {t} where truck_id in ({ids})" for t in tables)
    before = {k: int(v) for k, v in (l.split("|") for l in psql(counts + ";").splitlines())}
    alerts_where = f"exists (select 1 from trucks t where t.external_vehicle_id in ({ext}) and alerts.context::text like '%' || t.id::text || '%')"
    n_alerts = int(psql(f"select count(*) from alerts where {alerts_where}").strip())
    print(f"{len(found)} trucks to remove" + (f" ({len(missing)} not in the database: {' '.join(missing)})" if missing else ""))
    print("rows that go with them:", {k: v for k, v in before.items() if v}, f"alerts {n_alerts}")

    # exact check: the real delete inside a transaction, rolled back
    after = psql(f"begin; delete from alerts where {alerts_where}; delete from trucks where external_vehicle_id in ({ext}); "
                 f"select 'trucks', count(*) from trucks; select 'journeys', count(*) from journeys; select 'visits', count(*) from visits; "
                 f"select 'sightings', count(*) from sightings; rollback;")
    print("after the delete (checked, rolled back):", " ".join(l.replace("|", " ") for l in after.splitlines() if "|" in l))
    if not a.apply:
        print("dry run: nothing changed. Add --apply to remove them."); return

    os.makedirs(BACKUPS, exist_ok=True)
    dump = os.path.join(BACKUPS, f"before-remove-trucks-{dt.datetime.now():%Y%m%d-%H%M%S}.dump")
    subprocess.run([os.path.join(HERE, "scale_db.sh"), "dump", dump], check=True)
    psql(f"begin; delete from alerts where {alerts_where}; delete from trucks where external_vehicle_id in ({ext}); commit;")
    psql("refresh materialized view visits_hourly; refresh materialized view journeys_hourly;")
    old = json.load(open(RECORD)) if os.path.exists(RECORD) else {"gids": [], "log": []}
    old["gids"] = sorted(set(old["gids"]) | {int(f.replace(PREFIX, "")) for f in found})
    old["log"].append({"at": dt.datetime.now().isoformat(timespec="seconds"), "removed": found, "backup": dump})
    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    json.dump(old, open(RECORD, "w"), indent=1, ensure_ascii=False)
    print(f"removed {len(found)} trucks. Backup: {dump}. List kept in {RECORD}. Reload the dashboard.")


if __name__ == "__main__":
    main()
