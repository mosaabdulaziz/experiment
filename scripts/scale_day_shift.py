#!/usr/bin/env python3
"""Move an imported day run in time, so it looks as if it had run on the day you pick.

The camera system and the platform stamp everything with the real clock, so a run that began at 09:46 on the 6th
shows up as 09:46 on the 6th. After the run has been imported (scripts/scale_day_import.sh, browse mode), this moves
every run timestamp by a fixed offset, e.g. to make footage 00:00 start at 00:00 of the 6th:

  scripts/scale_day_shift.py --from 2026-10-06T09:46:00 --to 2026-10-06T00:00:00        (what it would change: --dry-run)

--from is the real wall time the footage began (the export's manifest.json says it: footage_began_wall), --to the
time it should appear to have begun. Local times, same time zone for both. What is moved:
  * the platform database: every timestamp column of every table that is at or after --from minus 30 minutes
    (so the setup rows created before the run are left alone), and the hourly views are refreshed;
  * the camera system's Redis crop index (updated_at), the plate readings (data/plates/*.json), the vehicle stays
    (data/tracking_stays.json) and the VLM arbitration records (data/vlm_arbitration/arbitration_log.jsonl).
Not moved (they keep the real times): the Grafana/Prometheus/Loki history, container logs, the events log. Run it
once (the camera-system app is stopped for a moment while its stays file changes): it leaves data/.time_shift and refuses to run again unless --force.
"""
import argparse, datetime as dt, json, os, subprocess, sys

DATA = os.environ.get("SCALE_DATA", os.path.expanduser("~/URCA/IRIS/scale-local/data"))
PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
IRIS = os.environ.get("IRIS_CONTAINER", "scale-iris-1")
TIME_KEYS = {"at", "saved_at", "wall_first", "wall_last", "submitted_at", "resolved_at", "updated_at", "episodes"}


def sh(cmd, inp=None):
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)[:120]}\n{r.stderr[-600:]}")
    return r.stdout


def psql(sql):
    return sh(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", "iris", "-v", "ON_ERROR_STOP=1", "-At", "-F", "|"], sql)


def shift_value(v, d, thr, key=None):
    """Shift epoch-second floats found under run-time keys (recursively); everything else is left as it is."""
    if isinstance(v, dict):
        return {k: shift_value(x, d, thr, k if k in TIME_KEYS else None) for k, x in v.items()}
    if isinstance(v, list):
        return [shift_value(x, d, thr, key) for x in v]
    if key and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= thr:
        return v + d
    return v


def shift_json_file(path, d, thr, dry):
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return False
    new = shift_value(data, d, thr)
    if new == data:
        return False
    if not dry:
        json.dump(new, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, dst = dt.datetime.fromisoformat(a.src), dt.datetime.fromisoformat(a.dst)
    d = (dst - src).total_seconds()
    thr = src.timestamp() - 1800
    marker = os.path.join(DATA, ".time_shift")
    if os.path.exists(marker) and not a.force and not a.dry_run:
        sys.exit(f"already shifted ({open(marker).read().strip()}); --force to shift again")
    print(f"shifting by {d:+.0f} s ({d / 3600:+.2f} h): everything at or after {dt.datetime.fromtimestamp(thr):%F %T}"
          + ("   [dry run]" if a.dry_run else ""))

    # --- platform database ---
    cols = psql("select table_name, column_name from information_schema.columns c where table_schema='public' "
                "and data_type='timestamp with time zone' and table_name in "
                "(select tablename from pg_tables where schemaname='public') and table_name <> 'schema_migrations' "
                "order by 1, ordinal_position")
    by_table = {}
    for line in cols.splitlines():
        t, c = line.split("|")
        by_table.setdefault(t, []).append(c)
    interval = f"interval '{d:.3f} seconds'"
    cutoff = f"to_timestamp({thr})"
    # Timescale cannot move a row to another chunk with UPDATE (its chunk's time constraint rejects it), so the
    # hypertables are copied out, emptied, shifted in the copy and put back.
    hyper = set(psql("select hypertable_name from timescaledb_information.hypertables").split())
    sql = ["begin;", "set session_replication_role = replica;"]     # no triggers: updated_at must keep the shifted value
    for t, cs in by_table.items():
        tmp = f"_h_{t}"
        if t in hyper and not a.dry_run:
            sql += [f'create temp table "{tmp}" as select * from "{t}";', f'delete from "{t}";']
        target = tmp if t in hyper and not a.dry_run else t
        for c in cs:
            sql.append(f"{'select' if a.dry_run else 'update'} " + (
                f"'{t}.{c}', count(*) from \"{t}\" where \"{c}\" >= {cutoff};" if a.dry_run else
                f"\"{target}\" set \"{c}\" = \"{c}\" + {interval} where \"{c}\" >= {cutoff};"))
        if t in hyper and not a.dry_run:
            sql.append(f'insert into "{t}" select * from "{tmp}";')
    sql += ["rollback;" if a.dry_run else "commit;"]
    out = psql("\n".join(sql))
    moved = {}
    for line in out.splitlines():
        if a.dry_run and "|" in line:
            k, n = line.rsplit("|", 1)
            moved[k] = int(n)
        elif line.startswith("UPDATE "):
            moved[len(moved)] = int(line.split()[1])
    if a.dry_run:
        print("database rows that would move:", {k: n for k, n in moved.items() if n})
    else:
        print(f"database: {sum(moved.values())} values moved in {len(by_table)} tables")
        psql("refresh materialized view visits_hourly; refresh materialized view journeys_hourly;")

    # --- camera system: Redis crop index ---
    code = ("import redis, json, sys\nr = redis.Redis(host='redis', db=0)\nd, thr, dry = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == '1'\n"
            "n = 0\nfor k, v in r.hgetall('iris:vehicle:crops').items():\n    e = json.loads(v)\n"
            "    if e.get('updated_at', 0) >= thr:\n        n += 1\n        if not dry:\n            e['updated_at'] += d\n            r.hset('iris:vehicle:crops', k, json.dumps(e))\nprint(n)\n")
    n = sh(["docker", "exec", "-i", IRIS, "python3", "-", str(d), str(thr), "1" if a.dry_run else "0"], code).strip()
    print(f"crop index: {n} entries {'would move' if a.dry_run else 'moved'}")
    if not a.dry_run:
        # keep the shifted index across a restart (Redis loads data/redis/dump.rdb when it starts)
        sh(["docker", "exec", "scale-redis-1", "redis-cli", "SAVE"])

    # --- camera system: files ---
    # The app keeps the stays in memory and writes them back to their file (every minute, and on every stop or restart):
    # edited while it runs, the shifted file is overwritten with the old times. So it is stopped while the files change and
    # started again with the new times loaded.
    if not a.dry_run:
        subprocess.run(["docker", "stop", IRIS], capture_output=True)
    files = [os.path.join(DATA, "tracking_stays.json")]
    for sub in ("plates",):
        base = os.path.join(DATA, sub)
        files += [os.path.join(base, f) for f in sorted(os.listdir(base)) if f.endswith(".json")] if os.path.isdir(base) else []
    changed = sum(shift_json_file(f, d, thr, a.dry_run) for f in files)
    log = os.path.join(DATA, "vlm_arbitration", "arbitration_log.jsonl")
    if os.path.exists(log):
        rows, c2 = [], 0
        for line in open(log):
            try:
                row = json.loads(line)
            except ValueError:
                rows.append(line.rstrip("\n")); continue
            new = shift_value(row, d, thr)
            c2 += new != row
            rows.append(json.dumps(new))
        if not a.dry_run:
            open(log + ".tmp", "w").write("\n".join(rows) + "\n")
            os.replace(log + ".tmp", log)
        print(f"VLM records: {c2} rows {'would move' if a.dry_run else 'moved'}")
    print(f"files: {changed} json files {'would change' if a.dry_run else 'changed'} (stays, plate readings)")
    if not a.dry_run:
        open(marker, "w").write(f"{a.src} -> {a.dst} ({d:+.0f} s) at {dt.datetime.now():%F %T}\n")
        subprocess.run(["docker", "start", IRIS], capture_output=True)
        print("done (the camera-system app was stopped while its stays were shifted, and is started again)")


if __name__ == "__main__":
    main()
