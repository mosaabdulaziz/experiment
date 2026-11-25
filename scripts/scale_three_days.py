#!/usr/bin/env python3
"""Three versions of the recorded day side by side in the platform, each on its own date:

    2026-10-05   the original export (all 365 tracked trucks)         trucks ااب-NNNN   (gid + 10000)
    2026-10-06   day-v1: after your manual false-positive removals    trucks ااا-NNNN   (the real gids, as imported)
    2026-10-07   day-v2-newrules: what the new start rules would keep trucks ااج-NNNN   (gid + 20000)

  scale_three_days.py            dry run: builds the copies in a transaction, prints counts, rolls everything back
  scale_three_days.py --apply    does it: restores day-v1 (the base), removes the live-run trucks, adds the other two days,
                                 adds their crop-index and stay entries to the camera system (its app is restarted)

  --from-archive --keep-base --base-day YYYY-MM-DD   (what `just demo-v2` uses) build everything from the archived original:
                                 the platform already holds the base day (the archive imported on BASE, minus the manual
                                 removals); this adds BASE-1 (all trucks) and BASE+1 (minus the manual and the new-rules removals,
                                 review/removed_trucks.json + review/whatif/removed_trucks_newrules.json), on any dates.

The platform has one row per truck (unique external id), so a copy needs new ids: every id of a copied row (trucks, journeys,
visits, sightings, paths, loops, transitions, ...) becomes a new uuid, every timestamp moves by whole days, and the truck's
external id gets the letters that encode gid + 10000 / + 20000 (the UI reads the camera system's id from them, so profiles keep
working). The camera system gets the same crop-index and stay entries under those ids (they point at the same crop files).
Reads only: the dumps (archive, versions day-v1 / day-v2-newrules). Scratch databases tmp_orig / tmp_v2 are created and dropped.
"""
import argparse, csv, datetime as dt, io, json, os, re, subprocess, sys, tempfile, uuid

HOME = os.path.expanduser("~")
PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
APP = os.environ.get("IRIS_CONTAINER", "scale-iris-1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DUMP = f"{HOME}/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL/db/platform.dump"
V2_DUMP = f"{HOME}/URCA/IRIS/versions/day-v2-newrules/platform.dump"
DATA = f"{HOME}/URCA/IRIS/scale-local/data"
CUTOFF = "2026-10-07 13:00"                                    # trucks created before this are the day's trucks
NS = uuid.UUID("5d0e6a1c-7f55-4b53-9c1e-2a7d6a5c0b11")
WALL_START_S = 9 * 3600 + 46 * 60                              # footage 00:00 played at wall 09:46 on the 6th
LETTERS = {1: "ااب", 2: "ااج"}                                 # the camera system's gid + 10000 / + 20000, as the UI decodes it
# tag, dump, scratch database, seconds every platform timestamp moves, step (gid offset = step * 10000),
# seconds the camera system's times move (its files are already on the 10-06 timeline)
REVIEW = f"{HOME}/URCA/IRIS/review"
MANIFEST = f"{HOME}/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL/manifest.json"


def wall_begin():
    """When footage 00:00 played (local wall time), from the export's manifest."""
    try:
        return json.load(open(MANIFEST))["footage_began_wall"]
    except (OSError, KeyError, ValueError):
        return "2026-10-06T09:46:00"


def delta_to(day, begin):
    """Seconds that move a timestamp of the original run so that footage 00:00 lands on `day` 00:00."""
    return (dt.datetime.fromisoformat(f"{day}T00:00:00") - dt.datetime.fromisoformat(begin)).total_seconds()


def removed_gids():
    out = set()
    for f in (f"{REVIEW}/removed_trucks.json", f"{REVIEW}/whatif/removed_trucks_newrules.json"):
        if os.path.exists(f):
            out |= set(json.load(open(f))["gids"])
    return out
EXT = re.compile("ااا-(\\d{4})")


def run(cmd, inp=None):
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)[:140]}\n{r.stderr[-900:]}")
    return r.stdout


def psql(db, sql):
    return run(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", db, "-v", "ON_ERROR_STOP=1", "-qAt", "-F", "\t"], sql)


def build_temp_db(db, dump):
    """Restore a dump into its own scratch database (the platform's database is not touched and the UI keeps running)."""
    psql("postgres", f'drop database if exists "{db}"; create database "{db}" owner iris;')
    psql(db, "create extension if not exists timescaledb; select timescaledb_pre_restore();")
    with open(dump, "rb") as f:
        subprocess.run(["docker", "exec", "-i", PG, "pg_restore", "-U", "iris", "-d", db, "--no-owner", "--role=iris"], stdin=f, capture_output=True)
    psql(db, "select timescaledb_post_restore();")
    print(f"  {db}: {psql(db, 'select count(*) from trucks').strip()} trucks from {os.path.relpath(dump, HOME)}")


def tables(db):
    """The tables of a truck's story: trucks and everything that carries a truck, journey or visit id (generated columns left out)."""
    t = {}
    q = ("select table_name, string_agg(column_name||':'||data_type||':'||is_generated, ',' order by ordinal_position) from information_schema.columns "
         "where table_schema='public' and table_name in (select tablename from pg_tables where schemaname='public') group by 1")
    for line in psql(db, q).splitlines():
        name, cols = line.split("\t")
        cs = [c.split(":") for c in cols.split(",")]
        if name == "trucks" or {c[0] for c in cs} & {"truck_id", "journey_id", "visit_id"}:
            t[name] = [(c[0], c[1]) for c in cs if c[2] != "ALWAYS"]
    return t


def where_for(tab, cols):
    names = {c for c, _ in cols}
    day = f"(select id from trucks where created_at < '{CUTOFF}')"
    if tab == "trucks":
        return f"created_at < '{CUTOFF}'"
    if "truck_id" in names:
        return f"truck_id in {day}"
    if "journey_id" in names:
        return f"journey_id in (select id from journeys where truck_id in {day})"
    return f"visit_id in (select id from visits where truck_id in {day})"


def copy_out(db, tab, cols, where):
    names = ", ".join(f'"{c}"' for c, _ in cols)
    out = psql(db, f"copy (select {names} from \"{tab}\" where {where}) to stdout with (format csv, null '\\N')")
    return list(csv.reader(io.StringIO(out)))


def move(v, secs):
    return (dt.datetime.fromisoformat(v) + dt.timedelta(seconds=secs)).isoformat(sep=" ")


def etl(tag, db, secs, step, drop=()):
    """-> (SQL that inserts the copy, row counts, the source gids of its trucks)"""
    if drop:                                                      # trucks that day does not have (cascades to their rows)
        ids = ", ".join(f"'ااا-{g:04d}'" for g in sorted(drop))
        psql(db, f"delete from trucks where external_vehicle_id in ({ids})")
    src = tables(db)
    new_ext = lambda m: f"{LETTERS[step]}-{m.group(1)}"
    rows, idmap = {}, {}
    for tab, cols in src.items():
        rows[tab] = copy_out(db, tab, cols, where_for(tab, cols))
        names = [c for c, _ in cols]
        if "id" in names:
            i = names.index("id")
            for r in rows[tab]:
                idmap[r[i]] = str(uuid.uuid5(NS, f"{tag}:{r[i]}"))
    parts, counts = [], {}
    for tab, cols in src.items():
        names = [c for c, _ in cols]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for r in rows[tab]:
            out = []
            for (c, typ), v in zip(cols, r):
                if v == "\\N":
                    out.append(v)
                elif typ == "uuid":
                    out.append(idmap.get(v, v))                 # copied rows get new ids; stops, cameras, users, ... stay
                elif typ.startswith("timestamp"):
                    out.append(move(v, secs))
                elif typ == "text":
                    out.append(EXT.sub(new_ext, v))
                else:
                    out.append(v)
            if tab == "trucks" and re.fullmatch("ا+", out[names.index("plate_letters")]):
                out[names.index("plate_letters")] = LETTERS[step]      # an unread plate shows the truck's own id letters
            w.writerow(out)
        counts[tab] = len(rows[tab])
        if rows[tab]:
            cl = ", ".join(f'"{n}"' for n in names)
            parts.append(f"copy \"{tab}\" ({cl}) from stdin with (format csv, null '\\N');\n{buf.getvalue().rstrip(chr(10))}\n\\.")
    ei = [c for c, _ in src["trucks"]].index("external_vehicle_id")
    gids = {int(EXT.search(r[ei]).group(1)) for r in rows["trucks"] if EXT.search(r[ei])}
    return "\n".join(parts), counts, gids


def camera_side(copies):
    """Crop-index and stay entries of the copies, under gid + 10000 / + 20000 (same crop files, times moved by whole days)."""
    sys.path.insert(0, f"{REPO}/scripts")
    import scale_day_shift as S
    spec = [{"step": step, "rel": rel, "gids": sorted(gids)} for step, rel, gids in copies]
    code = ("import json, sys, redis\nr = redis.Redis(host='redis', db=0)\nspec = json.loads(sys.argv[1])\nn = 0\n"
            "for k, v in r.hgetall('iris:vehicle:crops').items():\n    k = k.decode(); parts = k.split(':')\n"
            "    if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) >= 10000: continue\n    e = json.loads(v)\n"
            "    for c in spec:\n        if int(parts[1]) in c['gids']:\n            g = int(parts[1]) + c['step'] * 10000\n"
            "            e2 = dict(e, global_id=g, updated_at=e.get('updated_at', 0) + c['rel'])\n"
            "            r.hset('iris:vehicle:crops', ':'.join([parts[0], str(g)] + parts[2:]), json.dumps(e2)); n += 1\nprint(n)\n")
    n = run(["docker", "exec", "-i", APP, "python3", "-", json.dumps(spec)], code).strip()
    run(["docker", "exec", "scale-redis-1", "redis-cli", "SAVE"])
    print(f"  crop index: {n} entries added")
    run(["docker", "stop", APP])                                  # the app rewrites its stays file: edit it while it is stopped
    tmp = os.path.join(tempfile.mkdtemp(), "stays.json")
    run(["docker", "cp", f"{APP}:/app/data/tracking_stays.json", tmp])
    d = json.load(open(tmp))
    added = 0
    for cam, v in d["cameras"].items():
        for g in list(v):
            if int(g) >= 10000:
                continue
            for step, rel, gids in copies:
                if int(g) in gids:
                    v[str(int(g) + step * 10000)] = S.shift_value({"x": v[g]}, rel, 0, None)["x"]
                    added += 1
    json.dump(d, open(tmp, "w"))
    run(["docker", "cp", tmp, f"{APP}:/app/data/tracking_stays.json"])
    run(["docker", "start", APP])
    print(f"  stays: {added} vehicle-camera entries added; app restarted")
    # the crop endpoint rebuilds the file name from the gid (<gid>[-stay].jpg), so every copy id needs a file of that name
    n = run(["docker", "exec", APP, "sh", "-c",
             "cd /app/data/vehicle_crops && n=0; for d in cam-*; do for f in $d/[0-9]*.jpg; do b=$(basename $f .jpg); g=${b%%-*}; s=''; "
             "case $b in *-*) s=\"-${b#*-}\";; esac; [ \"$g\" -lt 10000 ] 2>/dev/null || continue; for k in 1 2; do t=$d/$((g+k*10000))$s.jpg; "
             "[ -e \"$t\" ] || { ln -s \"$(basename $f)\" \"$t\" && n=$((n+1)); }; done; done; done; echo $n"]).strip()
    print(f"  crop files: {n} links to the real crops added")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--base-day", default="2026-10-06", help="the date of the middle day (day-v1); the others are the day before and after")
    ap.add_argument("--from-archive", action="store_true", help="build the third day from the archive and the removal lists, not from the day-v2 dump")
    ap.add_argument("--keep-base", action="store_true", help="the platform already holds the base day: do not restore day-v1 or delete live trucks")
    a = ap.parse_args()
    base = dt.date.fromisoformat(a.base_day)
    prev, nxt = (base - dt.timedelta(days=1)).isoformat(), (base + dt.timedelta(days=1)).isoformat()
    begin = wall_begin()
    sources = [("orig", ARCHIVE_DUMP, "tmp_orig", delta_to(prev, begin), 1, -86400, ())]
    if a.from_archive:
        sources.append(("newrules", ARCHIVE_DUMP, "tmp_v2", delta_to(nxt, begin), 2, 86400, removed_gids()))
    else:
        if a.base_day != "2026-10-06":
            sys.exit("the day-v2 dump is on 2026-10-06: other dates need --from-archive")
        sources.append(("newrules", V2_DUMP, "tmp_v2", 86400, 2, 86400, ()))
    for p in {s[1] for s in sources}:
        if not os.path.exists(p):
            sys.exit(f"missing dump: {p}")
    print(f"== days: {prev} original, {a.base_day} base, {nxt} new rules")
    print("== scratch databases from the dumps")
    for tag, dump, db, secs, step, rel, drop in sources:
        build_temp_db(db, dump)
    try:
        if a.apply and not a.keep_base:
            print("== base: day-v1 as 2026-10-06, without the live-run trucks")
            run([f"{REPO}/scripts/scale_day_version.sh", "restore", "day-v1"])
            psql("iris", f"delete from trucks where created_at >= '{CUTOFF}'")
        parts, copies = [], []
        for tag, dump, db, secs, step, rel, drop in sources:
            sql, counts, gids = etl(tag, db, secs, step, drop)
            parts.append(sql)
            copies.append((step, rel, gids))
            print(f"== {tag}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
        body = "begin;\nset session_replication_role = replica;\n" + "\n".join(parts) + "\n"
        out = psql("iris", body + ("commit;" if a.apply else "select 'trucks in the platform with the copies', count(*) from trucks;\nrollback;"))
        print("== inserted" if a.apply else "== dry run (rolled back):", out.strip().replace("\t", " "))
        if a.apply:
            psql("iris", "refresh materialized view visits_hourly; refresh materialized view journeys_hourly; analyze;")
            print("== camera system")
            camera_side(copies)
            print(f"done: dashboard days {prev} (original), {a.base_day} (day-v1), {nxt} (new rules)")
    finally:
        for tag, dump, db, secs, step, rel, drop in sources:
            psql("postgres", f'drop database if exists "{db}"')


if __name__ == "__main__":
    main()
