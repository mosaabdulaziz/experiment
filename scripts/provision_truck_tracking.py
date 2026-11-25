#!/usr/bin/env python3
"""Register our tracked cameras in a local truck-tracking database.

Data only - nothing in the truck-tracking repo is modified. Reads the
`truck_feed` + `mtmc` sections of config.yaml and creates, under the tenant ids
the bridge sends (app/truck_feed.py), one company, one branch, a published site
map, one stop per camera (in traversal order) and one camera per stop at the
configured IP. Re-running is safe (upserts). The seeded admin user is moved
under the company so `admin@example.com` can see it.

    python3 scripts/provision_truck_tracking.py [--container iris-postgres]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.truck_feed import tenant_uuid  # noqa: E402


Y = 0.00025  # latitude every stop sits on


def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="iris-postgres")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tf, mt = cfg["truck_feed"], cfg["mtmc"]
    co, br = tenant_uuid(tf["company_id"]), tenant_uuid(tf["branch_id"])
    names = mt.get("camera_names", {})
    site_map = "00000000-0000-5000-8000-%012d" % 1
    n = len(tf["camera_ips"])
    stops, cams = [], []
    for i, (cam, ip) in enumerate(tf["camera_ips"].items()):
        code = names.get(cam, cam).upper().replace(" ", "_")
        # truck-tracking stores plan positions as lng/lat degrees inside its plan
        # frame (centre 0.0004,0.00025; half 0.0018 x 0.0012). Spread the stops
        # left to right across the middle of it.
        x = round(-0.0012 + i * (0.0032 / max(n - 1, 1)), 6)
        sid = "00000000-0000-5000-8000-%012d" % (100 + i)
        cid = "00000000-0000-5000-8000-%012d" % (200 + i)
        # The factory is entered and left through the first camera's gate, so reaching the
        # last camera doesn't end a trip: only an exit event does.
        kind = "entry" if i == 0 else "checkpoint"
        stops.append((sid, code, kind, x, i, names.get(cam, cam)))
        cams.append((cid, code, ip, sid, x, cam))
    geo = json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"kind": "road", "roadClass": "primary"},
         "geometry": {"type": "LineString", "coordinates": [[-0.0014, Y], [0.0022, Y]]}}]
        + [{"type": "Feature", "properties": {"kind": "stop", "code": c, "stopKind": k},
            "geometry": {"type": "Point", "coordinates": [x, Y]}} for _, c, k, x, _, _ in stops]})
    graph = json.dumps({"nodes": [{"id": c, "x": x, "y": Y} for _, c, _, x, _, _ in stops],
                        "edges": [{"from": stops[i][1], "to": stops[i + 1][1], "length": 20}
                                  for i in range(n - 1)]})
    # 0 = no timeout: a vehicle stays on site until it is seen leaving. The platform default is 240 min.
    idle = int(tf.get("journey_idle_timeout_minutes", 0)) or 525600
    sql = [
        "begin;",
        f"insert into companies (id,name,slug,plan,settings) values ({q(co)},"
        f"'{{\"ar\":\"المهندس\",\"en\":{json.dumps(tf['company_id'])}}}'::jsonb,{q(tf['company_id'].lower())},'trial',"
        "'{\"defaultLocale\":\"ar\",\"defaultDigitStyle\":\"arabic-indic\",\"defaultTimezone\":\"Africa/Cairo\"}'::jsonb)"
        " on conflict (id) do nothing;",
        f"insert into branches (id,company_id,name,code,timezone) values ({q(br)},{q(co)},"
        f"'{{\"ar\":\"الفرع\",\"en\":{json.dumps(tf['branch_id'])}}}'::jsonb,{q(tf['branch_id'])},'Africa/Cairo')"
        " on conflict (id) do nothing;",
        f"insert into site_maps (id,company_id,branch_id,source,crs,geojson,raw_input,graph,status,published_at,"
        f"bounds,scale_meters_per_unit) values ({q(site_map)},{q(co)},{q(br)},'drawn','local',{q(geo)}::jsonb,"
        f"{q(geo)}::jsonb,{q(graph)}::jsonb,'published',now(),'{{\"minX\":-0.0014,\"minY\":-0.00095,\"maxX\":0.0022,\"maxY\":0.00145}}'::jsonb,2)"
        # An existing map is the user's (drawn in the UI): never overwrite it, only create it when missing.
        " on conflict (id) do nothing;",
    ]
    for sid, code, kind, x, i, nm in stops:
        sql.append(
            f"insert into stops (id,company_id,branch_id,site_map_id,name,code,kind,position_x,position_y,sort_order,"
            f"expected_dwell_seconds,max_dwell_seconds) values ({q(sid)},{q(co)},{q(br)},{q(site_map)},"
            f"{q(json.dumps({'ar': nm, 'en': nm}))}::jsonb,{q(code)},'{kind}',{x},{Y},{i},120,600) on conflict (id) do nothing;")
    for cid, code, ip, sid, x, cam in cams:
        sql.append(
            f"insert into cameras (id,company_id,branch_id,name,ip_address,external_camera_id,stop_id,position_x,"
            f"position_y,status) values ({q(cid)},{q(co)},{q(br)},{q('CAM-' + code)},{q(ip)},{q(cam)},{q(sid)},{x},{Y + 0.00004},'online')"
            " on conflict (id) do nothing;")
    sql += [
        f"update branches set settings = settings || jsonb_build_object('journey_idle_timeout_minutes', {idle}) "
        f"where id={q(br)};",
        f"update users set company_id={q(co)} where email='admin@example.com';",
        f"insert into branch_memberships (user_id,branch_id,company_id) select id,{q(br)},{q(co)} from users "
        "where email='admin@example.com' on conflict do nothing;",
        "commit;",
    ]
    text = "\n".join(sql)
    if a.print_only:
        print(text)
        return
    r = subprocess.run(["docker", "exec", "-i", a.container, "psql", "-U", "iris", "-d", "iris",
                        "-v", "ON_ERROR_STOP=1"], input=text, text=True, capture_output=True)
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
