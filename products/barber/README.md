# products/barber

Barber shop monitoring for IRIS (use case: `Barber_Shop_Monitoring_Use_Case.docx`).
It follows each customer across **three overlapping cameras**, anonymously,
from entry to exit through the waiting area and a barber chair. It records
waiting time and haircut duration, credits each haircut to the barber working
that chair (via the roster), and raises alerts when a wait or a haircut runs
past its target.

It is a **separate deployable**: a FastAPI host process on port **8003** with
its own SQLite database. Like products/attendance, it never imports IRIS code;
Redis is the only connection. Staff are recognized by uniform, not by face:
the sidecar scores each sampled crop's uniform probability, and a track
becomes staff once enough of those samples agree.

```
barber-c ─┐
barber-l ─┼─▶ go2rtc ─▶ deepstream-sidecar-barber ──Redis──▶ barber-service (:8003) ──▶ REST + /ws
barber-r ─┘   (raw-barber-<cam>) (sidecar.barber.yaml)  iris:meta:<cam>       acl ▶ ShopEngine ▶ SQLite
                                  person pgie + tracker  (person boxes)
                                  + appearance sampling  iris:appearance:<cam>
                                                          (uniform score + re-ID crops)
```

Each camera's frames and sampled appearances feed one `ShopEngine` (not one
engine per camera): zones drawn in two cameras can be paired to the same
shop **area** (a chair seen from the centre and from the side), so a customer
handed off between cameras stays one visit.

## What was reused

The modules listed as forked below are **copies** of products/attendance code
at `1f5b27b`, not imports. The barber service ships separately and is expected
to change these modules. Each forked file's header names its source and what
changed. The rule for sharing: extract a common package once a third use case
needs the same pieces.

| Module | Origin |
|---|---|
| `acl.py` | forked: Redis stream readers, camera clock, `LiveCameraWatch` |
| `zones.py`, `jsonstore.py` | forked: polygon geometry, hot-reloading JSON stores; the entrance line (`DoorLine`) and cross-camera zone pairing are new |
| `storage.py` | forked: `connect` (WAL, busy timeout, schema + migrations on every connect); everything else is the barber tables |
| `server.py` | forked: the websocket and the app factory (`create_app`); the rest is new |
| `engine.py`, `customers.py`, `chairs.py`, `identity.py`, `tracklets.py`, `domain.py`, `alerts.py`, `stats.py` | new |

On the IRIS side there is only a small amount of wiring:
- a `barber` sidecar role (`perception/engine/sources.py`);
- an `extends:` overlay loader, so `products/barber/sidecar.yaml` holds only the
  barber differences;
- a tracker config tuned for accuracy (`products/barber/tracker.yml`);
- the `deepstream-sidecar-barber` compose service (profile `barber`);
- go2rtc entries;
- `just barber-*` recipes.

## Zones

Zones are normalized to 0..1 and stored per camera (`zones.json`:
`{cam_id: [zone, ...]}`). They are set with `PUT /api/monitoring/zones/{cam}`,
the same contract as the attendance zone editor; a bad zone rejects the
whole write (a silently vanished chair would silently stop counting
haircuts).

| Type | Shape | Meaning |
|---|---|---|
| `entrance_line` | `points: [[x1,y1],[x2,y2]]`, `inside: [x,y]` | the doorway: a line drawn **on the floor across the threshold**; `inside` is any point on the shop side |
| `waiting` | polygon, 3+ points | a waiting area (benches, sofa) |
| `chair` | polygon, 3+ points | one barber chair's presence area |
| `seat` | polygon, 3+ points, `seat_of: <chair zone id, same camera>` | the seat itself; a still, non-staff body inside it starts or resumes a haircut |
| `washing` | polygon, 3+ points | the wash station; a haircut that dips through here and back stays one haircut |

`waiting`, `chair` and `washing` zones can carry `pair` and `owner`: a shop
**area** is what a zone stands for -- its own id, or its `pair` id when the
same physical place is drawn in two cameras (a chair in the centre view and
again in the side camera facing it). The pair's `owner: true` zone's camera
decides what happens there (seating, haircut start); an unpaired zone owns
itself. `GET /api/monitoring/zones/problems` reports inconsistencies across
cameras: a pair drawn in only one camera or drawn twice in one, a pair
without exactly one owner, or an owner chair with no `seat`.

The entrance line is tested against each person's **feet** (the bottom of
their box). Every area zone is tested against the **lower torso** (80% down
the box), because a chair or bench hides a seated person's feet.

## How a visit is measured

Every camera's frames and sampled appearances feed one `ShopEngine`
(`engine.py`); every threshold named below is a knob in `policy.py`. Per
frame, in order:

1. **Staff never carry a customer.** A bound track the staff vote calls staff
   is released; a person left with no track has its visit voided if it can be
   (no haircut yet).
2. **The door.** An unbound, not-staff track crossing inward creates a person
   and an unconfirmed visit; a bound one crossing out marks a pending exit
   (crossing back in before it's confirmed cancels it). In the centre view
   the door line runs next to the front chairs, so a pending exit is also
   cancelled when another of the customer's tracks sits in his own chair's
   seat for `seat_dwell_s` (out of it no longer than `seat_jitter_s`): lasting
   seat evidence means he never left. A moment of it doesn't -- that is a
   side view lagging a real exit.
3. **Cross-camera handoff.** An unbound, not-staff track tries a paired-zone
   handoff for each paired area it's in (ranked by how close in time the
   entries were, `sync_window_s`, and vetoed or broken by body re-ID
   similarity), then a re-ID re-attach to a customer recently lost from every
   view (`reattach_s`).
4. **Missed entries.** An unbound track the vote says is NOT staff, in a
   waiting area its camera owns, for `waiting_debounce_s`, creates a
   confirmed visit flagged `entry_observed=false`.
5. **Evidence and last-known places.** Every bound track notes where its
   person is, is presence evidence for their chair, and marks a washing visit
   while inside a washing area.
6. **Seats.** Only the camera that **owns** a chair (the zone marked
   `owner`, see "Zones") decides it; a seat another camera draws for the same
   chair starts and claims nothing. A body may sit when `seat_ok` says so:
   - a track latched as a customer (it sat `seat_dwell_s`) always may;
   - a track whose staff vote said staff while it stood **outside every
     seat** for `staff_outside_confirm_s` on end never may (a seat frame or a
     non-staff vote restarts that count);
   - otherwise only a staff-score mean at or above `seat_staff_threshold`
     rules it out -- caped, seated customers score close to the barbers.
   A still body that may sit, in the seat for `seat_dwell_s`, starts (or,
   back from washing, resumes) a haircut, backdated to when the stillness
   began -- once its vote says it is not staff. A haircut starts in a
   different chair from the customer's running one only if none of his
   other tracks is still in his own chair; otherwise the seated body is
   someone else and gets its own visit.
   **Seat claims:** the tracker often gives a seated customer a new track.
   An unbound body sitting still in the seat of a running haircut joins its
   customer only if `identity.may_claim_seat` allows it: he has no live
   track in that camera, every live track he has is inside that chair (none
   at all: lost), and body re-ID doesn't veto it (`reid_veto`). A declined
   claim is no evidence for the chair.

Then tracks unseen for `track_gone_s` (the camera's own clock) end, and the
time rules run:
- a running haircut survives its chair showing no presence evidence for up to
  `chair_gap_s`; a sit shorter than `min_service_s` in total is dropped, not a
  haircut;
- an unconfirmed visit is confirmed after `min_visit_s` inside, or finished at
  the door if it already crossed out;
- a crossed-out visit finishes once outside for `exit_confirm_s`, or as soon
  as the person leaves every camera's view;
- a visit backed by nobody in any view finishes after `lost_timeout_s`, dated
  to when it was lost;
- a haircut suspended for washing resumes if the customer is back in the same
  seat within `wash_return_s`, else it stays ended;
- an alert opens when a running wait crosses `wait_target_s` or a running
  haircut crosses `service_target_s`, and closes when that wait or haircut
  ends. Targets are stored on the visit / session, so changing the policy
  never rewrites history.

**Cameras that stall or die.** The engine clock is the newest frame of any
camera. A track older than `track_gone_s` on it -- its camera stalled -- is no
handoff partner, doesn't block a seat claim, and isn't listed among its
customer's `cameras` in the live view, and the live scene drops its boxes.
When a camera's frames come back after a gap longer than `track_gone_s`,
every track it had before is over: the tracker's ids restart with the
sidecar, so an id seen again is someone new. A camera whose newest frame
lags the engine clock by `camera_dead_s` is dead: its tracks end, so its
customers can go lost and finish after `lost_timeout_s`. The live payload's
`cameras` gives each camera's `last_frame_age_s` on the same clock (so a
stall of every camera at once reads 0 for the newest; the sidecar's fps
gauge shows that case). On the sidecar side, the stall watch restarts a
source whose frames stop for `source_stall_seconds`; one that stays dead
after `source_stall_max_failed_restarts` restarts ends the sidecar, and the
container restart brings every camera back (`sidecar.barber.yaml`).

**Staff** are recognized by uniform, not by face: the sidecar scores each
sampled appearance crop's uniform probability (`iris:appearance`'s
`staff_p`), and a track counts as staff once `staff_min_samples` crops
average at or above `staff_threshold`. **Haircuts are credited through the
roster**, not through per-track attribution: `GET/PUT /api/barber/roster`
says who works which chair right now, and a haircut's `barber_id` is whoever
`roster_at` says worked that chair when the haircut started.

## Run

```bash
just barber-up                                 # barber sidecar + barber-service (:8003)
just barber-up barber-c,barber-l,barber-r      # ...and stream the three synced test clips
just barber-down
just barber-test                               # this service's tests
```

**Test footage.** `barber-c` / `barber-l` / `barber-r` play the shop's own
synced three-camera recording (`bekky-10min`, minutes 20:00-30:00 of a 39-min
clip). It isn't in git, so link or copy the files into `data/uploads/barber/`
as `barber/bekky-10min/{center,left,right}.mp4` (see
`test_streamer/stream_recordings.py`'s `CAM_FILES`). Start all three together
-- the engine pairs zones across them, so a partial start leaves some areas
one-camera-only.

It's wired up in two places:
- `test_streamer/stream_recordings.py` `CAM_FILES` maps each camera id to its
  file, published as `raw-<cam>`.
- `barber.yaml` sets `clip_duration_s: 600` for each camera so a loop back to
  the start resets the engine, and `clear_on_loop: true` so each lap starts
  with clean stats. **For a real camera, set both back to `null` / `false`**:
  a live stream never loops, and a duration would make the service reset
  itself every 600 s.

To set up a shop:

1. **Draw the zones** (see [Zones](#zones)): the entrance line, waiting
   areas, each chair with its seat (paired across cameras where two views
   cover it), and the wash station.
2. **Enroll barbers.** `POST /api/staff {name}`, then put them on a chair:
   `PUT /api/barber/roster/{chair_id} {"staff_id": ...}`.
3. **Tune targets.** `PUT /api/barber/policy {"wait_target_s": 600, ...}`
   (a partial update).

**Config** is in `barber.yaml`: cameras, frame size and data dir. The
environment can override it with `REDIS_URL`, `BARBER_DATA_DIR` and
`BARBER_CONFIG`. The port is `BARBER_BACKEND_PORT` (default 8003). Camera ids
must match `products/barber/sidecar.yaml`.

### Second site (Palm Hills)

```bash
just barber_site=palm-hills barber-up     # its sidecar file, 7 cameras, its test clips (ph-1..ph-7)
just barber-restart                       # restarts the running site; another site is refused
just barber-down                          # then barber-up again to switch sites
```

- `site.sh` maps a site to its two files: `barber.palm-hills.yaml` (the 7
  named cameras, `wait_camera: ph-3`, a `policy:` block of site defaults) and
  `sidecar.palm-hills.yaml` (extends `sidecar.yaml`: 7 sources, its own
  crops dir, its own staff model). No site given = Bekky.
- **Per-site data.** `data_dir: data/palm-hills` (zones, policy, DB) and
  `crops_dir: ../../data/barber/crops-palm-hills`, matching the sidecar's
  `customer_crops_dir`. Never share or nest crops roots: each service's sweep
  deletes every crop its own DB doesn't reference.
- **Policy.** `policy.json` holds only the keys set in Setup; every other key
  follows the site's `policy:` block, then the defaults.
- **No door camera.** `wt_end_mode 1`: a wait ends at the waiting area's
  edge and a return (own track or re-ID) reopens it. `visits_source 1`:
  visits = haircuts + waits that led to none.
- **Seated faces.** `register_source 2` with a `face_area` zone per chair
  (any camera, tied to its chair): the sidecar runs face detection on people
  whose box overlaps a face area, and a face joins the chair's running
  haircut when its own box's centre is inside the area.
- **Staff.** Palm's own classifier, `staff_dinov2_s_palm.onnx` (round 4), with
  `staff_threshold 0.90`. `just barber-palm-staff r3` goes back to the round-3
  model (`staff_dinov2_s_palm_r3.onnx`, 0.79) and `r4` returns: it sets the
  model and its threshold together and restarts a running Palm site.

**The shop's live cameras** (`palm-hills-live`): `just barber_site=palm-hills-live barber-up` takes the 6 RTSP feeds
(no ph-4) from `.env`'s `PALM_HILLS_RTSP` (it holds the credentials: never print it); `barber.palm-hills-live.yaml`
is the Palm Hills site with live cameras (no clip lengths, nothing cleared), on the same data dir.

### Runs on recordings (`palm-hills-16oct`)

```bash
just barber_site=palm-hills-16oct barber-up    # the shop's 16 Oct, played by the test streamer as ph-1..ph-7 (no ph-4)
just barber-restart                            # soft: the service only, rows kept; the streamer and sidecar run on
```

- **What it plays.** The recorder's exports synced into one file per camera,
  `/data/IRIS/data/BekkyBarber/palm_hills2/synced/Cam*_2026-10-16_09-30-00_to_23-59-59_SYNCED.mp4`
  (`site.sh` `clip_files`), the whole day: `clip_duration_s: 52200` per camera in `barber.palm-hills-16oct.yaml`
  (it extends `barber.palm-hills.yaml`). Its own data dir `data/palm-hills-16oct` (DB, zones, policy) and crops
  `crops-palm-hills-16oct`, so the live site's rows stay as they are.
- **Its clock.** `footage_start: 2026-10-16T09:30:00`: the reports give the footage's date and times
  (`footage_clock.py`; the rows stay in device time). The dashboard's "Today" still counts the device's shop day.
- **One lap = one played day (14 h 30).** When the clips play out, the lap's reports are saved into their own folder,
  `data/palm-hills-16oct/reports/day/2026-10-16/lap-<lap end YYYY-MM-DD_HHMM>/` (analysis + haircut photos, pdf +
  html, en + ar: `saved_reports`; the HTML at once, the PDFs just after), then the rows are cleared and the next lap
  starts. Laps never block each other. The lap's start is in the DB's `service_state` `clip_run` ([start ts, length,
  lap]).
- **Never restart the streamer or the sidecar mid-lap** (a new run of the clips clears the rows); a soft
  `just barber-restart` keeps them (`hard` clears them).

## API

| Endpoint | What |
|---|---|
| `GET /api/barber/live` | one live snapshot: waiting customers, each chair's running haircut and barber, the scene (boxes), each camera's `last_frame_age_s` (`cameras`), zones and zone problems, the roster, `shop_day`, `day_start_hour`, `roster_confirmed`, open/recent alerts, today's figures |
| `GET /api/barber/stats?from&to&bucket=hour\|day\|week\|month` | the ten statistics for `[from, to)` (default: today), optionally as a series |
| `GET /api/barber/visits/{id}/events` | one visit's state-change audit trail |
| `GET /api/barber/handoffs?from&to` | confirmed cross-camera handoffs (for tuning `sync_window_s`) |
| `GET /api/barber/alerts?from&to&open` | excessive wait / haircut alerts (each with its chair, barber and `visit_entered_at`) |
| `GET /api/barber/alerts/{id}/snapshot` | the frame taken when the alert opened (JPEG, shop network only; 404 if none was saved). Alerts carry `has_snapshot`, `subject_cam`/`subject_box` and `ticket` |
| `GET/PUT /api/barber/policy` | tunable knobs |
| `GET /api/monitoring/zones`, `PUT /api/monitoring/zones/{cam}` | zones |
| `GET /api/monitoring/zones/problems` | zone-pairing problems across cameras, as structured codes (below) |
| `GET /api/barber/roster`, `PUT /api/barber/roster/{chair_id}` (`{"staff_id": str \| null}`) | who works which chair; `?detail=1` returns `{chair_id: {staff_id, since}}` instead of `{chair_id: staff_id}` |
| `GET/POST /api/barber/roster/confirmation` | today's (shop day's) roster confirmation: `{day, confirmed, confirmed_at}`; `POST` is idempotent |
| `/api/staff` (CRUD) | barbers |
| `WS /ws` | `live_snapshot` on connect, on every change and every 5 s; `scene` up to `scene_fps` times a second |

What each statistic counts, and when:
- **Customer visits:** by entry time.
- **Haircuts:** by end time.
- **Average and longest wait:** customers who reached a chair.
- **Excessive-wait cases:** every measured wait, walk-outs included.
- **Customers currently waiting:** a live value, taken from the running engine.
- **Served customers (`served_customers`):** haircuts that started in range. **Visitors (`visitors`):** customers who came through the door (door crossings in).

The zone display switches are the policy keys `zone_show_entrance_line`, `zone_show_chair`, `zone_show_seat`, `zone_show_waiting`, `zone_show_washing` and `zone_show_row` (1 = draw that zone type on the dashboard, 0 = hide it); the live payload carries them as `zone_display` (`{type: bool}`).

### The shop day

"Today" is not midnight to midnight: it runs from `day_start_hour` (policy
knob, default `5`, i.e. 5am) to the same hour the next calendar day, so a
haircut just after midnight still counts toward the evening it belongs to.
`shopday.py` holds the pure functions (`shop_day`, `shop_day_range`,
`day_start`); a `from`/`to` of just a date (`YYYY-MM-DD`) on `/stats` or `/alerts`
is resolved against it, and `/stats`' day/week/month buckets start
at that hour too (hour buckets are plain clock hours). The live snapshot's
`shop_day` is today's date under this rule (with `day_start_hour` alongside,
so the dashboard can place a timestamp on its shop day), and daily ticket numbers (below) and the roster
confirmation both reset on this boundary, not at midnight.

### Daily ticket numbers and `state_since`

Each visit is given a small, human-readable **ticket number** (`#1`, `#2`, …)
the moment it's confirmed, counting up within the shop day and restarting at
the next one (`TicketCounter` in `shopday.py`, seeded from the day's
`max_ticket` on startup so a restart doesn't repeat numbers). It's on the
visit row, the live customers list and the WS `scene` entries as `ticket`;
an unconfirmed visit has `ticket: null`. `state_since` is the ISO timestamp
of the customer's current state (waiting / haircut / washing), so the UI can
render its own ticking timer instead of polling.

### Structured zone problems

`GET /api/monitoring/zones/problems` reports each inconsistency as a dict
with a `code` the caller (the UI) translates, not a pre-rendered string:

| Code | Fields | Meaning |
|---|---|---|
| `pair_single_camera` | `pair`, `camera` | a paired area drawn in only one camera |
| `pair_twice_in_camera` | `pair` | the same pair id drawn twice in one camera |
| `pair_owner_count` | `pair`, `count` | a pair without exactly one `owner: true` zone |
| `chair_without_seat` | `camera`, `zone` | an owner chair with no `seat` zone (`zone` is the chair's name) |

### Crops are LAN-only

`GET /api/barber/crops/{kind}/{row_id}` (customer face/body photos) refuses
non-local clients with `403`: `netguard.is_local_client(host)` allows
loopback, private (RFC1918) and link-local addresses only, everything else is
refused. Behind the Vite dev proxy every request arrives from `127.0.0.1`, so
the proxy forwards the browser's address (`xfwd: true`) and the app trusts
`X-Forwarded-For` from loopback only (uvicorn's `ProxyHeadersMiddleware`,
added in `create_app`); the guard then judges the right-most untrusted address,
so a spoofed header from a remote browser does not help. This is a
network-position guard, not authentication -- **a login is
required before any real deployment**, since anyone on the shop's LAN can
currently fetch a crop by id.

### Reports

`GET /api/barber/report?kind=day|week|month&start=YYYY-MM-DD&lang=en|ar&format=html|pdf|docx` (the Insights page's
Report panel) builds one branch's report from the saved rows: a shop day, 7 shop days from `start`, or `start`'s
month (shop days from `day_start_hour`; a running period up to now, a future one `400`). Figures and charts as the
Insights page, haircuts per chair with start / end / duration, alert counts and snapshots (week / month: the 12
longest; the full lists in an appendix, collapsible in HTML). `report_data.py` builds the model, `report_html.py`
the page (RTL for Arabic), `report_pdf.py` prints it with headless Chrome, `report_docx.py` writes Word
(python-docx). File names: `11oct_BekkyBarber_palmhills_analysis_report.pdf`, `11oct_17oct_...` (week), from the
site yaml's `place` / `branch_slug`. Layout (2026-10-15): a header band with the IRIS mark and the shop's logo (the site yaml's `logo:`, png / jpg /
svg; unset = a text wordmark), figure cards, shaded tables, A4 pages numbered "Page X of Y"; the panel's "HTML file" downloads
the page (`download=1`). Snapshots only for shop-network clients (as the crops). **PDF and Word need
Chrome or Chromium on the host** (`google-chrome` / `chromium` on PATH); without it they answer `503`, HTML works.

### Haircut photo report

`GET /api/barber/report/photos?start=YYYY-MM-DD&lang=en|ar&format=html|pdf[&download=1]` (2026-10-22; the Report
panel's "Haircut photos" line, "Daily" only): one shop day, per chair every haircut with a start photo and an end photo
of its chair, side by side (`photo_report.py`, the analysis report's header and clock; about three haircuts a page).
**Shop network only** (`403` otherwise; `/api/barber/report/photos/allowed` tells the panel). Off unless the site sets
`hp_photos: 1` (the Palm Hills sites do).
- **When.** The chair engine (`photo_plan.py`) asks for a start photo when a sitting opens a new haircut, then an end
  photo every `hp_end_every_s` (120) -- the newest is kept -- only while the chair camera's newest frame has a non-staff
  person on the seat; none while the customer is away or while re-ID decides who took the seat. A sitting that never
  becomes a haircut drops its photos.
- **How.** `haircut_photos.py` (a background thread, never the engine's) fetches one go2rtc frame
  (`/api/frame.jpeg`, 3 s timeout), cuts the chair's close-up (its outline joined with the sitter's box, widened by
  `hp_crop_margin`, raised by `hp_crop_head`), saves `<crops_dir>/haircuts/<haircut id>_<kind>.jpg` at most
  `hp_photo_px` wide and links it in `haircut_photo`. go2rtc down: no photo ("No picture"), one warning a minute.
- **Kept** as alert snapshots: `customer_retention_days`, and a recording lap's clear removes them. A saved photo report
  keeps its pictures for good (Saved reports).
- `dev/photos/contact_sheet.py` cuts close-ups from recordings at a written report's haircut times, to tune the crop.
- Spec and plan (gitignored): `docs/superpowers/specs/2026-10-22-barber-haircut-photos-design.md`,
  `docs/superpowers/plans/2026-10-22-barber-haircut-photos.md`.

### Saved reports

`saved_reports.py` (2026-10-25; spec (gitignored) `docs/superpowers/specs/2026-10-25-barber-saved-reports-design.md`), on when the
site yaml has `saved_reports: {langs: [en, ar], formats: [pdf, html], week_start: sat}` (the Palm Hills sites; the old
`daily_report` key is refused with its new name). Saved into `<data_dir>/reports/`, never deleted by the service:
- `day/<YYYY-MM-DD>/` -- when the shop day ends (the site's `day_start_hour`: 09:00 at Palm Hills, in its `policy.json`): the analysis report and, with `hp_photos`,
  the haircut photo report; up to 7 days back after a service that was down; a day without visits gets none;
- `week/<Saturday>/` after its Friday, `month/<YYYY-MM>/` after its last day: the analysis report;
- a run on recordings: `day/<footage day>/lap-<lap end>/` at each lap's end, nothing by the clock (a clip run without
  `footage_start` saves nothing).

Each folder's `report.json` is written last (a folder without one is not finished); the PDFs are printed on the
saver's thread from `.print/` (a restart finishes them); Chrome's temp folders are in `<data_dir>/.tmp/`.
`GET /api/barber/reports/saved?kind=day|week|month` lists them (the dashboard's **Reports** page);
`GET /api/barber/reports/saved/file?id=<folder>&name=<file>` serves a listed file, **shop network only**.
The old flat files: `just barber-reports-migrate data/<site> [--laps] [--label NAME=TEXT] [--apply]`, once.

## Dashboard (Barber Ops)

The dashboard lives in the `frontend` submodule on the branch
**`feat/barber-multicam`** (a rebuild of `feat/barber-ops`, which grew out of
the attendance dashboard, `feat/fire-smoke-tab`, and uses the same visual
system). It is bilingual, English and Arabic (RTL) -- see
`frontend/README.md` for the frontend's own docs (run, tests, i18n rules).

- **Start it:** run `just frontend-use barber` once, then `just barber-up`,
  which serves it on **:5190**.
- **What Vite proxies:** `/barber/api` and `/barber/ws` go to this service;
  `/go2rtc` goes to go2rtc: live video is JPEG frames over go2rtc's
  WebSocket (`/go2rtc/api/ws?src=<cam>`, one socket per tile, so more than
  the browser's 6 connections per host), stills are plain HTTP.
- **In production** a reverse proxy has to do what Vite does: on `/go2rtc`
  strip the prefix, upgrade WebSockets (`Upgrade`/`Connection` headers,
  HTTP/1.1), turn buffering off and allow long read timeouts (a camera can
  stall). Keep go2rtc's `api.origin: "*"`. Each open dashboard costs one
  ffmpeg MJPEG transcode per camera on the host.
- **Why video comes from go2rtc:** it is the raw video, so every overlay is
  drawn from this service's own scene data.
- **How the overlay stays in step with the video:** the WebSocket carries two
  messages.
  - `live_snapshot` holds everything. It goes out on connect, on every change,
    and every 5 s.
  - `scene` holds the person boxes (each with its chair test point, the
    chair it falls in and whether it counts) and each chair's occupancy
    state. It goes out up to `scene_fps` times a second (policy, default 5,
    max 30, 0 = off), and only when something changed.
  - Raise `scene_fps` for smoother boxes. It is only useful up to the
    sidecar's `sample_fps`, since boxes can't change faster than it samples.
  - Change it in Setup → Advanced settings or `PUT /api/barber/policy`; it
    takes effect without a restart.
  - Video and boxes travel separate paths, so expect boxes to lead or trail
    the picture by a fraction of a second.

It has six screens:

| Screen | What it shows |
|---|---|
| Live | A Shop board / Cameras toggle. The **board** shows each chair as a card (barber, ticket, running timer) and the waiting queue, coloured against the targets, plus a side panel of open alerts and today's figures. If the day's roster isn't confirmed yet, a banner offers **Confirm chairs**. The **cameras** view shows the raw feed from each configured camera with the scene overlay (person boxes, each chair's occupancy state) drawn on top, so it can be checked against the board's tickets. |
| Insights | Filters, ten KPIs, three charts, and the visit table with an expandable timeline. |
| Customers | The customer register: a list with a thumbnail, sorted by last seen or visit count, and a detail panel (visits, stored face/body crops). Crops only render for LAN clients (see below). |
| Alerts | The alerts log. |
| Staff & chairs | The barber list (add / rename / retire) and the chair roster editor, with the shop day's confirm-chairs banner. |
| Setup | The zone editor (entrance line, waiting areas, chairs with a seat, the wash station, paired across cameras), a live area-consistency table, targets and advanced knobs (including `day_start_hour`). |

## Customer register

Project C: recognise a returning customer by face, anonymously -- no name is
ever stored, only a generated customer id.

**What's captured.** The sidecar publishes candidates for every track it
samples, customer or not: faces on `iris:customer_face` (a crop file plus a
512-d unit ArcFace embedding), bodies riding on `iris:appearance` (an item's
`crop_path` / `crop_quality`, next to its re-ID embedding and `staff_p`).
barber-service pools a candidate into a visit only when its track is bound to
that open visit and is not staff; every other candidate's crop file is deleted
at once. A visit spanning two cameras (a handoff mid-cut) pools candidates
from every track bound to it, so it registers once.

**Staff gates.** A track can be bound to a customer's visit and still show a
barber (a tracker id swap on the seated customer's track, a seat claim, or a
vote not decided yet), so binding alone isn't trusted:
- a body sample whose own `staff_p` is at or above `staff_threshold` is never
  pooled;
- a face is pooled only while its track's staff vote isn't "staff" and the
  mean of its last `staff_min_samples` samples is below `staff_threshold`
  (the mean catches a latched customer track, which the vote never calls
  staff, now showing the barber);
- when a track is voted staff (or released as staff), everything it already
  pooled is dropped from every open visit and its crops deleted.
Bodies are ranked by `1 - staff_p` (how unlike staff the crop looks), not by
crop height, which favoured a standing barber over a seated customer.

**Quality gates.** The sidecar only publishes a face crop whose box's shorter
side is at least 35 px at capture resolution, is sharp enough and not too far
off-axis (see `dev/faces/eval_faces.py`). Within a visit, a face too unlike
the visit's medoid face (below `visit_face_min_sim`) is dropped as a stray;
if fewer than `min_visit_faces` remain, the visit has no usable face and is
still registered, just with `face_count: 0` -- it is never matched against
later.

**Caps.** Per visit: at most `visit_max_faces` (15) face and
`visit_max_bodies` (10) body candidates are kept, best quality first. Per
customer: at most `customer_max_faces` (5) face and `customer_max_bodies` (3)
body crops are kept as the customer's reference set; adding a better one
evicts the worst.

**Matching.** Matching happens once, when a visit ends, and only for visits
with at least one completed haircut (a walk-out with no haircut registers
nobody, and its crops are deleted). Every known customer is scored by the
**median**, over the visit's clean faces, of each face's best similarity to
that customer's stored faces; the top-scoring customer is linked only if its
score clears `match_threshold` **and** leads the runner-up by at least
`match_margin`. Otherwise the visit becomes a new customer. This is
deliberately conservative: two different people are almost never merged, but
a returning customer is sometimes counted twice.

`match_threshold` (0.62), `match_margin` (0.05) and `visit_face_min_sim`
(0.43) are strict and provisional: the 2026-09-25 gate check on this footage
had too thin a sample of real returning customers to tune them properly, and
a face this far off failed to clear a sensible bar. A re-test with more
footage is planned; until then these knobs favour missing a match over
merging two different people. Live scoring against real returning customers
is postponed with project B's scoring work -- it needs footage spanning more
than one day.

**Retention and the orphan sweep.** Nightly at 03:00, customers with no
visit in `customer_retention_days` (365) are deleted -- their crop files go
with them, but their visits stay, with `customer_id` set to `NULL`. The same
pass sweeps `crops_dir` for files no register row references and older than
a day (candidates of visits still running are younger, so they're never
mistaken for orphans); it also removes any day directory left empty. This
catches crop files a delete couldn't remove earlier, a loop reset or restart
mid-visit, and anything else that fell through the cracks. The sweep also
runs once when the consumer starts.

**API.**

| Endpoint | What |
|---|---|
| `GET /api/barber/customers?limit&offset&sort=last_seen\|visit_count` | paged customer list with a thumbnail |
| `GET /api/barber/customers/{id}` | one customer's detail (visits, stored crops) |
| `DELETE /api/barber/customers/{id}` | delete a customer and its crop files; their visits keep with `customer_id: null` |
| `GET /api/barber/crops/{kind}/{row_id}` | serves one crop file, only for a `customer_face` / `customer_body` row that still exists in the register (path traversal and orphaned/unreferenced files are refused) |

**Stats.** `GET /api/barber/stats` adds `returning_visits` (`count` /
`percent` of visits in range whose customer had a prior visit) and
`avg_days_between_visits` (mean gap, in days, between a returning customer's
consecutive visits).

**Where the crops live.** Under one root, `data/barber/crops/` at the repo
root (`BARBER_DATA_DIR` / `crops_dir` in config); the sidecar writes it as
`/workspace/iris/data/barber/crops`. Layout: `faces/<YYYY-MM-DD>/<uuid>.jpg`
and `bodies/<YYYY-MM-DD>/<uuid>.jpg`. Every `crop_path` stored anywhere is
relative to that root. The sidecar runs as root and writes crop directories
`0o777` and files `0o666` so this (non-root) service can still delete them.

## Known limits (to validate on real footage)

- **The staff score is from a trained classifier.** The uniform probability comes from a trained staff/customer classifier (DINOv2 ViT-S/14 fine-tuned on labelled crops of this shop, `models/staff/staff_dinov2_s.onnx`, selected by the sidecar's `appearance_staff_model: classifier`; zero-shot CLIP stays as the fallback, `appearance_staff_model: clip`).
- **Classifier limits.** It has only seen this shop's staff and uniforms, so new staff looks, a uniform change or a second shop need new labelled crops and a retrain; `staff_threshold` 0.70 is set for its calibrated score; `seat_staff_threshold` (0.9) is kept for now, pending a decision in the B backlog.
- **No entrance line means no observed entries.** Without one, visits are only
  counted once someone waits or sits, and exits fall back to the lost timeout.
- **Tracker re-ID is off.** The ReIdentificationNet model is wired in
  (`just barber-reid-model` fetches it), but barbers dress alike, and on the
  test clip re-association made barber names change hands more often. See the
  tracker config header to switch it back on.
- **Thresholds are tuned on test clips only.** The dated notes in `policy.py`
  come from replays of recorded laps of the test clips against a
  hand-annotated ground truth (`dev/replay_raw.py`), not from real footage.
- **Restarts.** A restart closes still-open rows as `interrupted`. Their
  haircuts stay unmeasured and are excluded from duration stats.
