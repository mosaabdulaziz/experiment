# safety

One tool for workplace safety: vehicle–pedestrian proximity, zone intrusion and overcrowding, PPE, falls and
fire/smoke, with the features toggled per camera. The app side is the IRIS app (`app/`) with its `events` table; the
UI is the `safety` product (`IRIS_PRODUCT=safety`). Plan and decisions: the safety integration story, tickets 1–3.

```
just safety-build          # once: the stack's images (or SAFETY_SIDECAR_IMAGE = an existing sidecar image)
just safety-up             # the models, then redis, minio, go2rtc, the app, the fall service, the controller and
                           # the role sidecars the cameras need
just safety-demo           # loop the demo cameras' clips into go2rtc (SAFETY_DEMO_CLIPS)
just safety-status         # containers, each role's cameras, controller / fall service / streamers
just safety-logs [what]    # follow the logs: all, or one role, iris, controller, fall-service, demo
just safety-gpu            # GPU memory and SM % of each role sidecar and the fall service
just safety-telegram-test  # a test message to the Telegram chat saved in Settings
just safety-down           # stop it all; the data dir stays
just safety-models         # the models it loads (safety.yaml `models`), from SAFETY_DVC_STORE, else dvc pull
just safety-test           # the product's backend, controller and console tests
```

`just --list` shows them under `[safety]`. The recipes run `uv`, which must be 0.13.x (`required-version` in
`pyproject.toml`).

It is its own compose project (`safety`) with its own ports and data dir (`scripts/safety_env.sh`, the `SAFETY_*`
settings in `.env.example`), so it runs beside any other stack on the machine and shares nothing with attendance: a
fire camera here is never seen by an attendance stack.

## One role sidecar per feature

| feature | role sidecar | what runs | zones |
|---|---|---|---|
| proximity | `safety-proximity` | the proximity product's pipeline (YOLO26l-seg 1280, depth) | `proximity` blocks |
| zones | `safety-zones` | base person detector + `zones` (intrusion, overcrowding, with end markers) | intrusion / overcrowding |
| ppe | `safety-ppe` | base person detector + the PPE detector on each person | `ppe` blocks, else whole frame |
| fall | `safety-fall` + the fall service | pose on each person; the decision in `products/fall_detection` | `fall` blocks, else whole frame |
| fire | `safety-fire` | the fire/smoke detector alone | `fire` blocks, else whole frame |

Each keeps the detector it was tuned on. Several run on one camera because the stack's sidecars use **role-scoped
keys** (`role_scoped_keys: true`): every per-camera Redis key carries `<camera>@<role>` (`iris:status:bay-a@ppe`,
`iris:meta:bay-a@ppe`, `iris:alert:bay-a@ppe`, ...), so two sidecars on one camera never overwrite each other's
status (and its `boot_id`), mix their frames or answer each other's clip requests (`perception/engine/sources.py`
`stream_key_id`). Every other deployment leaves the setting off and keeps its key names.

All roles share one mux size (`common` in `safety.yaml`, 1280x720): the zones' points are stored in mux pixels and
every role reads the same zones file. Each role records only the clips of its own alerts, in its own folder
(`data/alert_svr/<role>`); there is no continuous recording.

## Cameras and toggles

`cameras.yaml` defines the cameras (`rtsp:` for a real one, credentials as `${VAR}` from `.env`; `clip:` for a demo
clip) and each one's default features. The UI renames cameras and toggles features; it saves them to
`<data>/safety/cameras_state.json` and asks the controller to apply them (`iris:ctrl:safety_apply`).

`tools/safety_controller.py` renders `<data>/safety/generated/<role>.yaml` (the role file, the common settings, the
role's own clip folder, its cameras), the fall service's config and go2rtc's, then **stops** role sidecars no camera
needs any more, **starts** missing ones and **restarts** only those whose config differs from what was last applied
(`<data>/safety/applied.json`, recorded step by step, so a failed apply is retried by the next and `safety-status`'s
render applies nothing); the fall service restarts
when the fall cameras change. A new camera's streams are added to the running go2rtc through its API before any
sidecar pulls them, so the other cameras keep their pull (a stream whose `${VAR}` credentials the controller cannot
read is left to a go2rtc restart, `safety_compose restart go2rtc`). It runs on the host (`just safety-controller-restart`, started by `safety-up`): the app
never gets the Docker socket. `just safety-apply` applies once by hand.

## The app side (IRIS app, `IRIS_PRODUCT=safety`; code in this folder, see Files)

- **Every role, one view.** `SIDECAR_CONFIG` lists every role's generated config; the app merges them
  (`app/sidecar_config.py`): every module, and every source as its stream id `<camera>@<role>`. Streams and events
  carry `camera` (the stream), `camera_id` and `feature` (`app/features.py`); zones and filter zones are kept per
  camera. The app re-reads the configs (cached until a file changes) on each discovery pass: after an Apply the new
  streams get their smart-record clips and the ROI file is rebuilt for the zones role's new camera order.
- **Incidents.**
  - Proximity and fire: incidents as before.
  - Intrusion: one incident per zone, from the first entry to the zone's `intrusion_cleared`; later entries raise it
    in place with a follow-up clip.
  - Overcrowding: an incident until `overcrowding_resolved` (`app/usecases/zone_incidents.py`; only when the zones
    module sends end markers).
  - PPE and fall: one event per episode.
- **Camera health** (`products/safety/camera_health.py`), feature `health`:
  - `offline`: every stream of a camera sends nothing.
  - `feature_down`: a role is not running on it.
  - `slow`: below 1 frame/s.
  - Each opens after 15 s and resolves on recovery.
  - Read from Redis's stream counters only, never the frame metadata (it carries JPEGs), off the event loop.
  - Roles the controller is restarting (`iris:safety:applying:<role>`) are held: their open incidents stay open, and
    a camera with a held role is never called offline. The controller releases the hold as soon as the role is live
    (at most 300 s), and only then marks the Apply applied.
- **Telegram** (`products/safety/notifier.py`):
  - Off until Settings turns Telegram on. Then an incident's opening, and an escalation that raises its severity, are
    sent with the camera's frame when its feature has notify on and the severity is at least the feature's minimum. Another entry into an open zone
    incident is not an escalation.
  - The bot token is `TELEGRAM_BOT_TOKEN` in `.env` (never stored by the app, never sent to the browser); the chat
    is set in Settings. Setting it up:
    1. Create a bot with @BotFather in Telegram (`/newbot`); it answers with the token.
    2. Put `TELEGRAM_BOT_TOKEN=<token>` in the stack's `.env` and run `just safety-up` again (the app reads `.env`
       when its container is created).
    3. Add the bot to the group that should get the alerts. Find chats sees the group from that moment, for about a
       day. With the bot's privacy mode on (BotFather's default) it does not see ordinary group messages, so for a
       bot added earlier, mention it in a message there (`@<bot> hello`).
    4. Settings -> Telegram -> **Find chats**, choose the group, **Save settings**, then **Send test message**.
  - `POST /api/safety/telegram/test` sends that test message to the saved chat; `GET /api/safety/telegram/chats`
    lists the chats the bot was recently added to or written in (Telegram's `getUpdates`, nothing consumed). Both say
    what is missing (400) or Telegram's own reason when it refuses (502).
  - Sending is queued and retried, never in the alert's way.
- **`/api/safety/*`** (`products/safety/api.py`, logic in `products/safety/service.py`):
  - `GET|PUT /settings`: per feature `ack_required`, `notify`, `min_severity`; Telegram's switch (`enabled`) and
    chat; the language. Defaults: Telegram off; fall and fire require an acknowledgement; once Telegram is on, fall,
    fire and health notify, proximity when critical, and zones and PPE only show.
  - `GET /cameras`, `PUT /cameras/{id}` (`name`, `features`): saved to `<data>/safety/cameras_state.json`.
  - `GET /cameras/{id}/snapshot`: a JPEG still of the camera from go2rtc's `raw-<id>` (`SAFETY_GO2RTC_URL`, default
    `http://go2rtc:1984`), one fetch per camera at a time, shared for 1.5 s.
  - `POST|GET /apply`: bumps `iris:ctrl:safety_apply`; `applying` until the controller writes
    `iris:analytics:safety_applied`.
  - `GET /unacknowledged`: incidents of features that require an acknowledgement and have none yet (ended ones too),
    filtered by feature in the query, so frequent alerts of other features never push them out.
    `POST /api/events/{id}/acknowledge` acknowledges.
  - `GET /fall/candidates`, `POST /fall/review/{id}`: the fall service's review queue (`SAFETY_FALL_URL`).
- **Zones** carry `ppe` / `fall` / `fire` blocks (`{"enabled": true}`) beside `proximity`
  (`/api/streams/{camera}/zones`). The `ppe` block may name the equipment checked in that zone (`"items": ["helmet"]`,
  from the PPE module's `required_items`; refused otherwise); GET lists the choices as `ppe.items`.

## The console (`frontend/src/products/safety`)

The `safety` dashboard product, the one the stack's app shows (`IRIS_PRODUCT=safety`). English by default, Arabic
(right to left) from the top bar; the stack's default language is a setting.

- **Live**: the camera wall and the latest incidents. Each role sidecar draws its own overlay, so a tile shows one
  feature's view at a time; its signs switch between them. A camera with an incident going on is outlined. A camera
  no feature runs on stays on the wall as a **Preview**: a still from go2rtc's `raw-<id>`
  (`GET /api/safety/cameras/<id>/snapshot`, no sidecar, refreshed every few seconds), and its zone editor draws on that
  still, so zones can go in before a feature starts. Above the incidents, **Needs setting up** lists the features
  switched on that cannot raise anything yet (proximity, or intrusion / crowding, with no zone for them), each with a
  button into that camera's zone editor.
- **Configure** (each tile's gear button; there is no separate Cameras page): iris's zone editor with the camera's
  settings under the picture -- its name and feature switches (saved at once, run after **Apply**, which says which
  features restart) -- only the sections of the camera's features, where each feature looks, and a warning when a
  change takes PPE / fall / fire off the whole frame.
- **Incidents**: every feature's incidents by day, filtered by feature, camera, severity and state (`?status=unacked`,
  `?feature=fire,ppe`, `?camera=`); **Needs review** lists the fall service's borderline falls to confirm or dismiss.
- **Health**: camera-health incidents still going on, and each camera's features with their state.
- **Settings**: acknowledgement, Telegram and its minimum severity per feature; the chat (typed, or found among the
  chats the bot is in) and a test message; the default language.
- **The banner**, on every page: incidents waiting for an acknowledgement, until someone gives one. No sound.

The pure logic (filters, coverage, Apply's diff, the feed, routes) is in `lib/` with its tests: `npm test -w
@iris/safety`. Incident messages and zone-editor labels come from the server and iris in English.

## Files

- `config.py` -- the camera file, the UI's toggles and their merge, and the Apply keys: one copy for the app and the
  controller
- `service.py`, `api.py`, `notifier.py`, `camera_health.py` -- the product's app-side code (settings, cameras and Apply,
  the acknowledgement queue, the fall proxy; Telegram; camera health), loaded by the IRIS app (`app/server.py`) only
  when `IRIS_PRODUCT` names safety. The platform pieces it relies on stay in `app/`: `features.py` (camera and feature
  of a stream or event), `usecases/zone_incidents.py`, the role-config merge in `sidecar_config.py`
- `safety.yaml` -- the features, the role file each extends, the settings every role shares, the fall service's base
- `roles/<role>.yaml` -- what each role runs (each extends a product's or preset's own sidecar config)
- `cameras.yaml` -- the cameras (the internal demo's four by default; `SAFETY_CAMERAS` names another file)
- `fall_detection.yaml` -- the fall service in this stack (`key_role: fall`, its port)
- `docker-compose.safety.yml`, `scripts/safety_env.sh`, the `safety-*` recipes -- the stack

## Licences

The PPE detector's weights are AGPL, approved for non-commercial use only
(`perception/models/DeepStream-Yolo-PPE/README.md`), and the fire detector's are accepted for demos only
(`perception/models/DeepStream-Yolo-Fire/README.md`). The stack is for the internal demo until that is resolved.
