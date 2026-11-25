# Running the products

One checkout can run one product, several, or all three. What each command
starts is below, along with the parts that are **not** wired up yet — so you
know when something is missing by design rather than broken.

Everything reads this checkout's `.env`: ports, the pipeline's settings file
(`SIDECAR_CONFIG`) and which products the dashboard offers (`IRIS_PRODUCT`).
Two checkouts on one machine must differ in `COMPOSE_PROJECT_NAME` and every
port — see `.env.example`.

## Attendance (with fire and smoke)

```bash
just up live          # the three real cameras, taken from the other checkout's go2rtc
just up cam-21        # a recorded clip instead: set SIDECAR_CONFIG=tests/configs/attendance-clip.yaml
just down             # stops everything
```

Starts: redis, minio, the IRIS app and go2rtc; the pipeline running
`products/attendance/sidecar.yaml`; the attendance service on
`ATTENDANCE_BACKEND_PORT`; the dashboard on `ATTENDANCE_FRONTEND_PORT`; and the
watchdog that restarts a stalled pipeline. It also re-registers employee faces
into Redis, which has no persistent volume.

Open the dashboard and pick **الحضور والانصراف**.

## Barber shop

```bash
just barber-up barber-1     # barber-1 = its recorded clip; omit for a live camera
just barber-down            # stops barber only; shared containers keep running
```

Starts: the shared containers, the pipeline running `products/barber/sidecar.yaml`
(compose profile `barber`), the clip streamer, the barber service on
`BARBER_BACKEND_PORT`, and it rebuilds the staff face gallery.

**It does not start the dashboard.** `just up` does — and `just up` also starts
attendance's service. For barber alone, run the dashboard directly:

```bash
cd frontend && set -a && . ../.env && set +a && npm run dev -- --host 0.0.0.0
```

Then pick **Barber Ops**, or open `?product=barber`.

## IRIS surveillance UI

It has no pipeline or service of its own: it reads the IRIS app (camera list,
camera images, alerts) on `IRIS_HOST_PORT`, which `just up` starts.

```bash
just up live
```

Then pick **IRIS Surveillance** in the switcher.

## All three at once

```bash
# .env: IRIS_PRODUCT=attendance,barber,iris
just up live
just barber-up barber-1
```

One dashboard, one address, all three products in the switcher. The two
pipelines run side by side on the same GPU; they own different cameras and
different Redis keys, so they do not collide. Watch the frame rates if the GPU
is busy with other work.

## Ports

| | Variable | This checkout |
|---|---|---|
| Dashboard (all products) | `ATTENDANCE_FRONTEND_PORT` | 5301 |
| Attendance service | `ATTENDANCE_BACKEND_PORT` | 8098 |
| Barber service | `BARBER_BACKEND_PORT` | 8097 |
| IRIS app | `IRIS_HOST_PORT` | 28100 |
| go2rtc web page | `GO2RTC_API_PORT` | 12084 |
| Redis | `REDIS_HOST_PORT` | 16479 |

## What is not wired up yet

- **`IRIS_PRODUCT` only decides the dashboard's menu.** `just up` always starts
  attendance's service, whatever it says, and barber has its own command. A
  product-aware `just up` is proposed in `MULTI_PRODUCT_PLAN.md` ("Proposed,
  not started").
- **The app container reads `IRIS_PRODUCT` when it is created**, so after
  editing `.env` the menu only changes once the container is recreated:
  `docker compose up -d --force-recreate iris`.
- **Cameras are pulled once per machine.** A second checkout must point
  `RAW_CAM_*_SRC` at the first checkout's go2rtc instead of at the cameras;
  pulling twice over the ~18 Mbit/s tunnel freezes both.
