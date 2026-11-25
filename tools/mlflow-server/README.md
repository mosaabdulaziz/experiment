# Central MLflow server (cmubuntu)

Sub-task 1 of the MLflow story (spike `docs/spikes/mlflow-tracking-registry.md` §7). One MLflow tracking server for
the team, on cmubuntu, with its artifacts on SharePoint until the MinIO from #83 exists.

## Using it

| | |
|---|---|
| Tracking URI | `http://10.2.0.50:15050` (also `http://cmubuntu.cairomotive.com:15050`, `http://cmubuntu:15050`) |
| UI | the same URL in a browser (basic auth prompt) |
| Auth | MLflow basic auth. Ask the admin for a user (`serverctl.sh create-user <name>`) |

The tracking helper (`mtmc/tracking`) and any MLflow client read:

```bash
export MLFLOW_TRACKING_URI=http://10.2.0.50:15050
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<password>
```

On cmubuntu itself: `set -a; . ~/mlflow-server/secrets/client-iris.env; set +a`. The helper's default stays the
offline store when `MLFLOW_TRACKING_URI` is unset; nothing in the code points at this server.

Runs record their artifacts as `mlflow-artifacts:/<experiment>/<run>/artifacts/...`: clients upload and download
through the server (`--serve-artifacts`), never to the storage directly, so moving the storage later changes nothing
for them.

## How it is deployed

```
                       LAN 10.2.0.50:15050                     127.0.0.1:15432
clients ── basic auth ──> mlflow server (host process, vlm) ──> Postgres 18.6 (Docker, iris-mlflow-postgres)
                                │ --serve-artifacts                tracking + registry: db mlflow, users: db mlflow_auth
                                ▼ --artifacts-destination ~/mlflow-server/artifacts/store
                          rclone mount iris-onedrive:IRIS/repo-data/mlflow-artifacts (VFS write cache, 3 GB)
                                ▼ uploads in the background (~6 MB/s measured, wired)
                          SharePoint PhyAI drive, IRIS/repo-data/mlflow-artifacts/store/
```

- **The server is a host process, not a container**: Docker cannot read a FUSE mount owned by vlm without
  `user_allow_other` in `/etc/fuse.conf`, which stays off.
- **Ports**: 15050 (MLflow, LAN interface `enp6s18` only), 15432 (Postgres, 127.0.0.1 only). The dev server
  (`docker-compose.mlflow.yml`) keeps 15000.
- **Deployed copy**: `install.sh` copies this folder to `~/mlflow-server/app` and syncs `~/mlflow-server/venv` from
  `uv.lock` (mlflow[auth] 3.18.0, the helper's version; psycopg 3). Cron runs the copy, so switching branches in the
  clone never touches the running server. Re-run `install.sh` after changing anything here.
- **Postgres**: `postgres:18@sha256:74935e72…` (pinned digest, named volume `iris-mlflow-pgdata`,
  `restart: unless-stopped`, compose project `iris-mlflow`).
- **Artifact cache**: `--vfs-cache-mode writes`, `--vfs-cache-max-size 3G`, `--vfs-cache-max-age 1h`, in
  `~/mlflow-server/rclone-cache`. A logged file lands in the cache at disk speed (350 MB in 1.7 s) and uploads in the
  background. 3 GB keeps the machine above its 20 GB free-disk floor. A file larger than the cache still passes:
  rclone cannot evict a file before it is uploaded, so the cache can exceed 3 GB during one big upload.

### Secrets (never in the repo)

`~/mlflow-server/secrets/` (mode 700, every file 600), generated once by `install.sh` and kept:

| file | what |
|---|---|
| `postgres.env` | `POSTGRES_PASSWORD` for the container |
| `pgpass` | the same password for libpq (`PGPASSFILE`): no password in any URI or command line |
| `basic_auth.ini` | MLflow basic-auth config, incl. the admin password (used only to create `admin` on first start) |
| `server.env` | `MLFLOW_FLASK_SERVER_SECRET_KEY` (basic-auth CSRF key, stable across restarts) |
| `client-<user>.env` | `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` of a user made by `serverctl.sh create-user` |

### Never writing artifacts to the local disk

MLflow writes under `artifacts/store/`. With the mount up that is a SharePoint folder; with it down, the local folder
holds a read-only **file** named `store`, so every artifact write fails ("Not a directory") instead of filling the
disk. On top of that, `serverctl.sh` refuses to start the server unless `mountpoint -q` passes, the mount is
`iris-onedrive:IRIS/repo-data/mlflow-artifacts`, and a test file round-trips (write, read back, delete) through it.
The watchdog stops a running server as soon as its mount fails that check (e.g. rclone died), clears the dead FUSE
mount, remounts and restarts.

## Operating it (`~/mlflow-server/app/serverctl.sh`)

| command | |
|---|---|
| `start` | mount, Postgres, mount check, server (each idempotent) |
| `stop` / `stop --all` | stop the server / also unmount and stop Postgres. Leaves a hold: the watchdog will not restart it |
| `status` | mount, server `/health`, Postgres, uploads to SharePoint still pending or in progress, cache size. Exit 0 all up and nothing pending, 1 something down, 3 uploads pending |
| `ensure` | the watchdog (cron, every 2 min) |
| `boot` | the `@reboot` entry: waits for Docker and SharePoint, then `start` (clears a hold) |
| `server` | start only the server step (never mounts; refuses without a good mount) |
| `create-user <name>` | a non-admin user; credentials in `secrets/client-<name>.env` |

**Boot and restart**: vlm has no systemd lingering (`loginctl show-user vlm -p Linger` = no), so systemd --user units
would not start at boot. `install.sh` installs a crontab block instead:

```
@reboot      serverctl.sh boot       # mount -> check -> Postgres -> server
*/2 * * * *  serverctl.sh ensure     # restart a crashed server, or one whose mount died
30 3 * * *   backup.sh               # daily pg_dump to SharePoint
15 * * * *   logrotate (user state)  # logs/*.log: 20 MB, 5 compressed rotations
```

Logs: `~/mlflow-server/logs/` (`mlflow-server.log`, `rclone.log`, `serverctl.log`, `backup.log`, `cron.log`).

**Upload status**: the mount runs rclone's remote control on a unix socket, `~/mlflow-server/run/rc/rclone.sock`
(folder mode 700, so only vlm can reach it; no TCP port). `status` asks it for `vfs/stats`: uploads in progress,
queued (rclone waits ~5 s after a file is closed before uploading it) and errored. If the socket is missing (a mount
started by an older `serverctl.sh`), `status` counts the files rclone marks dirty in its cache metadata instead.

### Before shutting down or rebooting the machine

A logged artifact is on SharePoint only once rclone has uploaded it (see Known limitations). Before a shutdown or
reboot, run status and wait until it says `uploads:  none pending` (exit code 0):

```bash
~/mlflow-server/app/serverctl.sh status
# or wait for it:
until ~/mlflow-server/app/serverctl.sh status; [ $? -ne 3 ]; do sleep 10; done
```

To keep new uploads from arriving while you wait, stop the server first (`serverctl.sh stop`); the mount stays up and
keeps uploading. A reboot leaves nothing to do afterwards: `@reboot` starts everything again. Files still in the cache
at shutdown are not lost (rclone uploads them on its next mount from the same cache folder), but until then they exist
only on this machine's disk, and a client that downloads them in the meantime gets an error.

## Backups

`backup.sh` (daily, 03:30): `pg_dump -Fc` of `mlflow` and `mlflow_auth`, uploaded to
`iris-onedrive:IRIS/repo-data/mlflow-backups/`, newest 14 of each kept. Artifacts are on SharePoint already.

`restore_test.sh`: restores the newest dumps into a throwaway Postgres container (same image, no network, no
volume), compares row counts with the live database, removes the container.

Full restore: `serverctl.sh stop`, `docker exec -i iris-mlflow-postgres pg_restore -U mlflow -d mlflow --clean
--if-exists < mlflow-<stamp>.dump` (and `mlflow_auth`), `serverctl.sh start`.

## Known limitations

### Artifacts exist only on this machine until rclone uploads them

A client's upload ends when the file is in rclone's VFS cache (`~/mlflow-server/rclone-cache`), not when it is on
SharePoint. rclone uploads it in the background (~6 MB/s measured): a 350 MB model takes about a minute. Until then the
file exists only on this machine's disk, so losing the disk or the cache folder loses it, and a shutdown delays it (see
"Before shutting down or rebooting the machine"). `serverctl.sh status` shows what is still waiting. A failed upload
is retried by rclone and shows as `ERRORED` in status; `logs/rclone.log` has the reason.

### The SharePoint token is Sarah's personal account

The `iris-onedrive` rclone remote (`~/.config/rclone/rclone.conf`, mode 600) holds an OAuth token for **Sarah's own
Microsoft account**, on the shared `vlm` Linux user. Consequences:

- Anyone who can log in as vlm can read and write everything Sarah can on SharePoint, not only the IRIS folders.
- Everything the server writes appears on SharePoint as written by Sarah.
- If Sarah's account loses access to the PhyAI drive, is disabled, changes its password or has its sessions revoked,
  or the token goes unused for 90 days, the mount fails: the watchdog stops the server (by design) and `rclone.log`
  shows the token error. Backups fail too.

To re-authenticate it (as Sarah again, or as someone else with write access to the PhyAI drive's `IRIS/` folder),
from a machine with a browser:

1. On that machine, install rclone (same or newer version as `~/.local/bin/rclone version` here) and run
   `rclone authorize "onedrive"`. Sign in in the browser that opens with the account the server should use. rclone
   prints a token JSON blob in the terminal: copy it (it is a credential, so send it by no channel others can read).
2. On cmubuntu, as vlm: `~/mlflow-server/app/serverctl.sh stop`, then `~/.local/bin/rclone config reconnect
   iris-onedrive:`. Answer **n** to "Use web browser to automatically authenticate?" and paste the blob.
   If it asks for the drive again, choose the same SharePoint site (PhyAI) and the same document library, so that
   `drive_type` stays `documentLibrary` and `drive_id` stays the same (to compare without printing the token:
   `grep -E '^drive_(id|type)' ~/.config/rclone/rclone.conf`, before and after).
3. `~/.local/bin/rclone lsf iris-onedrive:IRIS/repo-data/mlflow-artifacts`, then `serverctl.sh start` (it remounts
   and runs the mount check), then `serverctl.sh status`.

A service account or app registration (client credentials) would remove the dependency on one person; until the move
to MinIO this is the accepted trade-off.

### HTTP basic auth, no TLS, LAN only

The server speaks plain HTTP on `10.2.0.50:15050`. Usernames and passwords (basic auth) and all artifacts cross the
network unencrypted, so anyone on the LAN who can capture traffic can read them. It is bound to the LAN interface
only and must not be exposed beyond it (no port forward, no VPN route to the internet). Each user should have their own
account (`create-user`) with a password used nowhere else. Default permission for logged-in users is READ.

### Temporary: until the MinIO from #83

SharePoint through rclone is a stopgap. Once MinIO exists, move the artifacts there:

1. Announce a short write freeze, then `serverctl.sh stop` and wait for `serverctl.sh status` to show
   `none pending` (the mount stays up and finishes its uploads).
2. Copy the artifacts, server side from SharePoint to the bucket:
   `rclone copy iris-onedrive:IRIS/repo-data/mlflow-artifacts/store <minio-remote>:<bucket>/<prefix> --transfers 8`,
   then `rclone check iris-onedrive:IRIS/repo-data/mlflow-artifacts/store <minio-remote>:<bucket>/<prefix> --size-only`
   (SharePoint and S3 hashes differ) until it reports 0 differences.
3. In `serverctl.sh`, switch `--artifacts-destination "$STORE"` to `--artifacts-destination s3://<bucket>/<prefix>`
   and pass `MLFLOW_S3_ENDPOINT_URL` plus `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from a new file in `secrets/`
   (never on the command line). Drop the mount steps (`do_mount`, the mount check before the server, the watchdog's
   mount check) and the upload lines in `status`. Add `boto3` to `pyproject.toml` and re-lock (`uv lock`); it is not
   in the lock today. Update the tests in `tests/test_mlflow_server_deploy.py` that pin the mount check.
4. `install.sh`, `serverctl.sh start`, then open an old run's artifacts in the UI and log a new run to confirm both.
5. Keep the SharePoint copy read-only for a while as a fallback, then delete it, and the rclone remote with it.

Run records keep their `mlflow-artifacts:/` URIs (clients go through the server), so no database change and no client
change is needed.
