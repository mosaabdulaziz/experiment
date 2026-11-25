#!/usr/bin/env bash
# Install (or update) the central MLflow server on this machine from this folder. Idempotent; starts nothing.
#
#   tools/mlflow-server/install.sh            then: ~/mlflow-server/app/serverctl.sh start
#
# The deployment runs from $IRIS_MLFLOW_HOME (default ~/mlflow-server), not from the git checkout, so switching
# branches in the clone never pulls the scripts from under a running server or the @reboot entry:
#   app/        a copy of this folder's scripts, compose file, pyproject.toml + uv.lock (re-run install.sh to update)
#   venv/       the server's environment, synced from uv.lock (mlflow 3.18.0, psycopg 3)
#   secrets/    mode 700, files 600, never in the repo: postgres.env, pgpass, basic_auth.ini (admin password),
#               server.env (the basic-auth CSRF key), client-<user>.env (a user's credentials, written by
#               `serverctl.sh create-user`). Generated once; kept.
#   artifacts/  the rclone mount point; MLflow writes under artifacts/store/ (a guard file while unmounted)
#   rclone-cache/  rclone's VFS write cache
#   logs/       rotated by logrotate (cron)      run/  pid, lock, logrotate state      backups/  pg_dump staging
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
IRIS_MLFLOW_HOME="${IRIS_MLFLOW_HOME:-$HOME/mlflow-server}"
APP="$IRIS_MLFLOW_HOME/app"
SECRETS="$IRIS_MLFLOW_HOME/secrets"
LOGS="$IRIS_MLFLOW_HOME/logs"
RCLONE="$HOME/.local/bin/rclone"
PG_PORT=15432

mkdir -p "$APP/initdb" "$LOGS" "$IRIS_MLFLOW_HOME/run" "$IRIS_MLFLOW_HOME/backups" "$IRIS_MLFLOW_HOME/rclone-cache" \
         "$IRIS_MLFLOW_HOME/artifacts"
if ! mountpoint -q "$IRIS_MLFLOW_HOME/artifacts"; then      # the guard: see serverctl.sh
    chmod 755 "$IRIS_MLFLOW_HOME/artifacts"
    [ -f "$IRIS_MLFLOW_HOME/artifacts/store" ] || : > "$IRIS_MLFLOW_HOME/artifacts/store"
    chmod 444 "$IRIS_MLFLOW_HOME/artifacts/store"
fi
install -m 755 "$SRC/serverctl.sh" "$SRC/backup.sh" "$SRC/restore_test.sh" "$APP/"
install -m 644 "$SRC/docker-compose.yml" "$SRC/pyproject.toml" "$SRC/uv.lock" "$SRC/.python-version" "$APP/"
install -m 644 "$SRC/initdb/"*.sql "$APP/initdb/"
sed "s|@LOGS@|$LOGS|" "$SRC/logrotate.conf" > "$APP/logrotate.conf"
echo "installed scripts into $APP"

# -- secrets (generated once, never overwritten) -----------------------------------------------------------------
umask 077
mkdir -p "$SECRETS" && chmod 700 "$SECRETS"
rand() { head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c "$1"; }
if [ ! -f "$SECRETS/postgres.env" ]; then
    pg="$(rand 32)"
    echo "POSTGRES_PASSWORD=$pg" > "$SECRETS/postgres.env"
    echo "127.0.0.1:$PG_PORT:*:mlflow:$pg" > "$SECRETS/pgpass"
    echo "generated $SECRETS/postgres.env and pgpass"
fi
if [ ! -f "$SECRETS/basic_auth.ini" ]; then
    cat > "$SECRETS/basic_auth.ini" <<EOF
[mlflow]
# MLflow basic auth (mlflow server --app-name basic-auth). Users live in the mlflow_auth database; the password of
# the mlflow role comes from PGPASSFILE (secrets/pgpass), not from this URI. admin_password is used only to create
# the admin user on the first start; rotate it afterwards through the update-password API.
default_permission = READ
database_uri = postgresql://mlflow@127.0.0.1:$PG_PORT/mlflow_auth
admin_username = admin
admin_password = $(rand 28)
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF
    echo "generated $SECRETS/basic_auth.ini (admin password)"
fi
if [ ! -f "$SECRETS/server.env" ]; then
    echo "MLFLOW_FLASK_SERVER_SECRET_KEY=$(rand 40)" > "$SECRETS/server.env"     # basic-auth CSRF key
    echo "generated $SECRETS/server.env"
fi
chmod 600 "$SECRETS"/*
umask 022

# -- the server's environment -----------------------------------------------------------------------------------
UV_PROJECT_ENVIRONMENT="$IRIS_MLFLOW_HOME/venv" uv sync --project "$APP" --frozen --quiet --no-python-downloads
echo "synced $IRIS_MLFLOW_HOME/venv from uv.lock ($("$IRIS_MLFLOW_HOME/venv/bin/python" -c 'import mlflow; print("mlflow", mlflow.__version__)'))"

# -- the SharePoint folders ---------------------------------------------------------------------------------------
"$RCLONE" mkdir iris-onedrive:IRIS/repo-data/mlflow-artifacts/store
"$RCLONE" mkdir iris-onedrive:IRIS/repo-data/mlflow-backups

# -- cron: boot, watchdog, backup, log rotation (no systemd --user: vlm has no lingering) -------------------------
BEGIN="# >>> iris-mlflow-server (tools/mlflow-server/install.sh) >>>"
END="# <<< iris-mlflow-server <<<"
block="$BEGIN
IRIS_MLFLOW_HOME=$IRIS_MLFLOW_HOME
@reboot $APP/serverctl.sh boot >> $LOGS/cron.log 2>&1
*/2 * * * * $APP/serverctl.sh ensure >> $LOGS/cron.log 2>&1
30 3 * * * $APP/backup.sh >> $LOGS/cron.log 2>&1
15 * * * * /usr/sbin/logrotate -s $IRIS_MLFLOW_HOME/run/logrotate.state $APP/logrotate.conf >> $LOGS/cron.log 2>&1
$END"
current="$(crontab -l 2>/dev/null || true)"
cleaned="$(printf '%s\n' "$current" | sed "/^$(printf '%s' "$BEGIN" | sed 's/[][\/.*^$]/\\&/g')\$/,/^$(printf '%s' "$END" | sed 's/[][\/.*^$]/\\&/g')\$/d")"
printf '%s\n%s\n' "$cleaned" "$block" | sed '/./,$!d' | crontab -
echo "crontab: @reboot boot, */2 ensure, 03:30 backup, hourly logrotate"
