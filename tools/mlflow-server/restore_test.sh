#!/usr/bin/env bash
# Restore test of the newest backup in iris-onedrive:IRIS/repo-data/mlflow-backups/: restores both dumps into a
# throwaway Postgres container (same pinned image, no published port, no volume), compares row counts with the live
# database, then removes the container and the downloaded files. Exit 0 = the backup restores and matches.
set -euo pipefail

IRIS_MLFLOW_HOME="${IRIS_MLFLOW_HOME:-$HOME/mlflow-server}"
SRC=iris-onedrive:IRIS/repo-data/mlflow-backups
RCLONE="$HOME/.local/bin/rclone"
IMAGE="postgres:18@sha256:74935e72241653ca55e0414067e6d8763aceb8a810eb51b452253ec3dcfc4336"
NAME=iris-mlflow-restore-test
WORK="$(mktemp -d "$IRIS_MLFLOW_HOME/backups/restore-test.XXXXXX")"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

TABLES="experiments runs metrics params tags model_versions registered_models"
count() {   # count <container> <db> -> "table=n ..." for the tables that exist
    for t in $TABLES; do
        n="$(docker exec "$1" psql -U mlflow -d "$2" -tAc "select count(*) from $t" 2>/dev/null || echo -)"
        printf '%s=%s ' "$t" "$n"
    done
}

docker run -d --rm --name "$NAME" --network none -e POSTGRES_USER=mlflow \
    -e POSTGRES_PASSWORD="$(head -c 18 /dev/urandom | base64)" "$IMAGE" >/dev/null
for db in mlflow mlflow_auth; do
    latest="$("$RCLONE" lsf --files-only "$SRC" | grep -E "^$db-[0-9]{8}T[0-9]{6}Z\.dump$" | sort | tail -1)"
    [ -n "$latest" ] || { echo "no $db backup in $SRC" >&2; exit 1; }
    "$RCLONE" copyto "$SRC/$latest" "$WORK/$latest"
    echo "$db: restoring $latest ($(stat -c %s "$WORK/$latest") bytes)"
    for _ in $(seq 1 60); do docker exec "$NAME" pg_isready -U mlflow >/dev/null 2>&1 && break; sleep 1; done
    sleep 2                                    # the image's init restarts the server once
    for _ in $(seq 1 30); do docker exec "$NAME" pg_isready -U mlflow >/dev/null 2>&1 && break; sleep 1; done
    [ "$db" = mlflow ] || docker exec "$NAME" createdb -U mlflow "$db"
    docker exec -i "$NAME" pg_restore -U mlflow -d "$db" --no-owner --exit-on-error < "$WORK/$latest"
done
restored="$(count "$NAME" mlflow)"
live="$(count iris-mlflow-postgres mlflow)"
users_restored="$(docker exec "$NAME" psql -U mlflow -d mlflow_auth -tAc 'select count(*) from users')"
users_live="$(docker exec iris-mlflow-postgres psql -U mlflow -d mlflow_auth -tAc 'select count(*) from users')"
echo "restored: $restored users=$users_restored"
echo "live:     $live users=$users_live"
if [ "$restored" = "$live" ] && [ "$users_restored" = "$users_live" ]; then
    echo "restore test PASSED (the counts match the live database)"
else
    echo "restore test: counts differ (runs logged after the backup was taken make this expected)"; exit 1
fi
