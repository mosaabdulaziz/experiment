#!/usr/bin/env bash
# Daily backup of the central MLflow server's databases (cron, see install.sh): pg_dump (custom format) of `mlflow`
# (tracking + registry) and `mlflow_auth` (users), uploaded to iris-onedrive:IRIS/repo-data/mlflow-backups/, keeping
# the newest 14 of each. Artifacts need no backup here: they already live on SharePoint
# (iris-onedrive:IRIS/repo-data/mlflow-artifacts).
#
# The backup folder is never created here: it must already exist on SharePoint (install.sh makes it). If it is missing,
# or the listing of its parent fails, the backup refuses and uploads nothing: a missing folder means the remote points
# somewhere unexpected (a recreated rclone remote on the wrong library, a moved or renamed folder), and quietly creating
# it there would scatter backups where nobody looks. The check runs before every upload, because `rclone copyto`
# creates missing parent folders on its own.
set -euo pipefail

IRIS_MLFLOW_HOME="${IRIS_MLFLOW_HOME:-$HOME/mlflow-server}"
DEST=iris-onedrive:IRIS/repo-data/mlflow-backups
KEEP=14
RCLONE="$HOME/.local/bin/rclone"
# shellcheck disable=SC1091
[ -f "$IRIS_MLFLOW_HOME/config.env" ] && . "$IRIS_MLFLOW_HOME/config.env"
LOGS="$IRIS_MLFLOW_HOME/logs"
WORK="$IRIS_MLFLOW_HOME/backups"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
mkdir -p "$LOGS" "$WORK"
log() { printf '%s [backup] %s\n' "$(date -Is)" "$*" | tee -a "$LOGS/backup.log" >&2; }

require_dest() {   # DEST must already be a folder in its parent; never create it
    local listing
    listing="$("$RCLONE" lsf --dirs-only --max-depth 1 "${DEST%/*}")" \
        || { log "REFUSED: cannot list ${DEST%/*} (remote down or misconfigured); nothing uploaded"; exit 1; }
    grep -qxF "${DEST##*/}/" <<<"$listing" \
        || { log "REFUSED: $DEST does not exist; it is never created here (install.sh makes it). Check that the rclone remote points at the right library. Nothing uploaded"; exit 1; }
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
require_dest
for db in mlflow mlflow_auth; do
    file="$WORK/$db-$stamp.dump"
    docker exec iris-mlflow-postgres pg_dump -U mlflow -d "$db" -Fc > "$file.part"
    mv "$file.part" "$file"
    require_dest
    "$RCLONE" copyto "$file" "$DEST/$(basename "$file")"
    remote_size="$("$RCLONE" size --json "$DEST/$(basename "$file")" | sed -n 's/.*"bytes":\([0-9]*\).*/\1/p')"
    [ "$remote_size" = "$(stat -c %s "$file")" ] || { log "FAILED: $db upload size mismatch"; exit 1; }
    log "$db: $(basename "$file") ($(stat -c %s "$file") bytes) uploaded to $DEST"
    rm -f "$file"
    # keep the newest $KEEP of this database (names sort by their UTC timestamp)
    "$RCLONE" lsf --files-only "$DEST" | grep -E "^$db-[0-9]{8}T[0-9]{6}Z\.dump$" | sort -r | tail -n +$((KEEP + 1)) \
        | while read -r old; do "$RCLONE" deletefile "$DEST/$old"; log "$db: pruned $old"; done
done
