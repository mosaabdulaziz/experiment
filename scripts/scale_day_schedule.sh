#!/usr/bin/env bash
# Schedule a full-day run on the server so it does not depend on this laptop staying awake:
#   scripts/scale_day_schedule.sh 2026-10-06T00:00:00        footage 00:00 of the recorded day begins at that server time
# On the server, in the background (nohup setsid, log /tmp/day-schedule.log):
#   ARM_MIN minutes before (default 12): preflight of the recordings, then scale13_deploy.sh day-arm (ON_SERVER=1)
#   then day-arm installs a cron watchdog (scripts/scale_day_ensure.sh) that restarts the stack/players after a reboot
#   and exports into /data/IRIS/TESTING_PLAYGROUND/day-export-<day> 10 minutes after the footage ends
# Needs the latest code synced there first (scripts/scale13_deploy.sh sync). Cancel: ssh iris-work "kill $(cat /tmp/day-schedule.pid)".
set -euo pipefail
BEGIN_AT="${1:?usage: scale_day_schedule.sh <BEGIN_AT e.g. 2026-10-06T00:00:00>}"
ARM_MIN="${ARM_MIN:-12}"
DAY="${DAY:-2026-09-13}"
DEST=/data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys-scale
ssh -o RequestTTY=no -o RemoteCommand=none "${SSH_HOST:-iris-work}" "bash -s" <<REMOTE
set -e
cd $DEST
[ -f /tmp/day-schedule.pid ] && kill -0 \$(cat /tmp/day-schedule.pid) 2>/dev/null && { echo "a schedule is already running (pid \$(cat /tmp/day-schedule.pid)); cancel it first"; exit 1; }
chmod +x scripts/*.sh scripts/*.py
python3 scripts/scale_day_preflight.py --day $DAY
cat > /tmp/day-schedule-run.sh <<'JOB'
#!/usr/bin/env bash
cd $DEST
wait_until() { t=\$(date -d "\$1" +%s); n=\$(date +%s); [ \$t -gt \$n ] && sleep \$((t-n)); return 0; }
wait_until "\$(date -d '$BEGIN_AT $ARM_MIN minutes ago' +%FT%T)"
echo "[\$(date +%T)] preflight"; python3 scripts/scale_day_preflight.py --day $DAY || echo "PREFLIGHT FAILED, arming anyway"
echo "[\$(date +%T)] day-arm"
ON_SERVER=1 START=${DAY}T00:00:00 BEGIN_AT=$BEGIN_AT scripts/scale13_deploy.sh day-arm
echo "[\$(date +%T)] armed; from here the cron watchdog (scripts/scale_day_ensure.sh) keeps it running and exports it"
JOB
chmod +x /tmp/day-schedule-run.sh
nohup setsid /tmp/day-schedule-run.sh > /tmp/day-schedule.log 2>&1 < /dev/null &
echo \$! > /tmp/day-schedule.pid
echo "scheduled (pid \$(cat /tmp/day-schedule.pid)): arm at \$(date -d '$BEGIN_AT $ARM_MIN minutes ago' +%T), footage begins $BEGIN_AT, export at \$(date -d '$BEGIN_AT 24 hours 10 minutes' '+%F %T')"
REMOTE
