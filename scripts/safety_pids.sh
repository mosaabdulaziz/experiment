# The safety stack's host processes (controller, fall service, demo streamers) and their pid files. Sourced by
# scripts/safety_env.sh. Each is started as `setsid bash -c 'echo $$ > <pid file>; ...'`: it leads its own process
# group and writes the file as it starts. A pid in such a file is trusted only while it is still that process -- a pid
# reused by another process after it ended (or after a reboot) must never be signalled, least of all as a group on a
# shared server.

# The pid a pid file names, printed, if it is still the process that wrote the file: a number above 1, alive, leading
# its own process group, and started no later than the file was written (2 s for the clocks' rounding). Fails silently.
# Read from /proc, not ps: a slim image (the CI's python:3.11-slim) has no procps.
safety_pid() {
    local pidf=$1 pid line fields ticks btime key value rest
    [[ -f $pidf ]] || return 1
    pid=$(tr -d '[:space:]' < "$pidf")
    [[ $pid =~ ^[0-9]+$ ]] && (( pid > 1 )) || return 1
    line=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
    read -ra fields <<< "${line##*) }"            # its fields from the 3rd (state) on: the name before may hold spaces
    [[ ${fields[2]-} == "$pid" ]] || return 1      # the 5th: its process group, which it must lead
    ticks=$(getconf CLK_TCK) || return 1
    while read -r key value rest; do [[ $key == btime ]] && btime=$value; done < /proc/stat
    [[ ${btime-} =~ ^[0-9]+$ && ${fields[19]-} =~ ^[0-9]+$ ]] || return 1
    # the 22nd: when it started, in clock ticks after boot (btime, in seconds since the epoch)
    (( btime + fields[19] / ticks <= $(stat -c %Y "$pidf") + 2 )) || return 1
    echo "$pid"
}

# Stop the process group a pid file names, when safety_pid vouches for it, and remove the file either way. Returns 1
# (saying so) when there was nothing of ours to stop.
safety_stop() {
    local pidf=$1 pid
    [[ -f $pidf ]] || return 1
    if pid=$(safety_pid "$pidf"); then
        kill -TERM -- "-$pid" 2>/dev/null
        rm -f "$pidf"
        return 0
    fi
    echo "$(basename "$pidf"): pid '$(tr -d '[:space:]' < "$pidf")' is no longer the process that wrote it" \
        "(it ended, or the machine restarted); nothing stopped" >&2
    rm -f "$pidf"
    return 1
}
