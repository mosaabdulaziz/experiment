#!/usr/bin/env bash
# Print the exports for one barber site (2026-10-14, branch 2): eval "$(products/barber/site.sh <site>)".
# <site> empty = the running site (.run/barber-site, written by barber-up), else bekky;
# an explicit 2nd arg replaces that lookup (barber-up passes bekky: no site given = bekky, not the last run).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
site="${1:-}"
[ -z "$site" ] && site="${2:-$(cat "$root/.run/barber-site" 2>/dev/null || echo bekky)}"
case "$site" in
  bekky)
    cfg="$root/products/barber/barber.yaml"; scfg="products/barber/sidecar.yaml"; clips="" ;;
  palm-hills)
    cfg="$root/products/barber/barber.palm-hills.yaml"; scfg="products/barber/sidecar.palm-hills.yaml"
    clips="ph-1,ph-2,ph-3,ph-4,ph-5,ph-6,ph-7" ;;
  palm-hills-live)
    # the shop's recorder (2026-10-15): go2rtc pulls these channels (products/barber/feeds.py); the address with
    # its credentials is .env's PALM_HILLS_RTSP=rtsp://user:pass@host:554/Streaming/Channels. No ph-4 (401).
    cfg="$root/products/barber/barber.palm-hills-live.yaml"; scfg="products/barber/sidecar.palm-hills-live.yaml"
    feeds="raw-ph-1=101 raw-ph-2=1101 raw-ph-3=901 raw-ph-5=501 raw-ph-6=601 raw-ph-7=801"; feed_base=PALM_HILLS_RTSP ;;
  palm-hills-16oct)
    # 2026-10-18 (user): the shop's 16 Oct, synced (palm_hills2/synced), played as ph-* (no ph-4); since 2026-10-21 the
    # whole day, 09:30-23:59:59 (the evening's export appended: dev/staff/palm2/extend_day.py). Its own
    # DB and crops, reports in the footage's time, its report written at the lap's end (barber.palm-hills-16oct.yaml)
    cfg="$root/products/barber/barber.palm-hills-16oct.yaml"; scfg="products/barber/sidecar.palm-hills-16oct.yaml"
    clips="ph-1,ph-2,ph-3,ph-5,ph-6,ph-7"
    d=/data/IRIS/data/BekkyBarber/palm_hills2/synced; t=2026-10-16_09-30-00_to_23-59-59_SYNCED.mp4
    clip_files="ph-1=$d/Cam1_Shampoo-Wash-Area_$t,ph-2=$d/Cam2_Barber-Stations-Wide_$t,ph-3=$d/Cam3_Reception-Desk-Front_$t"
    clip_files="$clip_files,ph-5=$d/Cam5_Reception-Lounge_$t,ph-6=$d/Cam6_Pedicure-Area_$t,ph-7=$d/Cam7_Barber-Chairs-Glass-Wall_$t" ;;
  palm-hills-6nov)
    # 2026-11-10 (user): the shop's 6 Nov, 09:00-24:00, synced (palm_hills/6nov-synced), played as ph-* (no ph-4); the
    # files are numbered differently from 16 Oct (barber.palm-hills-6nov.yaml)
    cfg="$root/products/barber/barber.palm-hills-6nov.yaml"; scfg="products/barber/sidecar.palm-hills-6nov.yaml"
    clips="ph-1,ph-2,ph-3,ph-5,ph-6,ph-7"
    d=/data/IRIS/data/BekkyBarber/palm_hills/6nov-synced
    clip_files="ph-1=$d/6nov_cam1.mp4,ph-2=$d/6nov_cam6.mp4,ph-3=$d/6nov_cam5.mp4"
    clip_files="$clip_files,ph-5=$d/6nov_cam2.mp4,ph-6=$d/6nov_cam3.mp4,ph-7=$d/6nov_cam4.mp4" ;;
  *) echo "unknown barber site '$site' (bekky | palm-hills | palm-hills-live | palm-hills-16oct | palm-hills-6nov)" >&2; exit 2 ;;
esac
printf 'export BARBER_SITE=%q BARBER_CONFIG=%q BARBER_SIDECAR_CONFIG=%q BARBER_SITE_CLIPS=%q BARBER_SITE_FEEDS=%q BARBER_SITE_FEED_BASE=%q BARBER_SITE_CLIP_FILES=%q\n' \
  "$site" "$cfg" "$scfg" "${clips:-}" "${feeds:-}" "${feed_base:-}" "${clip_files:-}"
