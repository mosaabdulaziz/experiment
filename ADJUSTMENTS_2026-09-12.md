# Adjustments — decisions & options log

Living doc for the 7 items raised in `prompt.md` on 2026-09-12. Each section
records what's decided, what's still open, and why — nothing agreed on
tonight gets lost, and the open items have the real tradeoffs in front of
you instead of a bare menu. Jargon is explained inline as it comes up —
skip a box if you already know the term.

**Nothing in this doc has been implemented yet** unless a section says
otherwise.

---

## 1. Camera time vs. table time — both phases, fully

**Decided:** both phases get built, not just the fallback.

### Phase 1 (fallback) — **done, implemented and verified live (2026-09-12)**

Every camera's sighting time is now the server's own real clock at the
moment it actually processed the frame — no more fake simulated date
(`2023-06-08`), no more "position in a looping file" math. Concretely:

- `consumer.py`: every sighting is timestamped with `datetime.now()`
  (server's real clock) instead of the old `wall_clock_from_stream_time`
  simulation. The old `RECORDING_START` constant and the now-unused
  simulation import were removed.
- `server.py`: every place that used to default to the fake pinned
  `SPIKE_DAY` (the attendance table's day filter, the monitoring snapshot,
  the websocket's initial push, employee history) now defaults to
  `date.today()` — real today, computed fresh on each request so it's
  still correct after a real midnight passes.
- `components.tsx` (frontend): the camera-tile clock overlay is now just
  real current time, ticking every second, still hidden when a camera
  isn't streaming (same as before).
- Cleaned up now-dead code this left behind: `CAM_RECORDING_START_OVERRIDES`
  in `cameras.py` (a per-camera version of the same simulation, added then
  reverted during a false start on 2026-09-11) and its now-unused imports.

**Verified live**: restarted just the products/attendance process (no need
to touch the cameras/Docker containers — cam-21's stream kept running
uninterrupted through the whole change), watched a real sighting come in
from the live elevator-clip stream, and confirmed it landed in the table
with today's real date and a real timestamp (`2026-09-12T19:16-19:17`),
and that `/api/attendance`'s default day query is `2026-09-12`, not the old
fake `2023-06-08`.

**Follow-up fixes found while testing (all 2026-09-12, all verified live):**
- *Table lagged ~25 s behind the camera:* `sidecar.yaml` listed 21 cameras
  but only 4 were actually streaming. The pipeline processes every *listed*
  camera each cycle, so the 17 dead ones slowed everything down. Unused
  cameras are now commented out — keep only the ones in use, uncomment one
  when you want to test it. Lag after the fix: ~1–2 s.
- *Table wiped itself every few minutes:* `cam-3` (a live camera) was still
  registered as an old looping test video with a fake 284 s length, so the
  service kept thinking the video "restarted" and cleared the table. Fixed
  in `cameras.py`. `cam-1`/`cam-2` were also never watched by the
  attendance service at all — now they are.
- *Tile clock ~30 s off the table:* the tile clock ran on the viewing PC's
  own clock, the table on the server's. The tile now corrects itself to the
  server's clock (`/api/time`).
- *Date picker stuck on 2023-06-08:* the dashboard's own default day was
  still the old fake date; now it's today.

### Phase 2 (camera's own clock over RTCP) — **done, wired end-to-end (2026-09-12)**

**What it does, plainly:** a camera streaming video over RTSP (the standard
camera-streaming protocol) can also send small side-channel messages called
**RTCP Sender Reports** that say "this frame was captured at *my* clock's
time" (in **NTP** format — the time format the internet uses for clock
syncing). The pipeline now reads that. When a camera sends it, that
camera's table times and its tile clock use the **camera's** time. When a
camera doesn't send it (or stops for more than 10 seconds), that camera
automatically uses the **server's current time** — Phase 1 — exactly the
fallback we agreed. It's decided per camera, so a camera with a clock and
one without work side by side.

**How it's wired** (no new libraries — DeepStream, the video library the
pipeline is built on, already carries this):
- `sidecar.py`: the stream muxer's `attach-sys-ts` setting is turned off, so
  it keeps the camera's time in each frame's `ntp_timestamp` field instead
  of overwriting it with the machine's clock (0 = the camera sent none), and
  each RTSP source is told to use the camera's Sender Reports
  (`configure_source_for_ntp_sync`). The camera time is published with each
  frame's data as `ntp_ts`.
- `acl.py` / `consumer.py`: the attendance service turns that into a
  per-camera "camera clock minus server clock" offset and stamps sightings
  with server time + offset; no fresh reading → plain server time.
- `server.py` / frontend: `/api/recordings` reports each camera's offset and
  `clock_source` (`camera` or `server`); the tile clock applies it. Hovering
  over the tile clock says which one it's showing ("ساعة الكاميرا (RTCP)" or
  "ساعة الخادم").
- On/off switch: `rtcp_camera_clock: true` in `sidecar.yaml` (restart the
  sidecar after changing it).

**Verified:** the sidecar runs normally with it on (detection and face
recognition unaffected). A simulated camera sending a clock 5 s ahead was
picked up within half a second (API switched that camera to camera time,
+4.96 s), and 10 s after it stopped, it fell back to server time on its own.

**Today, every camera shows server time — and why:** none of the current
cameras' clocks reach the pipeline. `cam-1/2/3` connect through `go2rtc`
(the relay configured in `go2rtc.yaml`), and go2rtc passes the video along
but drops these RTCP messages (checked in its source code, version 1.9.14);
in tests connecting straight to those cameras, they didn't send any
either. A camera that does send them, and whose `sidecar.yaml` source
points straight at it (`rtsp://user:pass@<camera-ip>/...` instead of
`rtsp://go2rtc:8554/...`), gets its own clock with no further changes.

---

## 2. Detection classes — **done (2026-09-12)**

Keep `person`, `fire`, `smoke`. Comment out the rest (`car`, `truck`, `bus`,
`motorcycle`, `bicycle`, `traffic light`, `stop sign`, `dog`, `cell phone`,
`laptop`, `tvmonitor`) — leftovers from an unrelated productivity-monitoring
feature, not used for identification. `fire`/`smoke` stay because removing
them would silently disable fire alerting too (same shared filter, see
earlier finding).

---

## 3. Face recognition accuracy — per-option decisions

### Changes applied 2026-09-12 (evening) — favouring precision over recall

- **Margin rule switched on (0.05).** `face_recognition_margin` had been in
  `sidecar.yaml` since 2026-08-31, but the sidecar never read it — it
  silently ran at 0 (off). Now, if the best-matching employee doesn't beat
  the runner-up by at least 0.05, the face stays "unknown" for that frame.
- **Match threshold 0.15 → 0.20, then back to 0.15** the same evening at
  your request, so emp2 can be recognized (his best score on the elevator
  clip is 0.158, and at 0.20 he was never matched). The margin rule and the
  7 re-added photos stay. For reference, 0.20 measured beforehand on ~100
  min of the clip: tracks matched to more than one employee 15 → 4, tracks
  that confirm a name 31 → 26.
- **7 clear elevator-camera photos re-registered — permanently this time.**
  Saved in `data/face_photos/multi/<person>/`, the folder startup
  re-registration reads (same place the UI's "add photo" saves to), so
  `just up` no longer wipes them. Registry 25 → 32, then **35** when you
  added emp2's 3 elevator crops (`…17-33-23`, `…17-33-43`, `…17-34-16`,
  same folder layout). Those three are blurrier than the clear set — if
  other people start being called emp2, they're the first to remove.
  Now: emp1 4, emp2 5, emp3 3, emp4 4, emp5 3, emp6 3, emp7 5, emp8 5, emp9 3.
- **`face_recognition_det_thresh` / `face_recognition_det_size` left alone —
  they have no effect:** the recognizer works directly on the face crops
  the sidecar already found and never runs its own face-finder.
- **Part A dropped** (lowering the face detector's own cut-off,
  `pre-cluster-threshold` 0.25 → 0.15, to catch more but blurrier faces):
  it trades precision for recall — the opposite of the goal.
- **Part B kept as a later experiment:** raise the processing resolution
  1080×720 → 1920×1080 so face crops are sharper (today faces are ~3.5×
  smaller than the 4K camera captured them); `products/attendance/zones.py`
  (`FRAME_W`/`FRAME_H`) must change with it.
- Not changed: "names held only while the detector sees the body" (Fix 2) —
  not requested.
- **Measured after one lap of cam-21** (small sample: 12 person-tracks,
  80 matches; before = ~100 min at 0.15, margin off, 25 photos):
  tracks matched to more than one employee 36% → 16%; tracks that confirm a
  name unchanged at 75%; all matches now ≥ 0.20 and 72% ≥ 0.25 (before, 36%
  sat at 0.15–0.20); "name taken" blocks 16 → 5 per 100 matches; the margin
  rule blocked 3 close calls.

### Option A — More enrollment photos per employee — **decided, in progress**

**Gallery audit, done just now** (this is "check whether they have it or
not"):

| employee | enrolled photos |
|---|---|
| emp1 | 3 |
| **emp2** | **2** |
| emp3 | 3 |
| emp4 | 3 |
| **emp5** | **2** |
| emp6 | 3 |
| emp7 | 3 |
| emp8 | 3 |
| emp9 | 3 |

`emp2` and `emp5` are the only two under-enrolled, and they're *also* two of
tonight's weaker-recognition cases from the elevator test — a real,
data-backed reason to expect this to help specifically them.

**Next step (queued, not done yet):** you asked me to sweep the *whole*
video (not just the known appearance windows like tonight's first pass) and
pull out many more candidate face crops, so you have a real range to choose
from per person when you enroll via the UI. I'll do that sweep next and
hand you the resulting folder, same as tonight's `candidate_review/` — you
pick and enroll, I don't touch the gallery myself.

### Option B — Face enhancement (`gfpgan`) — **on hold, as you said**

Waiting until we've seen how far A (and C, if you go that way) get accuracy
on their own before spending the extra GPU cost here.

### Option C — Recognition model swap — **decided: try it, compare, and here are lighter alternatives too**

You asked for alternatives that are smaller/cheaper on the GPU, not just
`antelopev2`. Here's the actual picture:

The recognition model in use, `buffalo_l`, is one member of a *family* of
models from the same source (InsightFace) that all do the same job at
different size/accuracy tradeoffs — think of it like a t-shirt in S/M/L,
same design, different size:

| model | size vs. `buffalo_l` | what it trades off |
|---|---|---|
| `buffalo_l` (current) | — | most accurate of the family, most GPU cost |
| `buffalo_m` | smaller | a bit less accurate, meaningfully cheaper |
| `buffalo_s` | smallest in the family | noticeably less accurate, cheapest — likely too far for an identification system |
| `antelopev2` | **larger**, not smaller | a *different* model line (not a size variant of buffalo), generally the most accurate option available, but the heaviest — the opposite direction from "smaller" |

So: `buffalo_m` is the real answer to "smaller and still reasonable" —
same family as what's running now, just a lighter tier. `antelopev2` is the
"if accuracy is the only thing that matters" option, at a GPU cost, not a
savings.

**Important, applies to every option in this row equally:** switching *any*
recognition model — `buffalo_m`, `antelopev2`, anything — means every
employee needs to be re-enrolled from scratch. Each model produces its own
kind of embedding "fingerprint," and one model's fingerprints aren't
comparable to another's at all — there's no way to convert existing
enrollments over. `register_face.py` currently has `buffalo_l` hardcoded, so
this is real migration work regardless of which model you land on, not a
config flip.

**What I'd suggest, so you can actually compare them ("A/B testing"):**
"A/B testing" here just means running two versions side by side against the
*same* footage and comparing which one gets more people right — like
tonight's before/after checks on the elevator clip, just repeated once per
model. I'd re-enroll everyone under one candidate model, re-run the exact
same elevator-clip test we already have a baseline for, and put the numbers
next to each other — real evidence instead of guessing. Let me know which
model(s) you want in that comparison (I'd suggest starting with `buffalo_m`
given you explicitly asked for something lighter, with `antelopev2` as a
second comparison if `buffalo_m` alone doesn't close the gap enough).

### Option D — Face-related settings reference (all of them, plain-language)

Everything face-related that lives in `perception/pipeline/sidecar.yaml`, grouped by
what it actually controls. "Detection" = *is there a face here at all*.
"Recognition" = *whose face is it*. Two separate steps, two separate sets of
knobs.

**Detection — finding a face at all**
| setting | current | what it means |
|---|---|---|
| `face_enabled` | `true` | Master switch — face detection runs at all, on top of person detection. |
| `face_confidence_threshold` | `0.18` | How sure the detector needs to be that something is a face before it counts as one. Lower = catches more faces, including some blurry non-faces. |
| `face_min_pixels` | `24` | Faces smaller than this (in pixels) are thrown out immediately — too small to reliably do anything with, not worth the cost of trying. |

**Recognition — matching a detected face to a known person**
| setting | current | what it means |
|---|---|---|
| `face_recognition_enabled` | `true` | Master switch — attempt to match detected faces against enrolled employees at all. |
| `face_recognition_model` | `buffalo_l` | Which model does the matching (Option C territory). |
| `face_recognition_threshold` | `0.15` | How similar a face needs to be to an enrolled photo to count as a match, on a 0–1 scale. Lower = matches more things, including some wrong ones. This is the main "precision vs. recall" dial we already tuned once tonight. |
| `face_recognition_margin` | `0.05` | The winning candidate has to beat the *second-best* candidate by at least this much, or the face stays "unknown." Protects against two similar-looking people both scoring high and the system just guessing which one. |
| `face_recognition_det_thresh` | `0.25` | The recognition model does its *own* internal face-finding pass before matching (separate from the main detector above) — this is that pass's own confidence gate. |
| `face_recognition_det_size` | `[640, 640]` | The image size the recognizer resizes a frame to internally before its own face-finding pass. |
| `face_recognition_ctx_id` | `0` | Which GPU runs the recognizer (0 = the first one). |
| `face_gallery_key` | `attendance:face:gallery` | Which Redis storage key holds the enrolled-employee photo/embedding list. |

**Stability — keeping a displayed name steady instead of flickering**
| setting | current | what it means |
|---|---|---|
| `face_recognition_vote_ttl` | `4.0` | A single match "counts" toward the running tally for this many seconds before it expires and stops counting. |
| `face_recognition_min_votes` | `2` | How many counted matches are needed before a name is trusted enough to actually display/log (this is the number we just decided to also require for the attendance table, issue 4). |
| `face_recognition_majority` | `0.6` | Of the votes currently counted, this fraction must all agree on the *same* name for it to win — stops the display flickering between two names that are both scoring close. |
| `face_recognition_lock_seconds` | `5.0` | How long a confirmed name keeps showing after the last real sighting, before it's dropped. This is the one changing for issue 5 (see below) — moving from a fixed time window to "as long as the same tracked body sticks around." |
| `face_recognition_max_jump` | `0.35` | Explained in full in the issue 5 section below. |

**Auto-capturing candidate photos** (the mechanism behind tonight's
`candidate_review/` folder)
| setting | current | what it means |
|---|---|---|
| `face_capture_enabled` | `true` | Automatically save a photo of an unrecognized (or weakly-recognized) face, so a human can later decide whether to enroll it. |
| `face_capture_min_pixels` | `24` | Faces smaller than this aren't worth saving as a candidate either. |
| `face_capture_min_det_confidence` | `0.25` | Minimum detector confidence before bothering to save a candidate photo. |
| `face_capture_novelty_threshold` | `0.25` | Skip saving a new candidate if it looks too similar to one already saved — avoids saving the same person's face 50 times in a row. |

**Image enhancement before recognition** (Option B territory)
| setting | current | what it means |
|---|---|---|
| `face_enhancement_enabled` | `true` | Whether to clean up a face crop before trying to recognize it. |
| `face_enhancement_backend` | `preprocess` | Which cleanup method: `preprocess` = fast, lightweight sharpening; `gfpgan` = much stronger AI-based restoration, real GPU cost, not currently active (Option B). |
| `face_enhancement_min_face_size` | `48` | Faces smaller than this get upscaled before recognition is attempted. |
| `face_enhancement_target_size` | `256` | The size a small face gets upscaled *to*. |
| `face_enhancement_weight` | `0.5` | Only used by `gfpgan` — how strongly its restoration is applied. |

---

## 4. Table doesn't match the live label — **done (2026-09-12), checked by you on cam-21**

**Built:** every match the sidecar publishes now carries a `confirmed` flag —
true only once the name has passed the same two-agreeing-matches vote the
live label uses. The attendance service skips unconfirmed matches, so the
table and the live label now come from one and the same decision.

**Decided:** unify them. The attendance table will require the same 2-match
confirmation (`min_votes`) the live label already does, instead of logging a
sighting off one single raw match. You're choosing precision over recall —
a name shouldn't land in the table on one marginal frame, even if that
occasionally costs the table a real-but-brief appearance it would have
caught before. This directly closes the gap that let a single stray
misidentified frame (the `emp7`-during-`emp8` glitch from tonight's testing)
count as a real sighting.

---

## 5. Persistence through occlusion / facing away — **done (2026-09-12), checked by you on cam-21**

**Built:** every frame, the sidecar tells the recognizer which people the
tracker is still following. A confirmed name now stays on a body for as
long as that body is tracked — face turned away or not — and is dropped
about 5 s (`face_recognition_lock_seconds`, now a grace period) after the
tracker loses the body. A tracker ID that comes back after that counts as
a new person and must earn a name again. The `max_jump` check now only
applies when the tracker actually lost the body in between, so someone
walking across the room with their face away keeps their name.

**Decided:** stop tying the displayed name to a fixed timer
(`face_recognition_lock_seconds`). Instead, keep showing the name for as
long as the system still believes it's looking at the *same physical body*
— which, underneath, already survives a lot (the tracker's own memory of
"this is probably still the same person even though I can't see their face
right now" already holds for 600 frames, about 15× more generous than the
out-of-the-box default). Only drop the name when the system actually loses
track of the body itself, not just when the face happens to be turned away
for a few seconds.

**What `max_jump` is, in full:** every tracked person has a box drawn
around their body, and that box has a position in the frame. When a name is
"locked" onto a tracked body, the system remembers roughly where that body
last was. If, later, the *same track number* suddenly shows up somewhere
much farther away in the frame than expected, that's a red flag — trackers
occasionally reuse an old ID number for a completely different, newly-
appeared person instead of assigning a fresh one, and if we don't check for
that, the *old* name would get attached to this *new*, different person.
`max_jump` is exactly that check: it measures how far the "same" track
number jumped between sightings, as a fraction of the screen's diagonal
(corner-to-corner) distance — currently `0.35`, meaning "35% of the way
across the screen." If the jump is bigger than that, the system assumes the
ID got reused for someone new and drops the name rather than risk
mislabeling them. It's the safety net for issue 5's change: even though
we're no longer dropping the name on a timer, this still drops it instantly
if the tracked body's identity itself looks suspicious.

This is a logic change in `perception/pipeline/face_recognizer.py`, not a config value
— flagging that since everything else tuned tonight was a value change.

---

## 6. Live-camera end-to-end test — queued after issue 1

Once issue 1's Phase 1 (and whatever Phase 2 support the real cameras
actually have) lands, run the same kind of check we did for the test clips
tonight, against `cam-1/2/3` for real: does the camera tile's clock agree
with the table, and does identification/recognition behave the same as it
did on the recordings. This is also literally where we find out which of
Phase 2's camera-clock support actually works on real hardware.

---

## 7. Identity "teleporting" between people in a crowded scene — **built (2026-09-12), awaiting your check on cam-21**

**Built** (all in `perception/pipeline/face_recognizer.py`; no products/attendance change —
the table already has its own crossing guard and, since issue 4, only takes
confirmed names, so it follows the live label):
- **A — crossing guard:** every frame, each tracked person's box position
  is remembered for ~3.5 s to tell whether they're moving (median of the
  older half vs. the newer half, so box jitter doesn't count). When two
  **moving** people's boxes overlap by ≥ 35% of the smaller box, any name on
  either is paused and must be re-earned with 2 fresh agreeing face matches
  from after the crossing (faces processed within 0.5 s of it get no vote).
  Same thresholds as the table's guard. A walker passing someone standing
  still pauses nobody. Log line: `collision with moving track … paused`.
- **C — re-check while locked:** if a named person's face keeps being seen
  but hasn't matched their name for `face_recognition_reconfirm_seconds`
  (3 s — lowered from 6 s at your request; new setting in `sidecar.yaml`)
  with at least 3 face sightings, the name is dropped until re-earned. Time
  with the face turned away doesn't count (a face gap over 2 s restarts the
  window). Log line: `not re-confirmed by its visible face … dropped`.
- **Trade-off:** someone standing side-on (face visible but not matching)
  loses their name after 3 s, until they face the camera again — consistent
  with precision over recall; tune with the same setting.

**Decided:** do both.

**A — Port the table's existing "collision guard" to the live label too.**
The attendance table already protects against this: when two people who are
both *moving* have overlapping boxes, the system suspends both their name
bindings until a fresh face match re-confirms which is which — built
specifically because a tracker can swap which ID number belongs to which
person when two people cross paths. Right now that protection only exists
on the table side (`products/attendance`); the live label (in the DeepStream
sidecar itself) has no equivalent, and its only defense (`max_jump`, above)
doesn't catch a swap between two people who are *near* each other, only one
that jumps *far* — so a swap between two people standing or walking close
together slips through today. Porting the same "two movers overlap → pause
until re-confirmed" logic to the live-label side closes that gap using an
approach already proven to work on the table.

**C — Re-check the face occasionally even while a name is "locked."**
Right now, once a name locks onto a tracked body, nothing re-verifies it
against the actual face again until the lock naturally expires — so if a
swap happens silently (not caught by A, or A doesn't apply because the two
people never technically overlapped), the wrong name can keep displaying
for a long time with nothing double-checking it. Requiring an occasional
fresh face-match, even during an otherwise-locked streak, means a silent
swap gets caught and corrected within seconds instead of riding out the
full lock duration.

Together: A catches a swap *at the moment* two people cross paths; C
catches one that already happened and got past A. Neither alone covers
both failure shapes.

---

## Status summary

| # | Status |
|---|---|
| 1 | Done — Phase 1 (server time) + Phase 2 (camera clock over RTCP, falls back per camera); today every camera shows server time because go2rtc drops RTCP |
| 2 | Done |
| 3 (settings) | Applied — margin rule on (0.05), threshold back to 0.15 (was briefly 0.20; lowered so emp2 shows), 7 elevator photos re-registered permanently; Part A dropped; Part B (sharper faces) = later experiment |
| 3A | Dropped (your call, 2026-09-12) |
| 3B | Dropped (your call, 2026-09-12) |
| 3C | Dropped (your call, 2026-09-12) |
| 3D | Reference delivered above |
| 4 | Done — checked by you on cam-21 |
| 5 | Done — checked by you on cam-21 |
| 6 | Dropped (your call, 2026-09-12) |
| 7 | Built — awaiting your check on cam-21 |
