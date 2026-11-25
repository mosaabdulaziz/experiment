# face_recognition

Names the people whose faces `face_detect` found, against a gallery of registered faces (ArcFace embeddings in a Redis
hash). The module docstring in `__init__.py` explains the pipeline; this page is the configuration and the tuning
history the shared yaml used to carry.

```yaml
modules:
  face_detect: {landmarks: true}
  face_recognition:
    gallery_key: attendance:face:gallery
    threshold: 0.28               # default calibrated for auraface (margin: 0.05)
    capture: {}                   # CCTV face capture with its defaults
    enhancement: {}               # preprocess backend with its defaults
```

`face_detect` must be listed before it. Every setting has the value the attendance product was tuned with as its
default (`model: auraface`, `threshold: 0.28`, `margin: 0.05`), except `gallery_key` (a product names its own gallery) and `capture`/`enhancement` (off unless listed). Because `auraface` and `buffalo_l` both output 512-D embeddings from incompatible embedding spaces, any pre-existing Redis gallery must be cleared and re-enrolled via `products/attendance/reregister_all.py` when switching models.

## Settings, with the keys they replace

| setting | was | meaning |
|---|---|---|
| `model`, `det_size`, `det_thresh`, `ctx_id` | `face_recognition_model`, `_det_size`, `_det_thresh`, `_ctx_id` | the InsightFace pack and the device |
| `threshold`, `margin` | `face_recognition_threshold`, `_margin` | cosine similarity, and the open-set guard |
| `gallery_key` | `face_gallery_key` | the Redis hash of registered faces |
| `vote_ttl`, `min_votes`, `majority`, `lock_seconds`, `reconfirm_seconds`, `max_jump` | `face_recognition_*` | per-track name stability |
| `align`, `align_debug_samples`, `align_debug_dir` | `face_align_*` | straighten faces with their landmarks |
| `capture` {`min_pixels`, `min_det_confidence`, `novelty_threshold`, `dir`} | `face_capture_*`, `face_captures_dir` | save unknown faces as gallery candidates |
| `enhancement` {`backend`, `min_face_size`, `target_size`, `weight`} | `face_enhancement_*` | upscale small faces before embedding |

## Tuning history

As it stood in the shared `sidecar.yaml` before the settings moved here. The values are the module's defaults.

```yaml
# Face detections are always published alongside their parent person detection;
# they share the same Redis stream but have label "face" and include face_bbox
# relative to the parent person bbox.
face_enabled: true
face_confidence_threshold: 0.18

# Face recognition via InsightFace (ArcFace embeddings).
# When enabled, detected faces are matched against the Redis-based gallery
# (iris:face:gallery) and results published to iris:face_rec:<source_id>.
face_recognition_enabled: true
face_recognition_model: "buffalo_l"
face_recognition_threshold: 0.25    # cosine similarity threshold (0-1, lower = more permissive).
                                      # 2026-09-12 (later): back to 0.15 at the user's request, so
                                      # emp2 (best 0.158 on the elevator clip) can be recognized
                                      # again; the margin rule (0.05) and the 7 re-added elevator
                                      # photos stay.
                                      # 2026-09-12: 0.15 -> 0.20 at the user's request (precision
                                      # over recall). Measured on ~100 min of the elevator clip at
                                      # 0.15: tracks matched to more than one employee 15 -> 4,
                                      # tracks that confirm a name 31 -> 26 (measured before 7
                                      # elevator-camera photos were re-added). Known cost: emp2
                                      # peaked at 0.158 on that clip. History below predates this.
                                      # History (all 2026-08-31): 0.3 -> 0.1 (too many unknowns)
                                      # -> 0.2 (0.1 accepted wrong identities: measured worst
                                      # cross-person gallery similarity is 0.091, so 0.1 sat on
                                      # the confusion line) -> 0.15: at 0.2 a full cam-10 lap
                                      # left 40 of 122 results unknown in the 0.10-0.20 band and
                                      # the user lost people the 0.1 runs used to catch. 0.15
                                      # reclaims the top of that band; the margin rule below is
                                      # what now guards against wrong-identity accepts.
                                      # 2026-09-10: was actually still sitting at 0.2 despite this
                                      # comment -- reverted to 0.15 to match. Confirmed live against
                                      # the elevator clip: catches emp2 (max seen 0.158) and emp8's
                                      # 11:27 appearance (max 0.195), stays comfortably above the
                                      # 0.091 cross-person floor. emp8's 12:06 appearance (max 0.117)
                                      # still won't clear this -- would need to sit inside the
                                      # documented danger band to catch that one, not worth it.
face_recognition_margin: 0.05       # 2026-09-12: only actually APPLIED from this date -- the sidecar
                                      # never read this setting before, so it silently ran at 0 (off).
                                      # open-set ambiguity guard: the best identity must beat the
                                      # best OTHER identity's score by at least this much, else
                                      # the face stays "unknown" for that frame. Stops the
                                      # highest-similarity person winning outright on a blurry
                                      # crop that matches several people almost equally
                                      # (prompt.md 2026-08-31 point 4). 0 = disabled.
face_recognition_det_thresh: 0.25     # face detection threshold for the recognizer
face_recognition_det_size: [640, 640]
face_recognition_ctx_id: 0           # GPU device ID (0 = first GPU, -1 = CPU)
face_gallery_key: "attendance:face:gallery"  # forked for the products/attendance spike
                                              # (ticket 02) — default is "iris:face:gallery"

# Per-track name stability (anti-flicker).  Recognition runs at ~2 fps and the
# top gallery match can alternate between two identities that both clear the
# threshold.  Each track keeps a short
# vote history; the label shown is the majority over that window, and a competing
# identity must hold >= majority of the votes (and >= min_votes total) before it
# can displace the current one.  Labels age out after lock_seconds without a sighting.
face_recognition_vote_ttl: 4.0       # keep votes newer than this many seconds
face_recognition_min_votes: 2        # votes needed before a label is adopted. Raised from 1 on
                                     # 2026-08-31: one bad frame could lock a wrong name that then
                                     # took a 60% counter-majority to dislodge. Recognition runs at
                                     # ~sample_fps (4/s), so 3 votes = a name locks after ~0.75s of
                                     # face visibility -- raw per-frame matches still publish
                                     # immediately, this only gates the sticky overlay label.
                                     # 2026-09-10: 3 -> 2, against the elevator clip: emp7@7:50 and
                                     # emp8@4:25 each only ever got 2 raw hits (brief, angled
                                     # glimpses) and never surfaced with min_votes at 3. Real cost:
                                     # reopens some of the fragility 3 was raised to fix -- a single
                                     # false-positive vote now needs only one more coincidental
                                     # agreeing frame to lock, not two. Watch for wrong-name locks if
                                     # this causes trouble; the margin rule (open-set guard) is the
                                     # remaining backstop against a wrong identity winning outright.
face_recognition_majority: 0.6       # share of window a name must hold to win
face_recognition_lock_seconds: 5.0   # grace period: how long a confirmed name survives after the
                                     # TRACKER loses the person's body. 2026-09-12 (issue 5): this
                                     # used to mean "released this long after the last face match",
                                     # which dropped names whenever someone faced away; now the name
                                     # stays for as long as the tracker still follows the body, face
                                     # or no face, and this is only the grace after the body itself
                                     # is lost. (History below refers to the old meaning.)
                                     # 2026-09-10: 3.0 -> 5.0 -- emp7@3:33 locked correctly (13 votes)
                                     # around pts 206-208s but released before the marked window
                                     # (213-222s) even opened: a 4.5s gap, longer than the old 3.0s
                                     # lock. 5.0 covers that observed gap with margin. Real cost:
                                     # more exposure to face_recognition_max_jump's failure mode --
                                     # longer for a stale name to wrongly reapply if a track ID gets
                                     # recycled onto a different person before the jump-guard fires.
face_recognition_reconfirm_seconds: 3.0  # issue 7C (2026-09-12): if a named person's face keeps
                                         # being SEEN but hasn't matched their name for this long
                                         # (with at least 3 face sightings), the name is dropped and
                                         # must be re-earned -- catches a tracker ID swap that slipped
                                         # past the crossing (collision) guard. Time with the face
                                         # turned away doesn't count. Lower = fixes swaps faster, but
                                         # drops names more often for people standing side-on.
                                         # 6.0 -> 3.0 the same day, at the user's request.
# Straighten (align) faces before recognizing (2026-09-12). The face detector
# already finds 5 points per face (both eyes, nose tip, both mouth corners);
# the recognizer uses them to rotate/scale each face into the same upright
# 112x112 layout registration uses, before computing its fingerprint. Without
# it, live faces were compared tilted against straightened photos -- measured
# on emp2's elevator crops: 0.35-0.42 straightened vs 0.075-0.22 not. A face
# whose points are missing or look wrong falls back to the old unstraightened
# crop. false = old behavior everywhere. Restart the sidecar after changing.
face_align_enabled: true
face_align_debug_samples: 20        # save this many raw + straightened face pairs to the folder
                                    # below for a visual check, then stop (0 = off)
face_align_debug_dir: "/workspace/iris/data/aligned_debug"
face_recognition_max_jump: 0.35      # spatial guard: if a locked label's track reappears farther than this
                                     # fraction of the frame diagonal from its last sighting, the track ID was
                                     # recycled (DeepStream reuses IDs after a track is lost) — release the label
                                     # so a name can't teleport onto a different person elsewhere in the frame

# Minimum face bounding-box side (pixels, in capture resolution). Faces whose
# smaller side is below this threshold are discarded entirely — they won't be
# rendered on the JPEG stream or fed to the face recognizer. Helps filter out
# low-quality / distant faces that degrade recognition accuracy.
face_owner:
  # Which person box a face belongs to. Per product, because it applies to every
  # camera of the process and changes what the recognizer sees:
  #   legacy  -- only faces the detector left unowned, smallest containing box.
  #              What this file's face_recognition_* values were tuned against.
  #   tracked -- every face re-decided by position: same track as last frame,
  #              else head-at-top, else `fallback` (highest-top | smallest).
  # Measure on that product's recorded clip before switching (2026-09-22).
  mode: legacy

face_min_pixels: 24

# Automatic face capture from CCTV footage for building the gallery.
# When enabled, unknown faces (or matches below rec_thresh) that are large
# enough and have good detection confidence get saved as gallery candidates.
# These appear in the web UI's Face Gallery page, where an operator can
# assign them to known persons.
face_capture_enabled: true
face_capture_min_pixels: 24           # minimum face side (pixels) to be saved as a capture; matches face_min_pixels since smaller faces never reach the recognizer
face_capture_min_det_confidence: 0.25  # minimum SGIE detection confidence for capture; matches base face_confidence_threshold
face_capture_novelty_threshold: 0.25  # cosine-distance: skip if too similar to existing captures
face_captures_dir: "/workspace/iris/data/face_captures"  # shared volume path for capture thumbnails

# Face enhancement — improves low-quality CCTV crops before recognition and
# before saving capture/gallery thumbnails. Backends:
#   preprocess : lightweight OpenCV (Lanczos upscale + CLAHE + unsharp), zero deps
#   gfpgan     : full GFPGAN restoration (requires torch + gfpgan pkg + ~340MB model)
#   none       : passthrough
face_enhancement_enabled: true
face_enhancement_backend: "preprocess"
face_enhancement_min_face_size: 48    # faces smaller than this (min side, px) get upscaled
face_enhancement_target_size: 256     # upscale target for small faces
face_enhancement_weight: 0.5          # restoration weight (used by gfpgan backend)
```
