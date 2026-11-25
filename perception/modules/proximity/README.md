# proximity

A person too close to a **moving** vehicle inside a zone raises one safety alert per episode. Model-free: it reads the
primary detector's persons and vehicles with their instance masks (a YOLO-seg detector; `products/proximity/` runs
YOLO26l-seg at 1280x736) and measures, per camera, in **person heights (ph)**. The design and every number behind it:
`RESEARCH_pedestrian-vehicle-proximity_2026-10-26.md` (branch `analysis/ped-vehicle-proximity`).

On every frame the detector ran on (nvinfer's own flag, not the engine's `analyze` counter, which can disagree with
nvinfer's `interval`; frames without a timestamp are skipped), for each zone with proximity on, a person-vehicle pair
is close when:

1. the person's foot (recovered when hidden, below) and the vehicle's ground reference point (the median of its mask's
   lowest outline) are both inside the zone;
2. the person is not riding the vehicle. Rider status is kept per person and vehicle and set only by positive
   evidence: on it by its geometry (box mostly inside, foot on its mask or on its bed) and moving along with it; on
   it by its geometry while it was parked (a driver who then drives off); or feet hidden, right beside its box and
   travelling with it (a cab the detector left out of the vehicle's box, an open forklift frame). It is kept while
   the person keeps up with the vehicle in any direction (in a turn the cab and the vehicle's reference point move
   differently), for up to 5 s since last confirmed, and lost when they stand still while it moves. A pedestrian
   walking toward or across a vehicle, or walking with it with their feet in view, never gets it. While co-motion
   cannot tell yet, someone inside its box on it by the geometry counts as its occupant, frame by frame: a person
   track under 2 s old, or someone who got on under 1.5 s ago (no break over 0.5 s), has stayed put on its box since
   (within half a person height) and keeps up with it -- the 2 s window still holds the walk up to a vehicle that
   set off as they climbed on, or a dropout that left their history shorter than the vehicle's. Someone standing
   still while it passes, or walking past behind it, gets no such benefit of the doubt;
3. the vehicle is moving: its reference point moved more than `stationary_motion` over `stationary_window_s`
   (median of the window's last third against its first third, so one jumpy frame does not count), and its box moved
   more than `box_motion` of that: the second largest of its four edges' movements (a merge with a neighbour
   stretches one edge, a vehicle driving sideways moves two), or the largest edge off the picture's border when the
   box touches it (a vehicle entering: one edge moves). The state is steady: a moving vehicle reads parked only after
   reading so for `parked_hold_s` in a row (reading moving again is immediate), and one with too little history to
   tell -- back from a detector dropout, or re-identified -- keeps its last state, wherever its point came back (a
   parked truck re-detected partly hidden shifts it too). Parked vehicles never alert and are tagged "parked" in the
   preview;
4. the gap from the foot to the nearest part of the lowest quarter of the vehicle's mask (`vehicle_band`: its wheels,
   side bumpers and forks near the ground, not the body above; a vehicle with no mask: its lower outline) is at most
   `max_gap`.

A pair alerts once it has been close for `persist_s` (gaps up to `grace_s` tolerated) and is confirmed, and only when
it opens a new episode -- pairs within 1.5 ph of the episode's last person or 2 ph of its last vehicle join it, so track
id switches do not re-alert. An episode that turns more severe (an unconfirmed one gets a confirmed close pair) alerts
once more, flagged `escalation=1`. An episode ends `grace_s + cooldown_s` after its last close pair (logged).

## Settings (`modules.proximity` in the sidecar yaml)

| setting | default | meaning |
|---|---|---|
| `config_path` | `data/zones_config.txt` | the zones file the app writes |
| `person_classes` | `[person]` | |
| `vehicle_classes` | `[car, truck, bus, motorcycle]` | COCO; a forklift comes through as truck or car |
| `max_gap` | 0.75 | ph, foot to the lowest `vehicle_band` of the vehicle's mask |
| `stationary_window_s` | 2.0 | s |
| `stationary_motion` | 0.75 | ph; displaced less than this over the window = parked (about 0.56 ph/s) |
| `persist_s` | 0.25 | s close before alerting (3 close analysed frames in a row at 6.7/s) |
| `grace_s` | 0.5 | s a close pair may drop out (counted beyond the frame interval) |
| `cooldown_s` | 10 | s an episode stays open after `grace_s` |
| `hidden_feet` | `unconfirmed` | hidden feet with no depth check: `unconfirmed` (alert at info, flagged) or `suppress` |
| `min_confidence` | 0.4 | persons below it are not evidence |
| `box_motion` | 0.5 | share of `stationary_motion` a vehicle's box must move too for it to be moving; 0 = the reference point alone |
| `parked_hold_s` | 0.75 | s a moving vehicle must read parked in a row before it is parked; 0 = no hold |
| `vehicle_band` | 0.25 | share of the vehicle's mask rows, from the bottom, that the gap is measured to; 0 = its lower outline only |
| `gap_smoothing_frames` | 1 | a pair's gap is the median of its last this many analysed frames; 1 = as measured |
| `line_steady_frames` | 7 | the drawn line's end on the vehicle follows where it lay over this many analysed frames (display only) |
| `depth_model` | `""` | the YOLO26m-depth ONNX (`dvc pull`, or `just proximity-models` exports it); empty = no depth check anywhere |
| `depth_max_gap` | 0.25 | relative depth gap at or under which a pair is at the same depth: the default of a zone that sets none (Ticket 6: a zone setting) |
| `depth_every_s` | 0.5 | s, at most one depth frame per camera this often; a verdict older than this is refreshed |
| `scale_min_samples` | 50 | clear standing people before the scale map is used |
| `scale_refit_every` | 100 | samples between refits |
| `scale_seed` | `""` | a JSON file of maps learned beforehand, `{camera id: {"a", "b", "n"}}`, for cameras with none saved |
| `preview_lines` | true | the live preview draws distance lines |
| `preview_masks` | false | the live preview draws the mask of every person and vehicle (person masks are then read on every analysed frame) |
| `preview_boxes` | true | the live preview draws the boxes (off: no box or label at all) |
| `preview_depth` | false | the live preview blends each depth-zone camera's latest depth map (Preview settings → Depth map) |
| `preview_line_range` | 2.0 | ph: a person and a moving vehicle further apart than the zone's max gap still get a line up to this |
| `preview_box_smoothing_s` | 0.08 | s: each drawn person/vehicle box follows its detections with this time constant (display only: the rule and the published detections keep the detector's); 0 = the detection's own box |
| `preview_box_hold_s` | 0.5 | s a drawn box stays where it was after its object vanished; 0 = none |

The `preview_*` layers are switched live for every camera from the dashboard's camera wall: the app writes
`iris:ctrl:proximity_preview` (`{"masks", "lines", "boxes", "depth": bool, "line_range": ph}`, `GET`/`POST
/api/proximity/preview`) and the module picks it up within a second; until then these settings apply. The same
setting's `clip_style` ("masks", the default, or "boxes") is the app's: how alert clips and snapshots draw the
incident's pair (an object whose mask was not read gets its box).

On the preview, masks (see-through) and boxes take the colour of what they are: persons green, moving vehicles magenta,
parked ones grey, and vehicles not seen long enough to tell (their first `stationary_window_s`) blue; a box another
module colours (a zone violation) keeps that colour. Lines (over a dark outline): red once close for `persist_s`,
orange while close, pale mint beyond the max gap. A hidden foot is a ring joined by a dashed leg to the
person's lowest visible point (visible feet: a dot); in a depth zone the depth check's reading follows the gap (green
same depth, red and the line dashed apart, grey waiting or unanswered).

The defaults were tuned in Ticket 3 on labelled site footage; the numbers: `products/proximity/README.md`, "Defaults
and measured accuracy".

## Per zone: the zones file

The dashboard's zone editor writes this block (its Proximity section); the app stores only the values that differ from
`zone_defaults.json`. A zone (as the app stores it, points in mux pixels) turns the rule on with a `proximity` block and may override any of
the rule's settings for itself:

```json
"proximity": {"enabled": true, "person_classes": ["person"], "vehicle_classes": ["car", "truck"],
              "max_gap": 0.75, "stationary_window_s": 2.0, "stationary_motion": 0.75, "persist_s": 0.25,
              "grace_s": 0.5, "cooldown_s": 10, "depth_check": true, "depth_max_gap": 0.25,
              "hidden_feet": "unconfirmed"}
```

The depth check is on by default and each zone has its own cut-off (Ticket 6): on Ticket 3's labelled close pairs,
`depth_max_gap` 0.25 kept 87 % of the real near misses and held back 25 % of the false ones (0.2: 77 % and 37 %). Turn
it off where a view cannot tell depth (a bird's-eye view such as D09), and tune the cut-off from the pairs it held
back.

A block that cannot be applied (a negative gap, an unknown `hidden_feet`, under 3 points) skips that zone only, with
an error in the log. The module tests zone membership itself, from the foot point it settled on, not through
nvdsanalytics: it does not need the `zones` module.

**Hot reload:** after writing the file the app bumps `iris:ctrl:proximity_version`; the module reloads and writes the
version to `iris:analytics:proximity_applied`. By hand: `redis-cli incr iris:ctrl:proximity_version`.

## The alert

`services.alert` on `iris:alert:<camera>`, so the app turns it into an event with a clip:

| field | |
|---|---|
| `kind`, `category` | `proximity`, `safety` |
| `severity` | `critical` under half of `max_gap`, `warning` otherwise, `info` when unconfirmed |
| `zone`, `pts`, `ntp_ts` | the zone id, the frame time, the camera clock when known |
| `person_ids`, `vehicle_ids` | every track id the episode has seen so far, comma-separated |
| `vehicle_class`, `gap`, `gap_unit`, `duration_s` | the smallest gap while close (`ph`), and for how long |
| `feet_visible`, `foot_source` | `1`/`0`; `visible`, `track` (last seen foot moved with the head), `head` (head row + scale map), `unrecovered` |
| `confirmation` | `image`, `depth`, or `none` (hidden feet, nothing to confirm them) |
| `escalation` | `1` when an already alerted episode turned more severe |
| `depth_gap`, `depth_max_gap` | in a depth-check zone, the check's latest relative depth gap for the pair, and the gap at or under which it is the same depth |
| `depth_image` | a depth-confirmed alert only: a base64 JPEG of the pair's area of the depth map (both outlines drawn); the app keeps it apart and serves it at `/api/alerts/depth/<event id>` |
| `incident`, `incident_state` | one id per episode; `opened` (its first alert) or `escalated` (one more, more severe) |
| `min_gap_pts` | when the episode's smallest gap was seen (frame time): the app takes the snapshot there |

**A pair the depth check held back** (close for `persist_s`, the depth check says one is behind the other; Ticket 6)
raises no alert, but once per episode of such pairs -- unless an episode that alerted already covers it (a reading
near the cut-off flips) -- the module sends a quiet record on the same stream: the fields
above with `severity` `info`, `suppressed` `depth`, `would_be` (the severity it would have alerted at), `depth_gap`
over the zone's `depth_max_gap`, `depth_image` when a map is at hand, and no `incident` or `confirmation`. The app saves
it with its clip, tags it `depth-suppressed` and never broadcasts it: the live view never shows it, the Incidents page
only behind its **Held back by depth** filter. If the depth check later puts the pair at the same depth, it alerts as
usual.

When an episode closes (its hold ran out with no close pair, or the clock went back) one more entry goes on the same
stream with `incident_state=resolved`, `pts` (when it was last close), `duration_s`, `min_gap`, `min_gap_pts`,
`person_ids`, `vehicle_ids` and the highest `severity`: the app closes the incident's event with it (no new event).

Each analysed frame with a person and a vehicle in a proximity zone also goes on `iris:overlay:<camera>`
(`EngineServices.overlay`), close or not, so an alert's clip can follow its pair from start to end: every such pair's
line (`style` latched, close or far -- far also when the vehicle is not moving -- with the `person` and `vehicle` ids)
and each of their boxes once, with the vehicle's `state` and the mask's outline (`mask`, a polygon) when there is
one, in mux pixels. Riders and drivers get no line. The app draws the incident's pair from them on the alert's clip and
its snapshot (app/overlay_track.py). A few KB an entry: up to about 5-7 MB of Redis per busy camera at the stream's
cap.

## Hidden feet

Feet count as hidden when the mask touches the frame's bottom edge, or it is under 0.8 of the standing height its head
row predicts. They come back from the person's own height: while their feet are in view, the module keeps their height
against the scale map's standing height (the median of their last 5 such sightings, for 30 s); with the feet hidden,
they stand that many standing heights below the head's row, and the column follows the head. Without that, the head row
plus the map's standing height (a person standing straight), or, before the map is ready, the last foot moved with the
head (up to 4 s). (Moving the last foot with the head put it 0.12-0.19 person heights too low as someone walked away
from the camera: the head moves less than the feet. Ticket 4 measured the own-height estimate within 0.02.)

With the map ready, hidden feet of someone showing under half a standing person are a guess: that person is not
measured -- no line, no alert, no depth check (Ticket 4: a man behind the D41 gatehouse counter, head and shoulders in
view, got feet in a truck's path). Other hidden-feet pairs are confirmed by the depth check where it is on; elsewhere
they alert flagged `confirmation=none` at info, or not at all with `hidden_feet: suppress`.

## The person-scale map

How tall a standing person is at each foot row, `h = a v + b`, learned per camera from clear, standing, untruncated
people (at least 5 different ones), robust to people partly hidden. Each person teaches it at most once a second, so
someone who lingers counts like a passer-by, and it remembers the latest 5000 samples (half an hour to two hours of a
camera). Until it is ready the rule uses each person's own box height. It is saved to `iris:proximity:scale:<camera>` at most once a minute and on stop, and restored at start;
a camera with none saved starts from its entry in `scale_seed` (maps learned from long recordings), if any.
When recent people stop matching it the log says "has the camera moved?" (at most every 10 minutes); the map re-learns
on its own. After re-aiming a camera, drop its entry from the seed file and `redis-cli del iris:proximity:scale:<camera>`
to start it afresh.

## The depth check

YOLO26m-depth through onnxruntime in a worker thread (`depth.py`), only on frames where a close pair waits for it, at
most every `depth_every_s` per camera (on the wall clock). A pair at the same depth (relative gap <= `depth_max_gap`) is
confirmed; a pair the depth puts apart is not, and stays so: its last verdict stands while a fresher one is measured.
A pair the check has never answered `persist_s` + 0.5 s after its first request goes without it (as with no model).
After its warm-up the model must be on CUDA: if ONNX Runtime fell back to the CPU, the log says so and the depth check
is off (a missing model file refuses startup). It is on in every zone by default, each judging the reading against
its own `depth_max_gap`; leave it off on bird's-eye views such as D09. Gauges:
`proximity_depth_dropped` (frames the busy worker refused), `proximity_depth_failed` (frames that raised).

## Limits

Image space, person heights, not metres (RESEARCH 5.7): gaps along the depth axis are foreshortened, a truck body
overhanging its wheels reads farther than it is, merged masks and people on vehicles confuse the geometry, and hidden
feet are only as good as their recovery. The rule says "close to a moving vehicle", not "in its path".

## Cost

The work is reading and outlining instance masks. A vehicle's mask is read on every inferred frame (its reference
point must keep coming from the same place, or a parked truck would "move" when the source changes); a person's mask
only while a zone holds a vehicle that is moving or not yet known to be parked -- otherwise the box is enough for the
scale map and the foot memory, and a yard full of parked trucks and people costs little.
