"""Tunable knobs, hot-reloaded from data/policy.json. Stored values override
DEFAULT_POLICY key by key, so a policy file written by an older version
still gets every knob added since."""
from __future__ import annotations

from pathlib import Path

from jsonstore import JsonFileStore

DEFAULT_POLICY: dict[str, float] = {
    # business targets (the configurable alerts in the use-case doc)
    "wait_target_s": 900.0,             # waiting longer than this is an excessive wait
    "service_target_s": 1800.0,         # a haircut longer than this is an excessive haircut
    "day_start_hour": 5.0,              # the shop day starts at this local hour (0-23): "today", daily
                                         # tickets, the roster confirmation
    # entrance line, centre camera (normalized distances)
    "line_band": 0.015,                 # dead band either side of the line: no flip-flopping on the threshold
    "door_band": 0.08,                  # first seen this close inside the door = walked in (entry observed)
    "exit_confirm_s": 30.0,             # crossed out and away this long (or out of every view) = left
    # visits
    "min_visit_s": 5.0,                 # inside this long after crossing in to count as a visit
    "waiting_debounce_s": 5.0,          # a body first seen inside counts once in the waiting area this long
    "track_gone_s": 2.0,                # a track unseen this long (on its own camera's clock) has ended
    "camera_dead_s": 150.0,             # 2026-09-26 (final B review C1b): a camera whose newest frame is this
                                         # far behind the newest frame of any camera is dead: its tracks end
                                         # (so lost_timeout_s can run). A stall the sidecar's stall-watch
                                         # revives (source_stall_seconds 15) comes back well before this
    "lost_timeout_s": 300.0,            # a customer out of every view this long has left (dated to when lost)
    # bodies
    "still_window_s": 3.0,              # "still" = no farther than still_radius over this window
    "still_radius": 0.03,               # normalized. 2026-09-26 (Task 5b): 0.02 -> 0.03; R3's customer
                                         # starts at his sit (GT 305) instead of 358. Replay IoU raw1/raw2 at
                                         # chair_gap_s 90: .02 .70/.41, .03 .72/.41, .04 .74/.43; .06 starts a
                                         # barber sitting in L4 (engine-only 277 s), as .035 + seat_jitter_s 3 does.
                                         # .03 is the conservative pick (.04 alone scored .74/.43)
    "still_box_frac": 0.1,              # 2026-09-24: ...or this x the box height, if larger. Task 16 tune: a
                                         # seated customer's box middle moves p50 0.034 / p90 0.116 box heights
                                         # over 3 s while a barber works (replay of one lap: IoU 0.63 -> 0.72)
    "staff_min_samples": 3.0,           # staff vote needs this many sampled crops
    "staff_threshold": 0.70,            # 2026-09-26 (project A): the trained classifier's calibrated staff_p
                                         # (was 0.55 for zero-shot CLIP). Val, per crop: 0.50 93.2% staff /
                                         # 2.1% customers pass, 0.55 92.8/1.8, 0.70 92.0/1.1, 0.90 87.1/0.4;
                                         # a customer called staff loses his track and pre-haircut visit
    "seat_staff_threshold": 0.9,        # 2026-09-24: in a seat, a body whose vote says staff may still sit
                                         # unless its staff_p mean reaches this, or its vote said staff outside
                                         # every seat (tracklets.seat_ok). Task 16 tune: seated customers'
                                         # per-track means run 0.39-0.86 (R3 0.84, R4 caped 0.74-0.81),
                                         # barbers by the chairs 0.73-0.76
    # 2026-09-29 (faster staff correction, off by default): the staff vote averages only the last
    # staff_window crops (15 = today; 5 proposed: a barber on a customer's swapped track is released at
    # p90 ~4 s instead of ~11 s, dev/staff_hold_rules.py); staff_latched_release k > 0 = a track latched
    # as a customer votes staff once its last k crops are all >= seat_staff_threshold (0 = never, today)
    "staff_window": 15.0,
    "staff_latched_release": 0.0,
    "staff_outside_confirm_s": 20.0,    # 2026-09-26 (Task 5a): the vote must say staff outside every seat
                                         # this long on end before the track is ruled out of the seats
                                         # (0 = the first frame, as before). Seated customers' seat test
                                         # misses for up to 14.4 s with a staff vote (raw1/raw2 replays);
                                         # IoU raw1/raw2: 0 s .31/.34, 5 s .45/.33, 15-30 s .45/.40
    # identity across cameras
    "sync_window_s": 3.0,               # paired-zone entries this close in time are "simultaneous"
    "reid_veto": 0.50,                  # 2026-09-24: Task 2's offline check on TAO Swin-T (stretch) --
                                         # negative pairs p50 0.497, p99 0.811 -- below this a handoff
                                         # candidate is ruled out
    "reid_margin": 0.05,                # two handoff candidates closer than this can't be told apart: wait
    "reid_match_min": 0.82,             # 2026-09-24: Task 2's offline check on TAO Swin-T (stretch) --
                                         # positives score above the negative p99 (0.811) 97.4% of the
                                         # time; a lost customer is re-attached at or above this similarity
    "reattach_s": 120.0,                # ...if lost no longer than this
    # 2026-09-28 (39-min testbed spec, 6.3): handoff rule. 0 = today: a candidate already inside for
    # minutes still counts; 1 = a partner qualifies only if its entry is within sync_window_s of the
    # new track's entry; 2 = rule 1, plus for chair areas the new track must be still to take a handoff;
    # 3 = rule 2, and still in that chair's seat (the barber standing still by it took the handoff under 2);
    # 4 = rule 1, and for chair areas in that chair's seat for still_window_s, stillness not needed
    "handoff_rule": 0.0,
    # 2026-09-27 (docs/superpowers/specs/2026-09-27-barber-39min-testbed-identity-design.md, section 4):
    # switches the 39-min evaluation compares; 0 = today's behaviour for each
    "create_in_owned_areas": 0.0,       # 0 = a missed entry is created only in a waiting area its camera
                                         # owns; 1 = also in an owned chair or washing area (option A)
    "identity_model": 0.0,              # 0 = every track is evidence everywhere; 1 = hybrid: only a track in
                                         # an area its camera owns (or with no owner) is evidence there;
                                         # 2 = full transfer: 1, and only such a track (or a centre-camera
                                         # track outside every zone) keeps its person from being lost
    "reid_gallery": 0.0,                # 0 = one blended re-ID mean per person; 1 = one mean per camera,
                                         # compared with the track's own camera's; 2 = one mean of samples
                                         # taken in an area the sampling camera owns
    # chairs
    "seat_dwell_s": 10.0,               # still in the seat this long starts a haircut (backdated)
    "seat_jitter_s": 2.0,               # out of the seat this briefly doesn't reset the sit
    "chair_gap_s": 90.0,                # a haircut survives no customer evidence at the chair this long.
                                         # Earlier history, 2026-09-24, Task 16 tune (pre-5a code): caped
                                         # customers go undetected behind the barber for 20-40 s (seat box
                                         # gaps); replay IoU 20 s 0.44, 30 s 0.63, 45 s 0.72 (60/90 s no
                                         # better); holdout lap 0.22 -> 0.35.
                                         # 2026-09-26 (Task 5b): 45 -> 90. After Task 5a the L2 customer's box
                                         # gaps (barber in front) run 45-75 s and cut his haircut into pieces
                                         # under min_service_s. Replay IoU raw1/raw2: 45 .45/.40, 60 .59/.40,
                                         # 75-100 .67/.41, 110+ keeps a barber sitting in L4 as a haircut
    # 2026-09-28 (39-min testbed spec, 6.1): chair resume. A still body sitting down in a chair's seat
    # (owner camera) within chair_resume_s of that chair's haircut ending is its previous customer --
    # unless he is seen elsewhere (a live track on the sofa or at the wash, or out through the door),
    # or his re-ID reference from the owner camera says no (cosine <= resume_sim_diff). 0 = off (today)
    "chair_resume_s": 0.0,              # proposed 180
    "resume_sim_same": 0.70,            # at or above: the previous customer
    "resume_sim_diff": 0.50,            # at or below: someone else; in between (or no embedding): him
    # 2026-09-28 (39-min testbed spec, 6.2): a bound track whose staff vote turns True is released.
    # 0 = today: a person left with no track has his visit voided (no haircut yet); 1 = void only a
    # person that very track created and who never had another track; anyone else is lost (lost_timeout_s
    # ends it; seat claim, re-attach, handoff, chair resume bring him back), and a still, non-staff,
    # unbound body in his running haircut's seat (owner camera) is bound to him at once (staff_recover)
    "staff_release_mode": 0.0,
    # 2026-09-28 (39-min testbed spec, 6.6): seat claim. 0 = today; 1 = a body in the seat for
    # still_window_s may be claimed though it jitters (a customer being worked on); 2 = 1, and the
    # customer's other track in that camera -- not in the seat, and matching his camera reference worse
    # than the seated body (both embeddings needed) -- is released for it (binding kind seat_swap)
    "seat_claim_mode": 0.0,
    # 2026-09-28 (user): 1 = a running haircut whose customer no camera sees is kept going while the
    # chair's owner camera sees a body voted staff in its seat (the barber hides the customer)
    "barber_cover": 0.0,
    # 2026-09-29: 1 = while a customer's haircut runs in a chair, only a track in that chair may be
    # re-attached to him (bodies in the next chair / aisle at re-ID 0.83-0.86 took him, ending the haircut);
    # 2 = only a track in ANOTHER chair is refused (an aisle / no-zone track may still take him)
    "reattach_anchor": 0.0,
    # 2026-09-29 (user, flagged to fix later): 0 = no customer is ever put in the washing state (a haircut
    # ends as any other; one already washing is ended at once); 1 = today's wash trip inside the haircut
    "washing_state": 1.0,
    # 2026-09-29 (user): customers page vs completed haircuts, and clean customer crops.
    # register_at 0 = register a visit when it ends; 1 = when its first haircut completes, topped up
    # with the later crops when it ends. crop_check 1 = faces joining a known customer must be
    # customer_face_min_sim like his stored faces; body crops, per camera, body_min_sim like the visit's
    # medoid body and his stored bodies (an ID swap slipped other people's crops in)
    "register_at": 0.0,
    "crop_check": 0.0,
    "customer_face_min_sim": 0.6,        # 2026-09-29 (user): 0.45 let barber and other customers' faces in
    "body_min_sim": 0.6,
    # 2026-09-29 (user): E-crops live check showed body re-ID can't tell caped customers apart (different
    # customers 0.36-0.85 vs same 0.74-0.94). body_from_seat 1 = a body crop joins the visit only when its
    # track sits in the seat of his running haircut; face_min_quality = faces below it never join
    "body_from_seat": 0.0,
    # 2026-09-29: a visit is never linked to a customer last seen less than this many hours before it
    # entered (0 = off); door entries (face register) count too, from their door crossing time. Top-down faces of different men matched: 5 visits in 25 min became one customer
    "min_return_gap_h": 0.0,
    # 2026-09-29 (user): 0 = never link a visit to an existing customer (the page keeps the day's
    # customers, one visit each, until return matching is validated); 1 = face matching as designed.
    # In chair mode the consumer forces it from register_source (0 chair crops: 0, 1 door faces: 1).
    "link_returns": 1.0,
    # 0 = customer cards from the haircut chair's body crops, 1 = the face register at the door (2026-10-06, user: faces
    # again; tuned on the 39-min lap, see products/barber/dev/chairrun/FACES.md), 2 = faces of the seated customer
    # from face_area zones (2026-10-14, branch 2)
    "register_source": 1.0,
    "face_min_quality": 0.0,
    "min_service_s": 180.0,             # shorter chair sits are not haircuts
    "avg_wait_min_s": 30.0,             # 2026-10-12 (user): waits shorter than this stay out of the average wait
                                        # (lap 2026-10-12 synced3: 15 of 53 waits under 30 s, average 124 -> 167 s)
    "wash_return_s": 900.0,             # back in the same seat within this after washing = same haircut
    # 2026-10-04 (project 3, chair engine; spec 2a; defaults from the p1b seat-rule spike).
    "ce_lower_frac": 0.35,          # the bottom fraction of a person's mask that is measured on a seat
    "ce_seat_min": 0.10,            # that part covers at least this share of the seat = occupied
    "ce_conf_min": 0.25,            # detector confidence floor for the seat rule
    "ce_carry_outline": 0.35,       # a staff mask covering this share of the chair outline = staff cover
    "ce_carry_seat": 0.10,          # ... or this share of the seat
    # 2026-10-15 (user; Palm Hills test hour, .run/gtp/sweep): < 1 = staff cover counts only when the bottom this share
    # of the staff mask is on the seat (outline test off). Lap 3 replay: 1.0 / 0.5 / 0.4 keep the barber's cover alive
    # through both 60-70 s hand-overs (the next customer merged into the haircut); 0.35 / 0.30 none, while still
    # covering the customer hidden by his barber; 0.20 covers nothing. 1.0 = today's rule (whole mask, outline or seat).
    "ce_carry_lower_frac": 1.0,
    # 2026-10-15 (Palm Hills, L7 moved to ph-5): the seat rule keeps a staff track staff until this many non-staff
    # samples in a row (one 0.67 among 0.95s made a resting barber a sitter; the start chain backdated the next
    # customer's start 45 s to it). 1 = the latest sample decides (today).
    "ce_staff_release_n": 1.0,
    # 2026-10-04 (project 3, chair engine; spec 2b, Task 7).
    "ce_start_window_s": 10.0,      # occupied in >= ce_start_frac of this window starts a sitting (backdated)
    "ce_start_frac": 0.70,
    "ce_end_gap_s": 60.0,           # no evidence (occupied or staff cover) this long ends the sitting
    "ce_dropout_s": 5.0,            # a gap between a camera's frames longer than this pauses its chairs' clocks
    # 2026-10-04 (project 3, chair engine; spec 2c, Task 8).
    "ce_resume_window_s": 360.0,    # a chair taken again within this of its sitting's end resumes that haircut
    # 2026-10-07 (resume_check.py; dev/chairrun/same_person/FINDINGS.md, bekky-50min, 7 real cases, one recording):
    # 1 = a chair taken again within ce_resume_max_s asks re-ID (stream embedding, the new sitter's first
    # ce_resume_new_s vs the previous customer's last ce_resume_ref_s). Within ce_resume_window_s he is the previous
    # customer unless sim < ce_resume_sim_low; after it only if sim >= ce_resume_sim_high (the finished haircut
    # reopens). 7/7 for low 0.60-0.65 x high 0.80-0.85; the time rule alone 6/7. 0 = the time rule (today).
    "ce_resume_check": 0.0,
    "ce_resume_max_s": 1800.0,
    # 2026-10-18 (user): 0.65 (was 0.62). Bekky labels, last 60 s: no same-customer pair under 0.70 (4 real, 22 built),
    # 9 of 113 different-people pairs in 0.62-0.65 now called new; Palm GT hour: no check in that band.
    "ce_resume_sim_low": 0.65,
    "ce_resume_sim_high": 0.82,
    "ce_resume_ref_s": 60.0,           # 2026-10-07 review: was 300 (+ 3 picked samples); see resume_check.py
    "ce_resume_new_s": 30.0,
    # 2026-11-10 (user: two VIP1 haircuts on 16 Oct were staff members' own): 1 = on this chair, a confirmed haircut whose
    # sitter looked staff (staff_threshold) in ce_staff_cut_share of his samples in the minute before he sat down AND in
    # the two minutes after he got up is a staff haircut: its session is deleted and its visit voided as staff. 0 = off.
    # Per chair: around the busy chairs a barber or cleaner by the seat fakes the cue (R2 15:31 on 16 Oct).
    "ce_staff_cut_check": 0.0,
    "ce_staff_cut_share": 0.8,
    # 2026-10-18 (user, Palm Hills L6 on ph-5: the barber cleaning the empty chair is cover, so the next customer's
    # sitting ran on in the first one's -- one haircut for two). > 0 (with ce_resume_check 1): a sitting's seat taken
    # again after this long with no one seen in it (cover only) asks re-ID the same question as a return: the same
    # customer (nothing changes) or a new one (the haircut ends at the last frame he was seen, a new one starts).
    "ce_reseat_check_s": 0.0,
    # 2026-10-18 (user: a bag on Palm Hills' VIP2 became a 7-min haircut): > 0 = a haircut is confirmed only once its
    # occupant's box has moved -- the spread (p90 - p10) of its top or its left edge, a fraction of the frame, above
    # this. The bag stayed within 0.002 for 5 min over 3 tracks; every real sitting that day moved 0.012 or more
    # (the larger of the two). A still customer is confirmed as soon as he moves. 0 = off.
    "ce_object_spread": 0.0,
    # 2026-10-22 (user: clothes left on Palm Hills' VIP2 at closing were a 5-min haircut): > 0 = a haircut is confirmed
    # only once the median detection conf of its occupant's boxes so far reaches this, too. The staff re-arranging the
    # pile and its box flickering between two outlines made the clothes "move" (0.043); the detector was half sure of
    # them (0.66; the 18 Oct bag 0.48), of every real VIP haircut on 16 and 18 Oct 0.81-0.88 (a crowded start confirms
    # later, it is not dropped). 0 = off.
    "ce_object_conf": 0.0,
    "ce_min_haircut_s": 60.0,       # less sitting time than this in total = no haircut, no customer
    # 2026-11-10 (user, Palm Hills 16 Oct: R1 18:20 was its customer back from the wash, re-ID 0.52 "new", 6.3 min):
    # > ce_min_haircut_s = a haircut opened by a re-ID "new customer" answer needs this much sitting to count. 0 = off.
    "ce_min_after_new_s": 0.0,
    # 2026-11-10 (user, Palm Hills pedicure chairs): a pedicure counts after this much sitting (16 Oct: one real 41-min
    # pedicure, every other occupancy of the area under 2 min). Pedicures are never customers.
    "pd_min_s": 300.0,
                                     # 2026-10-04 (final review I-1 fix; the end gap no longer counts): bekky-50min
                                     # offline and live lap alike, 30 = 60 = 120 (err 4, customers 9: the 15 s R4
                                     # fragment is gone), 180 drops 2 real customers (err 6). One recording only.
    "ce_picture_window_s": 180.0,   # body crops for the customer card come from the haircut's first minutes
    # 2026-10-04 (Task 14b; dev/chairrun/RESULTS.md, bekky-50min door diagnosis: people walking along the glass
    # hook across the line in bursts; net crossing with a hold cut door errors 38 -> 5; plateau hold 2-5 s, lost 0.5-2 s).
    "ce_door_confirm_s": 2.0,       # a door crossing counts once the foot stayed this long on the new side
    "ce_door_lost_s": 1.0,          # ... or the track was unseen this long after crossing (dated to the crossing)
    # 2026-11-13 (user; Palm Hills 6 Nov, L7 21:08: a staff member brushing an empty chair's zone -- cover in 3-14 % of
    # frames -- held the sitting 100 s, into the next customer's): > 0 = a cover frame holds a sitting only while cover
    # fills this share of the last ce_cover_window_s of frames. 0 = any cover frame (as before). Per chair.
    "ce_cover_min_frac": 0.0,
    "ce_cover_window_s": 10.0,
    # 2026-11-14 (user; Palm Hills 6 Nov L6 19:47, 16 Oct L5 14:09 / 14:54): a haircut whose sitter was on the seat in
    # less than this share of its sittings' frames is kept but marked "low occupancy" (possibly not a customer): the
    # report counts it apart and highlights it. 0 = off. Per chair. A short haircut (less than ce_low_occupancy_short_s
    # of sitting) needs the higher ce_low_occupancy_short (if set): the standing staff member's 10 min at L6 reached
    # 30 %, a real customer hidden by his barber for 21 min only 26 %; short real haircuts were all above 40 %.
    "ce_low_occupancy": 0.0,
    "ce_low_occupancy_short": 0.0,
    "ce_low_occupancy_short_s": 900.0,
    "ce_carry_max_s": 120.0,        # 0 = no cap; staff cover alone keeps a sitting alive at most this long after the
                                     # last occupied frame (spec 2c). 2026-10-04 (Task 14b, bekky-50min seg): 120 and 180 both
                                     # fix R4-2's +906 s end (err 5 -> 4, nothing added); 300 only halves it; 0 = old behaviour
                                     # (barber cover bridged every 60 s gap). One recording only.
                                     # 2026-10-12 (user: away shows too late): 180 -> 120. Replays: bekky-50min live lap
                                     # unchanged (err 2), 1 away 60 s sooner; synced3 lap b: 4 aways 50-70 s sooner, R4's
                                     # haircut no longer held ~9 min by the R3 barber standing over its seat; one reclined
                                     # shave (R1, end of lap) hidden > 3 min shows away 60 s sooner than 180 would.
    # 2026-10-04 (Task 16c, dev/chairrun/cover_check): on the live lap the barber serving the reclined R3 customer
    # stands in R4 (cover) and his track sometimes carries a customer's staff_p (id switch) or none yet: lone
    # "occupied" frames at 1136 / 1148 s restarted the cap and R4-2 ended at 1294 s (GT 964). Live err 5 -> 4,
    # offline 4 -> 4, nothing else moved; clean for run 2-5 (after 60 s) and after 10-60 s (run 3). bekky-50min only.
    # ce_refresh_run_n counts FRAMES, so it is rate-dependent: 3 is ~0.4 s live (7.3 Hz meta) and 0.25 s offline (12 Hz).
    "ce_refresh_run_n": 3.0,        # more than ce_refresh_after_s after the last occupied frame (dropout-paused clock),
    "ce_refresh_after_s": 30.0,     # only the n-th consecutive occupied frame counts again (1 = every frame, old)
    # 2026-10-05 (haircut accuracy; dev/chairrun/accuracy/FINDINGS.md, fixscore.py: real-code replay, bekky-50min).
    # Haircut errors off/lap/eng/l16/l16rim2: both off 4/5/4/6/4, chain only 2/4/2/4/2, gate only 4/5/3/5/3,
    # both 2/4/1/3/1 (= the sim); door untouched. Plateaus: chain 30-90 s, gate 0.20-0.45. One recording. 0 = off.
    "ce_start_chain_s": 45.0,       # empty/starting: occupied frames closer than this chain; a confirmed start is
                                     # backdated to the chain's first frame, and a live chain keeps an away haircut taken
    "ce_seat_inside_min": 0.30,     # a person occupies a chair only if this share of the whole mask is in its outline
    # 2026-10-05 (review fix): a chain older than this (first frame to now) is dead: not taken, no backdating (stray
    # frames < ce_start_chain_s apart held a haircut forever). The longest chain that fixed an error: 186 s (off R4).
    "ce_start_chain_max_s": 240.0,
    # 2026-10-05 (far-chair per-chair rules; this shop's L5, the far chair of barber-r: seated hit rate ~10 % at conf
    # 0.25; dev/chairrun/farchair/FINDINGS.md, real-code sweep l5sweep.py -> out/l5table.txt, bekky-50min, 5 runs).
    # "<key>@<chair>" = that knob for that chair only (zone names are per install: another shop overrides or ignores
    # these). Haircut errors lc2/base2/lc/eng/l16rim2 none 2/3/3/1/1 (L5 missed or +1131 s) -> 0/1/1/2/1, other chairs
    # unchanged, 0 false sittings; = seat 0.06 / gap 600 everywhere (0.06 starts L5 +538 s on the live lap vs +1132,
    # both one start error); seat 0.09 loses L5's start on lc and l16rim2; gap <= 300 ends L5 at 2153 (one end error per
    # offline run). Gap 900 would also keep L5 to the clip end on eng and l16rim2 (-1 error each). One recording.
    "ce_seat_min@L5": 0.08,
    "ce_start_window_s@L5": 30.0,
    "ce_start_frac@L5": 0.30,
    "ce_end_gap_s@L5": 400.0,
    # 2026-10-12 (user: the lap-start haircut on L4 starts late): L5's start rule on L4 too. The barber stands over the
    # caped customer; l-seg merges the two (scored staff) or misses him -> occupied 0-36 % per 30 s, start at 342 s.
    # Replays: synced3 lap b L4 starts at 1 s, nothing else changes; bekky-50min err 2 unchanged. (Not counting staff
    # cover frames in every start window instead: bekky err 2 -> 5, rejected.)
    "ce_start_window_s@L4": 30.0,
    "ce_start_frac@L4": 0.30,
    # 2026-10-04 (project 4 C, "Waiting now"; spec 2026-10-04-barber-waiting-count-design.md).
    "wc_lower_frac": 0.35,          # the bottom fraction of a person's mask tested against the waiting zone
    "wc_zone_min": 0.5,             # that part must lie this much inside the zone
    "wc_dwell_s": 9.0,              # in the area this long before counting (user, 2026-10-04: walking through)
    "wc_dwell_gap_s": 2.0,          # out of the area longer than this restarts the dwell
    "wc_stale_s": 5.0,              # a camera whose newest frame is older than this is left out of the count
    # 2026-10-04 bekky-50min replay (dev/chairrun/WAITING.md): up 5 / down 10 vs the first 2 / 20, vs GT seated +
    # standing: MAE 0.24 -> 0.13 (live lap 0.27 -> 0.11), exact 77% -> 87%, reversals 10 -> 2 (live 15 -> 7), for rises
    # ~3 s later; wc_zone_min 0.3 / 0.5 / 0.7 made no difference (kept 0.5).
    "wc_up_s": 5.0,                 # a higher count held this long raises "Waiting now"
    "wc_down_s": 10.0,              # a lower count held this long lowers it
    # Customer waits (chair engine, project 4 A + B, 2026-10-05; centre camera only; starting values, tuned by eye
    # from the user's feedback -- spec docs/superpowers/specs/2026-10-05-barber-waits-design.md)
    # 2026-10-14 (Palm Hills, branch 2 spec section 6): 0 = the wait follows its track to a row, the door or loss
    # (Bekky); 1 = it ends when the customer steps out of every waiting zone, and a returning customer is matched by
    # his own track or a strict re-ID and reopens it (time away counted)
    "wt_end_mode": 0.0,
    "wt_out_s": 3.0,             # wt_end_mode 1: out of every waiting zone this long ends the wait
    "wt_return_sim": 0.82,       # a return reopens a wait when the mean re-ID cosine reaches this (strict)
    "wt_return_max_s": 1800.0,   # ...and the wait ended no longer ago than this
    "wt_return_min_samples": 2.0,  # re-ID samples a new dwell needs before it is compared
    "wt_return_extra_s": 30.0,   # a dwell short of samples waits this much longer for them
    # 2026-10-14 (spec section 7, user: tight, 150 s): a wait counts as served when a haircut starts within this of its end
    "wt_serve_s": 150.0,
    # 0 = visits from the door (baseline + door ins); 1 = door-free: served + waits that never led to a haircut
    "visits_source": 0.0,
    "wt_start_s": 9.0,          # in the waiting area this long (gaps <= wc_dwell_gap_s) starts a wait (user's threshold)
    # a wait starts only if the median detection confidence over the start dwell reaches this (2026-10-05: a bag on the sofa, median 0.55)
    "wt_conf_min": 0.6,
    "wt_row_min": 0.5,          # a row zone holds a person when this much of his lower mask lies inside it
    "wt_row_s": 3.0,            # in a row zone this long ends his wait (he has gone to a chair)
    "wt_lost_s": 2.0,           # a waiting track unseen this long puts its wait on hold
    "wt_reconnect_s": 10.0,     # a held wait is taken over by a newly seen nearby track within this long of its last sighting
    "wt_reconnect_dist": 0.10,
    # 2026-10-07 (review finding 3; waits.py docstring): one odd staff sample no longer ends a wait, a lost wait's
    # own track back within wt_rejoin_s rejoins it, a silent centre camera ends waits instead of aging them.
    "wt_staff_samples": 3.0,    # a holder turns staff after this many staff samples in a row
    # 2026-10-15 (user, Palm Hills side sofa; extra_waits.py): 1 = a waiting zone paired with zones of other cameras
    # counts the higher of its own count and theirs added up; people beyond the open waits get extra waits (no track).
    # Lap 3 + 1 simulation: the two teens seen as one by ph-3 -> 2 waits, nothing else added. 0 = off (today).
    "wt_extra_views": 0.0,
    "wt_extra_end_s": 30.0,     # the count below the open waits this long ends the newest extra wait (dated back)
    # 2026-10-15 (sub-project 4, the kids car; kids_car.py): a one-class detector on one go2rtc frame per camera
    # every kc_every_s places the Kids chair where the car stands. 0 = off (the stored Kids zones).
    "kc_enabled": 0.0,
    "kc_every_s": 20.0,         # one snapshot per camera this often (user, 2026-10-15)
    "kc_conf_min": 0.35,        # a detection counts from this confidence (held-out clips: P 1.0, R 0.9-1.0 at 0.25-0.35)
    "kc_confirm_n": 2.0,        # snapshots in a row at a new place before the chair moves
    "kc_lost_s": 300.0,         # unseen this long: no Kids chair until the car is found
    "kc_release_s": 120.0,      # a manual place yields after the car is seen elsewhere this long
    "wt_rejoin_s": 120.0,       # a wait ended lost is rejoined by one of its own tracks back within this
    "wt_stale_s": 60.0,         # the centre camera silent this long (engine clock): open waits end lost  # ... first seen no further than this (fraction of the frame width) from that sighting
    # Haircut photos (2026-10-22, user; spec 2026-10-22-barber-haircut-photos-design.md): 1 = every haircut gets a
    # start photo and an end photo of its chair -- a close-up, one go2rtc frame each -- for the photo report. 0 = off.
    "hp_photos": 0.0,
    "hp_end_every_s": 120.0,    # while the customer is in the chair, a new end photo this often (the newest is kept)
    "hp_crop_margin": 0.25,     # the close-up: the chair outline joined with the sitter's box, widened this much of
                                # its width on each side
    "hp_crop_head": 0.5,        # ... and raised this much of its height above, for the head
    "hp_photo_px": 640.0,       # saved at most this wide
    # live view
    "scene_fps": 5.0,                   # scene pushes per second (0 = only with the full snapshot)
    # customer register (project C)
    "visit_max_faces": 15.0,            # best face candidates kept per visit
    "visit_max_bodies": 10.0,           # best body candidates kept per visit
    "match_threshold": 0.70,            # 2026-10-06: 0.70 (was 0.62, which merged 2 different people on the
                                         # 39-min lap's top-down door faces); strict, provisional;
                                         # re-test later. Link only at or above this
    "match_margin": 0.05,               # strict, provisional; the 2026-09-25 gate failed on a thin sample;
                                         # re-test later. ...and this far ahead of the runner-up; otherwise a
                                         # new customer
    "visit_face_min_sim": 0.43,         # strict, provisional; the 2026-09-25 gate failed on a thin sample;
                                         # re-test later. A visit's face this unlike the visit's medoid face
                                         # is dropped (a stray face)
    "min_visit_faces": 1.0,             # fewer clean faces than this = no usable face (face_count 0, never matched);
                                         # 2026-10-12 (user): 2 -> 1, one valid face makes a card (lap: 3 -> 12 of 27)
    "min_link_faces": 2.0,              # a link to a known customer needs this many face moments (review finding 4)
    "customer_max_faces": 5.0,         # best face crops kept per customer
    "customer_max_bodies": 3.0,        # best body crops kept per customer (reference only)
    "customer_retention_days": 365.0,  # a customer with no visit for this long is deleted
    # 2026-10-05 (user): which zone types the dashboard's camera views draw, for every screen
    "zone_show_entrance_line": 1.0, "zone_show_chair": 1.0, "zone_show_seat": 1.0,
    "zone_show_waiting": 1.0, "zone_show_washing": 1.0, "zone_show_row": 1.0, "zone_show_door_area": 1.0,
    "zone_show_face_area": 1.0,
    # Face register at the door (2026-10-05): a non-staff `in` crossing registers its track's faces from fr_before_s
    # before to fr_after_s after the crossing (at crossing + fr_after_s); unclaimed faces are kept fr_keep_s.
    "fr_before_s": 8.0,
    "fr_after_s": 30.0,                # 2026-10-06: faces come 4-20 s after the crossing, walking toward the camera;
                                       # 2026-10-11: 30, night entrants walk in slower (capture more entrants)
    "fr_keep_s": 20.0,
}

# Upper bounds for knobs where a large value is a mistake, not a preference.
MAXIMUMS: dict[str, float] = {"staff_window": 15.0, "staff_latched_release": 15.0,  # the vote keeps 15 crops
                              "scene_fps": 30.0, "staff_threshold": 1.0, "seat_staff_threshold": 1.0, "reid_veto": 1.0,
                              "reid_match_min": 1.0, "reid_margin": 1.0, "still_radius": 1.0,
                              "still_box_frac": 1.0, "match_threshold": 1.0, "match_margin": 1.0,
                              "visit_face_min_sim": 1.0, "day_start_hour": 23.0,
                              "create_in_owned_areas": 1.0, "identity_model": 2.0, "reid_gallery": 2.0,
                              "resume_sim_same": 1.0, "resume_sim_diff": 1.0, "staff_release_mode": 1.0,
                              "handoff_rule": 4.0, "seat_claim_mode": 2.0, "barber_cover": 1.0, "reattach_anchor": 2.0, "washing_state": 1.0, "register_at": 1.0, "crop_check": 1.0,
                              "ce_lower_frac": 1.0, "ce_seat_min": 1.0, "ce_conf_min": 1.0, "ce_carry_outline": 1.0, "ce_carry_seat": 1.0, "ce_carry_lower_frac": 1.0, "ce_staff_release_n": 15.0,
                              "ce_start_frac": 1.0, "ce_seat_inside_min": 1.0, "ce_resume_check": 1.0, "ce_resume_sim_low": 1.0, "ce_resume_sim_high": 1.0, "ce_start_chain_s": 600.0,
                              "ce_start_chain_max_s": 3600.0, "wc_lower_frac": 1.0, "wc_zone_min": 1.0,
                              "wt_row_min": 1.0, "wt_reconnect_dist": 1.0, "wt_conf_min": 1.0,
                              "customer_face_min_sim": 1.0, "body_min_sim": 1.0, "body_from_seat": 1.0, "face_min_quality": 1.0, "link_returns": 1.0, "register_source": 2.0,
                              "zone_show_entrance_line": 1.0, "zone_show_chair": 1.0, "zone_show_seat": 1.0, "zone_show_waiting": 1.0, "zone_show_washing": 1.0, "zone_show_row": 1.0, "zone_show_door_area": 1.0,
                              "wt_end_mode": 1.0, "visits_source": 1.0, "wt_extra_views": 1.0, "kc_enabled": 1.0, "kc_conf_min": 1.0, "kc_confirm_n": 10.0, "wt_return_sim": 1.0, "zone_show_face_area": 1.0,
                              "ce_object_spread": 0.1, "ce_object_conf": 1.0, "ce_staff_cut_check": 1.0, "ce_staff_cut_share": 1.0, "pd_min_s": 3600.0, "ce_cover_min_frac": 1.0, "ce_low_occupancy": 1.0, "ce_low_occupancy_short": 1.0, "ce_low_occupancy_short_s": 7200.0,
                              "hp_photos": 1.0, "hp_end_every_s": 1800.0, "hp_crop_margin": 1.0, "hp_crop_head": 2.0,
                              "hp_photo_px": 1280.0}

# Switches: whole numbers only (2026-09-27, 39-min testbed spec).
INTEGER_KEYS = {"staff_window", "staff_latched_release", "ce_staff_cut_check",
                "create_in_owned_areas", "identity_model", "reid_gallery", "staff_release_mode", "handoff_rule",
                "seat_claim_mode", "barber_cover",
                "reattach_anchor", "washing_state", "register_at", "crop_check",
                "body_from_seat", "link_returns", "register_source", "ce_refresh_run_n", "ce_resume_check", "wt_staff_samples", "ce_staff_release_n", "wt_extra_views", "kc_enabled", "kc_confirm_n",
                "zone_show_entrance_line", "zone_show_chair", "zone_show_seat", "zone_show_waiting", "zone_show_washing", "zone_show_row", "zone_show_door_area",
                "wt_end_mode", "visits_source", "wt_return_min_samples", "zone_show_face_area", "hp_photos", "hp_photo_px"}

# Lower bounds above 0. 2026-09-26 (final B review I4): staff_min_samples 0 divided by zero every cycle.
# 2026-10-04 (project 3 final review I-2): chair-engine knobs where 0 breaks the engine -- a 0 start window
# crashed every cycle, a 0 dropout / end gap stops every sitting, a 0 seat / carry share lets anyone anywhere
# occupy or cover a chair.
# 2026-10-05: a 0 wait threshold / lost time / reconnect distance / row share breaks waits. (0 stays valid where it means "off": resume window, min haircut, door holds, carry cap.)
MINIMUMS: dict[str, float] = {"staff_min_samples": 1.0, "staff_window": 1.0, "ce_refresh_run_n": 1.0,
                              "ce_start_window_s": 1.0, "ce_dropout_s": 1.0, "ce_end_gap_s": 1.0,
                              "ce_seat_min": 0.01, "ce_carry_outline": 0.01, "ce_carry_seat": 0.01, "ce_carry_lower_frac": 0.01,
                              "wc_lower_frac": 0.01, "wc_zone_min": 0.01, "wc_stale_s": 1.0,
                              "ce_start_chain_max_s": 1.0, "ce_resume_ref_s": 1.0, "ce_resume_new_s": 1.0,
                              "wt_staff_samples": 1.0, "ce_staff_release_n": 1.0, "wt_extra_end_s": 1.0, "kc_every_s": 5.0, "kc_conf_min": 0.01, "kc_confirm_n": 1.0, "kc_lost_s": 20.0, "kc_release_s": 20.0, "wt_stale_s": 1.0,
                              "wt_start_s": 1.0, "wt_lost_s": 0.5, "wt_out_s": 0.5, "wt_return_min_samples": 1.0, "wt_reconnect_dist": 0.01, "wt_row_min": 0.01, "fr_keep_s": 1.0,
                              "hp_end_every_s": 30.0, "hp_photo_px": 320.0}


def policy_store(data_dir: Path) -> JsonFileStore:
    """data_dir/policy.json: the keys set in Setup (a policy PUT stores only those), read as a layer over the base
    (DEFAULT_POLICY <- the site's `policy:` block): effective(stored, base). No file = nothing set ({}), and nothing is
    written until the first PUT. 2026-10-15 (final review I3): the store used to write the whole base on its first
    load, which pinned every key, so a later change to the site's block never showed through. A full file written
    before still works: it just pins every key."""
    return JsonFileStore(data_dir / "policy.json", dict, persist_default=False)


# 2026-10-05 (far-chair per-chair rules, dev/chairrun/farchair/FINDINGS.md): a flat key "<base key>@<chair>" overrides
# one of these chair-engine knobs for that chair only, with the base key's bounds. <chair> is the engine's chair id
# (zones.area_of: the pair id, else the zone id -- Bekky "L5", Palm Hills "ph3-L6"), per install.
# seat.py and chairs2.py read a chair's knobs through policy_for(). Flat keys keep the policy a plain
# {name: number} map: the file, the hot reload, the endpoint and the dashboard's Advanced list take them as is.
PER_CHAIR_KEYS = frozenset({"ce_seat_min", "ce_conf_min", "ce_seat_inside_min", "ce_start_window_s",
                            "ce_start_frac", "ce_end_gap_s",
                            # 2026-10-18 (user: L5, L6, R1, L7 go away while their barber hides the customer)
                            "ce_carry_max_s", "ce_carry_lower_frac", "ce_carry_outline", "ce_carry_seat", "ce_cover_min_frac", "ce_low_occupancy", "ce_low_occupancy_short",
                            # 2026-10-18 (user): the re-ID thresholds per chair (haircuts.py reads them per chair)
                            "ce_resume_sim_low", "ce_resume_sim_high", "ce_reseat_check_s", "ce_staff_cut_check", "ce_staff_cut_share",
                            # 2026-10-18 (user: a bag on VIP2): an object is never a haircut
                            "ce_object_spread",
                            # 2026-10-22 (user: clothes on VIP2): nor one the detector is only half sure of
                            "ce_object_conf"})


def _base_key(k: str) -> str | None:
    """The knob a policy key sets: itself if known, the base of a valid "<base>@<chair>" override, else None."""
    if k in DEFAULT_POLICY and "@" not in k:
        return k
    base, sep, chair = k.partition("@")
    if sep and base in PER_CHAIR_KEYS and chair and "@" not in chair and chair == chair.strip():
        return base
    return None


def policy_for(policy: dict, chair: str) -> dict:
    """The policy one chair runs on: its "<key>@<chair>" overrides applied (the same dict when it has none)."""
    over = {k[:-len(chair) - 1]: v for k, v in policy.items() if k.endswith("@" + chair) and _base_key(k)}
    return {**policy, **over} if over else policy


def effective(stored: dict | None, base: dict[str, float] | None = None) -> dict[str, float]:
    """`base` (default DEFAULT_POLICY) overlaid with whatever the file holds, numbers only (per-chair overrides
    included). The site's defaults come in through `base`."""
    merged = dict(base) if base is not None else dict(DEFAULT_POLICY)
    for k, v in (stored or {}).items():
        if _base_key(k) is not None:
            merged[k] = float(v)
    return merged


def validate_policy_update(payload: dict) -> dict[str, float]:
    """Only known knobs (or per-chair overrides of them), only non-negative numbers -- raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("expected an object")
    out = {}
    for k, v in payload.items():
        b = _base_key(k) if isinstance(k, str) else None
        if b is None:
            raise ValueError(f"unknown policy key: {k}")
        try:
            num = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number") from None
        if num < MINIMUMS.get(b, 0.0):
            raise ValueError(f"{k} must be >= {MINIMUMS.get(b, 0.0):g}")
        if b in MAXIMUMS and num > MAXIMUMS[b]:
            raise ValueError(f"{k} must be <= {MAXIMUMS[b]:g}")
        if b in INTEGER_KEYS and not num.is_integer():
            raise ValueError(f"{k} must be a whole number")
        out[k] = num
    return out


def validate_policy(p: dict) -> None:
    """Rules between knobs, on a whole (merged) policy -- raises ValueError. 2026-10-07 (review): an inverted
    pair was accepted silently (stricter inside the wash window than after it; every wash return a new customer)."""
    if p["ce_resume_sim_low"] > p["ce_resume_sim_high"]:
        raise ValueError("ce_resume_sim_low must be <= ce_resume_sim_high")
    for chair in sorted({k.partition("@")[2] for k in p if k.startswith(("ce_resume_sim_low@", "ce_resume_sim_high@"))}):
        pc = policy_for(p, chair)
        if pc["ce_resume_sim_low"] > pc["ce_resume_sim_high"]:
            raise ValueError(f"ce_resume_sim_low must be <= ce_resume_sim_high on chair {chair}")
    if p["ce_resume_max_s"] < p["ce_resume_window_s"]:
        raise ValueError("ce_resume_max_s must be >= ce_resume_window_s")
