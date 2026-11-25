"""Barber shop domain model (three cameras).

Inputs, re-declared from the sidecar's wire format by acl.py:
  TrackObs / Frame -- one camera frame's person tracks (iris:meta),
  Appearance       -- one sampled crop's body embedding + staff probability
                      (iris:appearance).
Records:
  Visit          -- one customer, entry to exit. `state` follows the zones they
                    move through: entered -> waiting -> haircut <-> washing ->
                    served -> finished ("entered" is never persisted: a visit
                    is only confirmed after min_visit_s inside).
  ServiceSession -- one haircut in one chair (washing trips inside it).
  StateEvent     -- every persisted state change, the audit trail.
  Handoff        -- one confirmed cross-camera handoff (to tune sync_window_s).
  Binding        -- every track that joined or created a person (merge audit).
Engine time is float epoch seconds (the Redis entry id of each frame); records
carry naive local datetimes (dt()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

ENTERED, WAITING, HAIRCUT, WASHING, SERVED, FINISHED = "entered", "waiting", "haircut", "washing", "served", "finished"
COMPLETED, WALKED_OUT, LOST, VOID_STAFF = "completed", "walked_out", "lost", "void_staff"
EXIT_DOOR, EXIT_LOST = "door", "lost"
SHOP = "shop"   # alerts.camera: the shop, not one camera


def dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts)


@dataclass(frozen=True)
class TrackObs:
    track_id: int
    box: tuple[float, float, float, float]   # left, top, w, h in mux pixels
    conf: float | None = None                # detector confidence (person_masks); None = not published
    outline: tuple[float, ...] | None = None # mask outline, flat normalised [x1, y1, ...] (chair engine)


@dataclass(frozen=True)
class Frame:
    camera: str
    ts: float
    tracks: tuple[TrackObs, ...]


@dataclass(frozen=True)
class Appearance:
    camera: str
    track_id: int
    ts: float
    reid: tuple[float, ...] | None    # unit-length body embedding
    staff_p: float | None             # probability the crop shows staff
    crop_path: str | None = None
    crop_quality: float | None = None


@dataclass
class Visit:
    id: str
    person_id: str
    entered_at: datetime
    entry_observed: bool
    state: str = ENTERED
    confirmed: bool = False
    waiting_started_at: datetime | None = None
    service_started_at: datetime | None = None   # first haircut's start = end of the wait
    wait_s: float | None = None
    wait_target_s: float | None = None
    excessive_wait: bool = False
    haircuts: int = 0
    exited_at: datetime | None = None
    exit_reason: str | None = None
    outcome: str | None = None
    open: bool = True
    ticket: int | None = None                 # the daily #n, assigned on confirm
    wait_unseen: bool = False                 # first seen sitting in a chair: his wait is unknown, not 0 s
                                              # (2026-09-28, user) -- the stats leave it out
    # runtime only
    exit_crossed_at: datetime | None = None   # crossed the door outward, not confirmed yet
    session_id: str | None = None             # haircut running, or suspended while washing
    state_since: datetime | None = None       # when `state` last changed


@dataclass
class ServiceSession:
    id: str
    visit_id: str
    chair_id: str
    started_at: datetime
    ended_at: datetime | None = None
    service_s: float | None = None
    wash_s: float = 0.0
    barber_id: str | None = None     # from the roster: who works this chair at the start
    service_target_s: float | None = None
    excessive_service: bool = False
    open: bool = True
    occupancy: float | None = None   # share of its sittings' frames with the sitter on the seat (chair engine)
    low_occupancy: bool = False      # occupancy below ce_low_occupancy: possibly not a customer (2026-11-14)
    washing_since: datetime | None = None   # runtime: suspended while the customer is at the wash


@dataclass
class Alert:
    """An excessive wait or haircut: opens when the running time crosses its
    target (started_at = start + target), closes when that wait or haircut ends."""

    id: str
    kind: str
    camera: str
    visit_id: str
    session_id: str | None
    started_at: datetime
    ended_at: datetime | None = None
    open: bool = True
    subject_cam: str | None = None                                   # where the snapshot is taken (chair engine)
    subject_box: tuple[float, float, float, float] | None = None     # normalised x, y, w, h of the subject


@dataclass(frozen=True)
class StateEvent:
    visit_id: str
    from_state: str | None
    to_state: str
    area: str | None
    camera: str | None
    at: datetime


@dataclass(frozen=True)
class Handoff:
    area: str
    camera_a: str
    track_a: int
    camera_b: str
    track_b: int
    offset_s: float | None
    reid_sim: float | None
    at: datetime


@dataclass(frozen=True)
class Binding:
    """A track joined or created a person (to audit merges)."""
    kind: str                     # door | waiting | owned_area | seat_start | handoff | reattach | seat_claim | seat_swap
                                  # | chair_resume | staff_recover
    camera: str
    track_id: int
    person_id: str
    area: str | None
    partner_camera: str | None
    partner_track: int | None
    reid_sim: float | None
    offset_s: float | None
    at: datetime


@dataclass(frozen=True)
class DoorCrossing:
    """A person crossed the centre camera's door line (chair engine, project 3)."""
    at: datetime
    direction: str          # in | out
    camera: str
    track_id: int
    staff: bool
    entry_id: str | None = None     # a non-staff `in`: the door entry the face register tracks


@dataclass(frozen=True)
class SittingRecord:
    """One sitting of a haircut (a wash return is a second sitting of the same session)."""
    session_id: str
    n: int                  # 0, 1, ... in time order within the session
    chair_id: str
    started_at: datetime
    ended_at: datetime | None


@dataclass(frozen=True)
class ResumeCheckRecord:
    """One re-ID check's answer (2026-10-18, user: keep every score, to tune the thresholds per chair)."""
    chair_id: str
    kind: str                   # return: the chair taken again after its customer left | reseat: after cover only
    sat_at: datetime            # when the sitter sat down
    decided_at: datetime
    gap_s: float                # how long the seat was left (return) or unseen under cover (reseat)
    sim: float | None           # None: no re-ID samples on one side, the time rule answered
    threshold: float | None     # what sim had to reach
    resumed: bool               # True: the same customer
    haircut_id: str             # the haircut the sitter got
    prev_haircut_id: str        # the one he was compared with
    ref_samples: int
    new_samples: int


@dataclass(frozen=True)
class WaitRecord:
    """One customer wait (chair engine, project 4 A + B): open (ended_at None) or ended."""
    id: str
    started_at: datetime
    ended_at: datetime | None
    wait_s: float | None
    ended_by: str | None        # chair | wash | door | lost | left | stall | restart (None while open)
    target_s: float
    camera: str
    track_id: int               # the track that started it
    last_track_id: int          # the track holding it last (reconnections)
    ticket: int
    last_seen_at: datetime      # refreshed while open; a restart closes the wait here


@dataclass(frozen=True)
class FaceCandidate:
    """One customer-face sample from the sidecar (iris:customer_face)."""

    camera: str
    track_id: int
    ts: float
    crop_path: str                     # relative to the crops root
    embedding: tuple[float, ...]       # unit-length ArcFace, 512-d
    quality: float
    # the face's own box (left, top, width, height) in the sidecar's capture px (= its mux px for barber, like
    # person boxes); None from an older sidecar or a recording without it (2026-10-15, final review I1)
    face_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BodyCandidate:
    camera: str
    track_id: int
    ts: float
    crop_path: str
    embedding: tuple[float, ...] | None   # the re-ID embedding of the same sample
    quality: float


@dataclass(frozen=True)
class Registration:
    """A visit with a haircut has ended: its best candidates, for register.apply."""

    visit_id: str
    ended_at: datetime
    faces: tuple[FaceCandidate, ...]
    bodies: tuple[BodyCandidate, ...]


@dataclass(frozen=True)
class PhotoRequest:
    """A frame wanted for a haircut's photo report (2026-10-22, spec haircut photos): its chair's camera, cut to
    `crop` (x0, y0, x1, y1 as fractions of the frame). `at`: the engine's time of the request (the rows' clock)."""
    haircut_id: str
    chair: str
    camera: str
    kind: str                                  # start | end
    at: datetime
    crop: tuple[float, float, float, float]


@dataclass
class PedicureRecord:
    """One pedicure session on a pedicure chair (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md): part of
    a customer's service after his haircut -- never a visit, a ticket or a served customer."""
    id: str
    chair_id: str
    started_at: datetime
    ended_at: datetime | None = None
    service_s: float | None = None   # sitting time (end gap excluded); None while open or interrupted
    open: bool = True


@dataclass
class EngineUpdate:
    """What one ShopEngine.observe() changed, for storage.apply_update."""

    visits: dict[str, Visit] = field(default_factory=dict)            # visit id -> confirmed visit
    sessions: dict[str, ServiceSession] = field(default_factory=dict)
    deleted_sessions: set[str] = field(default_factory=set)           # too short to be haircuts
    alerts: list[Alert] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    registrations: list[Registration] = field(default_factory=list)
    discard_crops: list[str] = field(default_factory=list)
    bindings: list[Binding] = field(default_factory=list)
    door_crossings: list[DoorCrossing] = field(default_factory=list)
    sittings: list[SittingRecord] = field(default_factory=list)
    waits: dict[str, WaitRecord] = field(default_factory=dict)        # wait id -> latest record
    resume_checks: list[ResumeCheckRecord] = field(default_factory=list)
    photos: list[PhotoRequest] = field(default_factory=list)          # haircut photos to take (photo_plan.py)
    dropped_photos: set[str] = field(default_factory=set)             # haircuts closed unconfirmed: their photos go
    pedicures: dict[str, PedicureRecord] = field(default_factory=dict)   # pedicure id -> latest record (2026-11-10)

    def __bool__(self) -> bool:
        return bool(self.visits or self.sessions or self.deleted_sessions or self.alerts
                    or self.events or self.handoffs or self.registrations or self.discard_crops
                    or self.bindings or self.door_crossings or self.sittings or self.waits or self.resume_checks
                    or self.photos or self.dropped_photos or self.pedicures)
