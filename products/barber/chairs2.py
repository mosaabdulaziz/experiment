"""Chair states for the chair-first engine (project 3, spec 2b).

empty -> starting: an occupied frame. starting -> sitting: occupied in >= ce_start_frac of the
frames of the ce_start_window_s window that begins at the first occupied frame (the start is
backdated to it); otherwise the window slides to the next occupied frame, or the chair is empty
again. sitting: evidence = occupied or staff cover; after ce_end_gap_s without evidence the sitting
ends, dated to the last OCCUPIED frame (cover never moves the end; with ce_carry_max_s > 0, cover counts only while
ts - last occupied <= that cap). After more than ce_refresh_after_s without an occupied frame, only the
ce_refresh_run_n-th consecutive occupied frame counts as occupied again (a lone one is a stray). Frames of one chair arrive
from one camera; a gap between them longer than ce_dropout_s (camera dropout) pauses the clock.
Start chain (ce_start_chain_s > 0; 2026-10-05, dev/chairrun/accuracy): while the chair is empty or starting,
occupied frames less than ce_start_chain_s apart form a chain; a start confirmed by the window rule is
backdated to the chain's first frame (a return half-hidden by the barber flickers empty <-> starting first),
and while the chain is live (chain_live) the chair counts as taken for the haircuts' resume window. A chain
older than ce_start_chain_max_s (its first frame to now) is dead: it neither counts as live nor backdates, and stays
dead until a gap of ce_start_chain_s breaks it (stray frames < ce_start_chain_s apart can't hold a haircut forever).
Gap and age are plain frame time: a camera dropout neither breaks nor pauses a chain. A backdated start counts the
chain time as sitting time (ce_min_haircut_s too); the age cap bounds that.
Per-chair overrides (policy "<key>@<chair>", 2026-10-05): step() runs each chair on policy_for(policy, chair), so
ce_start_window_s, ce_start_frac, ce_end_gap_s and (2026-10-18) the cover cap ce_carry_max_s may differ per chair.
Reseat (2026-10-18, ce_reseat_check_s > 0): sitting, an occupied frame that counts after at least that long without one
(cover held the sitting; the clock is the dropout-paused one) is a Reseat -- the haircuts ask re-ID whether it is him.
Cover share (2026-11-13, ce_cover_min_frac > 0, per chair): a cover frame is evidence only while cover fills at least
that share of the frames of the last ce_cover_window_s -- a barber at the chair, not someone brushing its zone now and
then (Palm Hills 6 Nov, L7 21:08: such flickers held an empty seat 100 s, into the next customer's sitting)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from policy import policy_for
from seat import ChairSignal

EMPTY, STARTING, SITTING = "empty", "starting", "sitting"


@dataclass(frozen=True)
class SitStart:
    chair: str
    ts: float
    track_id: int | None


@dataclass(frozen=True)
class SitEnd:
    chair: str
    ts: float


@dataclass(frozen=True)
class Reseat:
    """Sitting: the seat taken again at ts after a stretch with no one seen in it since last_occ."""
    chair: str
    ts: float
    track_id: int | None
    last_occ: float


@dataclass
class _Chair:
    state: str = EMPTY
    window: deque = field(default_factory=deque)      # (ts, occupied, track_id) while starting
    start_ts: float | None = None
    last_occ: float | None = None
    last_evidence: float | None = None
    cap_ref: float | None = None      # last_occ shifted by camera dropouts: the cap's clock only
    last_frame: float | None = None
    run: int = 0                      # consecutive occupied frames
    chain_first: float | None = None  # start chain: first / last occupied frame while empty or starting
    chain_last: float | None = None
    cover_win: deque = field(default_factory=deque)   # (ts, cover) of the sitting's latest frames (ce_cover_min_frac)


class ChairStates:
    def __init__(self, chairs: Iterable[str]):
        self._c: dict[str, _Chair] = {}
        self.set_chairs(chairs)

    def set_chairs(self, chairs: Iterable[str]) -> None:
        self._c = {ch: self._c.get(ch, _Chair()) for ch in chairs}

    def state(self, chair: str) -> str:
        return self._c[chair].state

    def since(self, chair: str) -> float | None:
        """Sitting: the sitting's start. Starting: its first occupied frame (2026-10-07, review finding 5: the
        dashboard shows "sitting down" only once this is a few seconds old, and got None -- never showed it)."""
        c = self._c[chair]
        if c.state == SITTING:
            return c.start_ts
        return c.window[0][0] if c.state == STARTING and c.window else None

    def occupied_until(self, chair: str) -> float | None:
        """The running sitting's last occupied frame (None: no sitting running)."""
        c = self._c[chair]
        return c.last_occ if c.state == SITTING else None

    def chain_live(self, chair: str, now_ts: float, policy: dict) -> bool:
        """A start chain is running: its last occupied frame is at most ce_start_chain_s old."""
        c = self._c[chair]
        gap = policy["ce_start_chain_s"]
        return (gap > 0 and c.chain_last is not None and now_ts - c.chain_last <= gap
                and now_ts - c.chain_first <= policy["ce_start_chain_max_s"])

    def no_view(self, chair: str, now_ts: float, policy: dict) -> bool:
        c = self._c[chair]
        return c.last_frame is not None and now_ts - c.last_frame > policy["ce_dropout_s"]

    def step(self, chair: str, ts: float, sig: ChairSignal, policy: dict) -> list:
        c = self._c.get(chair)
        if c is None:
            return []
        policy = policy_for(policy, chair)
        dropout_detected = c.last_frame is not None and ts - c.last_frame > policy["ce_dropout_s"]
        if dropout_detected:
            if c.state == STARTING:
                c.state, c.window = EMPTY, deque()
            elif c.last_evidence is not None:
                c.last_evidence += ts - c.last_frame          # dropout: pause the evidence clock
                if c.cap_ref is not None:
                    c.cap_ref += ts - c.last_frame
        c.last_frame = ts
        c.run = c.run + 1 if sig.occupied else 0
        out: list = []
        gap = policy["ce_start_chain_s"]
        if gap > 0 and sig.occupied and c.state in (EMPTY, STARTING):
            if c.chain_last is None or ts - c.chain_last > gap:
                c.chain_first = ts
            c.chain_last = ts
        if c.state == EMPTY:
            if sig.occupied:
                c.state, c.window = STARTING, deque([(ts, True, sig.track_id)])
            return out
        if c.state == STARTING:
            c.window.append((ts, sig.occupied, sig.track_id))
            out += self._try_start(chair, c, ts, policy)
            if out and gap > 0:
                if (c.chain_first is not None and c.chain_first < c.start_ts
                        and ts - c.chain_first <= policy["ce_start_chain_max_s"]):
                    c.start_ts = c.chain_first
                    out = [SitStart(chair, c.chain_first, ev.track_id) for ev in out]
                c.chain_first = c.chain_last = None
            return out
        # SITTING
        if policy["ce_cover_min_frac"] > 0:
            c.cover_win.append((ts, bool(sig.cover)))
            while c.cover_win[0][0] <= ts - policy["ce_cover_window_s"]:
                c.cover_win.popleft()
        # 2026-10-04 (Task 16c): a lone occupied frame after a long gap is a stray (a barber's track with a
        # customer's staff_p after an id switch, or none yet); it must not move the end or restart the cap.
        if sig.occupied and (c.run >= policy["ce_refresh_run_n"] or ts - c.cap_ref <= policy["ce_refresh_after_s"]):
            if policy["ce_reseat_check_s"] > 0 and ts - c.cap_ref >= policy["ce_reseat_check_s"]:
                out.append(Reseat(chair, ts, sig.track_id, c.last_occ))
            c.last_occ = c.last_evidence = c.cap_ref = ts
        elif (sig.cover and self._steady_cover(c, ts, sig, policy)
              and not (policy["ce_carry_max_s"] > 0 and ts - c.cap_ref > policy["ce_carry_max_s"])):
            c.last_evidence = ts
        if ts - c.last_evidence > policy["ce_end_gap_s"]:
            out.append(SitEnd(chair, c.last_occ))
            c.state, c.start_ts, c.last_occ, c.last_evidence, c.cap_ref = EMPTY, None, None, None, None
            c.cover_win.clear()
        return out

    @staticmethod
    def _steady_cover(c: _Chair, ts: float, sig: ChairSignal, policy: dict) -> bool:
        """ce_cover_min_frac: cover in at least that share of the last ce_cover_window_s of frames (0 = any frame)."""
        need = policy["ce_cover_min_frac"]
        if need <= 0:
            return True
        return sum(1 for _, cov in c.cover_win if cov) >= need * len(c.cover_win)

    def _try_start(self, chair: str, c: _Chair, ts: float, policy: dict) -> list:
        k, frac = policy["ce_start_window_s"], policy["ce_start_frac"]
        while c.window and ts - c.window[0][0] >= k:
            first = c.window[0][0]
            inside = [w for w in c.window if w[0] < first + k] or [c.window[0]]   # k 0: never empty
            if sum(1 for w in inside if w[1]) >= frac * len(inside):
                occ = [w for w in c.window if w[1]]
                c.state, c.start_ts = SITTING, first
                c.last_occ = c.last_evidence = c.cap_ref = occ[-1][0]
                c.window = deque()
                return [SitStart(chair, first, inside[0][2])]
            c.window.popleft()
            while c.window and not c.window[0][1]:
                c.window.popleft()
        if not c.window:
            c.state = EMPTY
        return []
