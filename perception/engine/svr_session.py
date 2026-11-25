"""Per-source FIFO correlating an SVR session's start to its `sr-done` completion.

`_newest_svr_file` (sidecar.py) used to resolve `sr-done` by scanning the SVR
directory for the newest file by mtime, on the theory that the smart-rec-status
guard allows only one session per source at a time, so an `sr-done` can only refer
to the session started on that source. That guard IS unambiguous for the decision
to *emit* `start-sr` (sidecar.py's `_SVR_EMIT_LOCK`) -- but not for the on-disk
write race: an alert incident that stays open re-arms SVR recording the
instant the previous session's `smart-rec-status` flips false, so a brand-new
session's file can start existing at essentially the same instant the
just-finished session's file gets its last write. An mtime comparison can then
match the *completion* event to the *wrong* (just-started, ~1s-of-content) file --
this is the traced cause of alert clips that are anomalously ~1 second long.

Sessions on one source are still strictly serialized by that same guard, so
completions arrive in the same order sessions started -- a FIFO is therefore
always the right structure, and recording each session's file at START time
(before the completion-time race window ever opens) makes the correlation exact
instead of guessed.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU, same
convention as svr_trigger.py and overcrowding_dedup.py.
"""

from collections import deque


class SvrFileTracker:
    """One instance per source. Records which file each SVR session got at start
    time; resolves each `sr-done` from the oldest still-unresolved one."""

    def __init__(self):
        self._pending = deque()

    def session_started(self, before_names, after_names) -> list:
        """Record newly-appeared prefix-matching name(s) as the just-opened
        session's file(s) -- the set difference between a directory listing taken
        immediately before `start-sr` was emitted and one taken after the file
        actually appears on disk. Returns what was recorded, for logging."""
        new_names = sorted(set(after_names) - set(before_names))
        self._pending.extend(new_names)
        return new_names

    def session_done(self):
        """Pop the oldest unresolved session's filename, or None if nothing is
        tracked (the caller should fall back to a diagnosed heuristic and log
        loudly -- this should not happen in normal operation)."""
        return self._pending.popleft() if self._pending else None

    def pending_count(self) -> int:
        return len(self._pending)
