"""Camera layout as groups and links, expanded into what the matcher needs.

A GROUP is a set of cameras that see the same area (overlapping fields of view): a vehicle can be
on two of them at once, so within a group any camera may follow any other with no travel time, and
a vehicle that is still ACTIVE on one may be the one a neighbour has just picked up.

A LINK joins two groups that are next to each other, with the minimum number of seconds a vehicle
needs to get from one to the other. Links work both ways (vehicles drive in and out through the same
route). Groups that are not linked are not adjacent: no vehicle goes straight between them.

Pure Python (no numpy / torch) so it can be imported and tested anywhere.
"""
from __future__ import annotations

from typing import Any


def expand_camera_groups(
    groups: Any, links: Any,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], float], set[tuple[str, str]]]:
    """-> (topology {cam: allowed next cams}, gaps {(src, dst): min seconds}, overlap pairs {(src, dst)}).

    groups: {name: [cam, ...]} (or a list of lists, named by index). links: [[group, group, seconds], ...].
    """
    if isinstance(groups, dict):
        named = {str(k): [str(c) for c in v] for k, v in groups.items()}
    else:
        named = {str(i): [str(c) for c in v] for i, v in enumerate(groups or [])}
    seen: dict[str, str] = {}
    for name, cams in named.items():
        for cam in cams:
            if cam in seen:
                raise ValueError(f"camera {cam} is in two groups ({seen[cam]}, {name})")
            seen[cam] = name
    topology: dict[str, set[str]] = {cam: set() for cams in named.values() for cam in cams}
    gaps: dict[tuple[str, str], float] = {}
    overlap: set[tuple[str, str]] = set()
    for cams in named.values():
        for a in cams:
            for b in cams:
                if a != b:
                    topology[a].add(b)
                    overlap.add((a, b))
                    gaps[(a, b)] = 0.0
    for link in links or []:
        if len(link) != 3:
            raise ValueError(f"a link is [group, group, seconds], got {link!r}")
        g1, g2, seconds = str(link[0]), str(link[1]), float(link[2])
        for g in (g1, g2):
            if g not in named:
                raise ValueError(f"link names an unknown group {g!r} (known: {sorted(named)})")
        for a in named[g1]:
            for b in named[g2]:
                topology[a].add(b)
                topology[b].add(a)
                gaps[(a, b)] = seconds
                gaps[(b, a)] = seconds
    return topology, gaps, overlap
