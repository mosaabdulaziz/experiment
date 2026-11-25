"""Customer photos are served to the shop's own network only (no login yet --
required before any real deployment)."""
from __future__ import annotations

import ipaddress


def is_local_client(host: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local
