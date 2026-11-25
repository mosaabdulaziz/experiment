"""Bridge settings from the environment. The vendored broker.py and messages.py read the same attribute names as the
shelf-demo service's Settings (amqp_url, amqp_user, target, inbound_queue, app_id, prefetch, reconnect_max_s,
safe_url, is_cloud, confidence_mode)."""
import os
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlsplit

CONFIDENCE_MODES = ("stability", "detector")


def _safe(url: str) -> str:
    """A URL without its password, for logs."""
    u = urlsplit(url)
    host = u.hostname or ""
    if u.port:
        host += f":{u.port}"
    user = f"{u.username}@" if u.username else ""
    return f"{u.scheme}://{user}{host}{u.path}"


@dataclass(frozen=True)
class Settings:
    # ---- iris-platform broker (iris-platform docs/ai-integration.md §3) ----
    amqp_url: str = ""               # amqp(s)://user:password@host:port/vhost -- a secret, never logged
    amqp_user: str = ""              # AMQP user_id property; must equal the connection user
    target: str = "cloud"            # "cloud" or "edge.<org-slug>"
    queue: str = ""                  # default from target: ai.cloud.inbound / edge.<slug>.inbound
    app_id: str = "shelf-iris-bridge"
    prefetch: int = 20
    reconnect_max_s: float = 30.0
    publish_catalog: bool = True     # cloud only; see shelf-demo GAPS.md "Who owns the cloud catalog?"
    confidence_mode: str = "stability"   # what goes in event.confidence (the other value is in metadata)
    # ---- this stack ----
    redis_url: str = "redis://redis:6379/0"
    mapping_file: str = "/config/shelf-iris-mapping.yaml"
    group: str = "shelf-iris-bridge"     # the bridge's OWN consumer group on the alert streams
    consumer: str = ""                   # default: the host name
    # ---- the shelf sidecar, whose alert streams are read (keys.py) ----
    sidecar_config: str = ""             # its preset yaml, the sidecar's own SIDECAR_CONFIG
    sidecar_role: Optional[str] = None   # IRIS_SIDECAR_ROLE as the sidecar gets it; None = unset (the yaml's)
    iris_root: str = ""                  # the checkout relative SIDECAR_CONFIG paths start from (default: cwd)

    def __post_init__(self):
        if self.confidence_mode not in CONFIDENCE_MODES:
            raise ValueError(f"CONFIDENCE_MODE must be one of {CONFIDENCE_MODES}, got {self.confidence_mode!r}")
        if self.target != "cloud" and not (self.target.startswith("edge.") and len(self.target) > 5):
            raise ValueError(f"TARGET must be 'cloud' or 'edge.<org-slug>', got {self.target!r}")
        if not self.group:
            raise ValueError("CONSUMER_GROUP must not be empty")

    @property
    def is_cloud(self) -> bool:
        return self.target == "cloud"

    @property
    def inbound_queue(self) -> str:
        if self.queue:
            return self.queue
        return "ai.cloud.inbound" if self.is_cloud else f"{self.target}.inbound"

    @property
    def safe_url(self) -> str:
        return _safe(self.amqp_url)

    @property
    def safe_redis_url(self) -> str:
        return _safe(self.redis_url)

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        url = env.get("AMQP_URL", "")
        return cls(amqp_url=url,
                   amqp_user=env.get("AMQP_USER") or unquote(urlsplit(url).username or ""),
                   target=env.get("TARGET", "cloud").strip(),
                   queue=env.get("AMQP_QUEUE", "").strip(),
                   app_id=env.get("APP_ID", "shelf-iris-bridge"),
                   prefetch=int(env.get("AMQP_PREFETCH", 20)),
                   reconnect_max_s=float(env.get("RECONNECT_MAX_SECONDS", 30)),
                   publish_catalog=env.get("PUBLISH_CATALOG", "true").strip().lower() not in ("0", "false", "no"),
                   confidence_mode=env.get("CONFIDENCE_MODE", "stability").strip().lower(),
                   redis_url=env.get("REDIS_URL", cls.redis_url),
                   mapping_file=env.get("MAPPING_FILE", cls.mapping_file),
                   group=env.get("CONSUMER_GROUP", cls.group).strip(),
                   consumer=env.get("CONSUMER_NAME", "").strip() or socket.gethostname(),
                   sidecar_config=env.get("SIDECAR_CONFIG", "").strip(),
                   sidecar_role=env.get("IRIS_SIDECAR_ROLE"),
                   iris_root=env.get("IRIS_ROOT", "").strip())
