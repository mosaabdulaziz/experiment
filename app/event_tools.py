import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def resolve_relative(value: str | None) -> str | None:
    """Map relative keywords (today/24h/7d) to UTC ISO timestamps; pass ISO through."""
    if value is None:
        return None
    v = value.strip().lower()
    now = datetime.now(timezone.utc)
    if v == "today":
        return now.strftime("%Y-%m-%dT00:00:00Z")
    if v == "24h":
        return datetime.fromtimestamp(now.timestamp() - 24 * 3600, tz=timezone.utc).strftime(_FMT)
    if v == "7d":
        return datetime.fromtimestamp(now.timestamp() - 7 * 86400, tz=timezone.utc).strftime(_FMT)
    return value


QUERY_EVENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "query_events",
        "description": (
            "Query the surveillance event database for past alerts/comments. Use this "
            "to answer questions about counts, history, or specific incidents beyond the "
            "recent log already shown. Returns a total match count plus matching events. "
            "since/until accept an ISO-8601 UTC timestamp OR the keywords 'today', '24h', '7d'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "description": "Camera/stream id, e.g. 'cam1'."},
                "event_type": {"type": "string", "enum": ["alert", "comment"]},
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "Match events having ANY of these tags."},
                "since": {"type": "string", "description": "Lower time bound (ISO or today/24h/7d)."},
                "until": {"type": "string", "description": "Upper time bound (ISO or today/24h/7d)."},
                "limit": {"type": "integer", "description": "Max events to return (default 20)."},
            },
            "required": [],
        },
    },
}


async def dispatch_tool_call(store, name: str, arguments: dict) -> str:
    """Execute a tool call, returning a compact JSON string. Never raises into
    the agentic loop — bad input returns a structured {"error": ...} string."""
    if name != "query_events":
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        args = arguments or {}
        limit = args.get("limit", 20)
        limit = int(limit)
        result = await store.query(
            camera=args.get("camera"),
            event_type=args.get("event_type"),
            tags=args.get("tags"),
            since=resolve_relative(args.get("since")),
            until=resolve_relative(args.get("until")),
            limit=limit,
        )
        return json.dumps(result)
    except Exception as exc:  # structured error, never propagate
        logger.warning("query_events dispatch failed: %s", exc)
        return json.dumps({"error": str(exc)})
