"""Prometheus text exposition for the app's own health numbers (no client library: the iris
image does not carry one). Samples are (metric name, labels, value)."""


def _esc(v: object) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(samples: list[tuple[str, dict, float]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for name, labels, value in samples:
        if name not in seen:
            seen.add(name)
            lines.append(f"# TYPE {name} {'counter' if name.endswith('_total') else 'gauge'}")
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in sorted(labels.items()))
        lines.append(f"{name}{{{lab}}} {float(value):g}" if lab else f"{name} {float(value):g}")
    return "\n".join(lines) + "\n"
