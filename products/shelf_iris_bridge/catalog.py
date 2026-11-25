# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/catalog.py. Copied unchanged below this header: fix it there first, then copy it again.
"""The catalog we publish to IRIS, and parsing of a pipeline's `config` against it.

IRIS builds the pipeline form from `params` and sends the values back in `pipeline.upsert`
as `config` (guide §5.2, §6.1). Missing optional params mean "use your default".
"""
from dataclasses import dataclass

PIPELINE_ID = "shelf-void"
EVENT_TYPE = "shelf.empty_spot"

PARAMS = [
    {"key": "confirmSeconds", "type": "number",
     "label": {"en": "Confirm after (seconds)", "ar": "التأكيد بعد (ثانية)"},
     "default": 2, "min": 0.5, "max": 600},
    {"key": "clearSeconds", "type": "number",
     "label": {"en": "Clear after (seconds without empty spots)",
               "ar": "إنهاء التنبيه بعد (ثوانٍ بدون فراغات)"},
     "default": 30, "min": 5, "max": 3600},
    {"key": "confidence", "type": "number",
     "label": {"en": "Minimum detection confidence (0-1)", "ar": "أدنى ثقة للكشف (0-1)"},
     "default": 0.08, "min": 0.01, "max": 1},
    {"key": "pauseOnPerson", "type": "boolean",
     "label": {"en": "Pause while a person blocks the shelf", "ar": "إيقاف مؤقت عند حجب شخص للرف"},
     "default": True},
]

CATALOG_ENTRY = {
    "id": PIPELINE_ID,
    "version": "0.1.0",
    "category": "retail",
    "modes": ["cloud", "edge"],
    "name": {"en": "Shelf Empty Spot Detection", "ar": "كشف الفراغات على الرفوف"},
    "description": {
        "en": "Detects empty spots on shelves inside the camera's zone. Ignores people blocking "
              "the shelf and only reports gaps that stay in place.",
        "ar": "يكتشف الأماكن الفارغة على الرفوف داخل منطقة الكاميرا، ويتجاهل الأشخاص الذين يحجبون الرف، "
              "ولا ينبه إلا عن الفراغات الثابتة.",
    },
    "eventTypes": [
        {"key": EVENT_TYPE, "label": {"en": "Empty spot on shelf", "ar": "مكان فارغ على الرف"},
         "defaultSeverity": "medium"},
    ],
    "params": PARAMS,
}


def catalog_snapshot():
    """Body of `catalog.snapshot`. Must list everything this backend offers (GAPS §3)."""
    return {"pipelines": [CATALOG_ENTRY]}


@dataclass(frozen=True)
class PipelineConfig:
    confirm_seconds: float = 2.0
    clear_seconds: float = 30.0
    confidence: float = 0.08
    pause_on_person: bool = True


_FIELDS = {"confirmSeconds": "confirm_seconds", "clearSeconds": "clear_seconds",
           "confidence": "confidence", "pauseOnPerson": "pause_on_person"}


def parse_config(raw):
    """Turn an upsert's `config` into a PipelineConfig.

    Never fails: a missing value takes the default; a wrong type takes the default and an
    out-of-range number is clamped, each with a warning (for pipeline.status detail / logs).
    Unknown keys are ignored (guide §11). Returns (config, warnings).
    """
    raw = raw if isinstance(raw, dict) else {}
    values, warnings = {}, []
    for p in PARAMS:
        key, default = p["key"], p["default"]
        v = raw.get(key)
        if v is None:
            v = default
        elif p["type"] == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:  # bool, str, NaN
                warnings.append(f"{key}: {v!r} is not a number, using {default}")
                v = default
            elif not p["min"] <= v <= p["max"]:
                clamped = min(max(v, p["min"]), p["max"])
                warnings.append(f"{key}: {v} outside {p['min']}-{p['max']}, using {clamped}")
                v = clamped
            v = float(v)
        elif p["type"] == "boolean" and not isinstance(v, bool):
            warnings.append(f"{key}: {v!r} is not true/false, using {default}")
            v = default
        values[_FIELDS[key]] = v
    return PipelineConfig(**values), warnings
