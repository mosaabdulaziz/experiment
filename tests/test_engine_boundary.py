"""The DeepStream engine knows no use case: perception/engine/ must not import use-case code and must not name a
product or use case. Use-case logic lives in perception/modules/ and is reached only through the module hook API.

The engine was cut out of one file that held everything, so it still contains use-case code. This is a ratchet:
tests/engine_boundary_ratchet.json records how many violations each file has today. A file may never gain a
violation, and when it loses one the recorded number must be lowered to match (so the gain cannot be undone).
The target is an empty ratchet. `python3 tests/test_engine_boundary.py --write` rewrites it from the current code.
"""
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "perception" / "engine"
RATCHET = pathlib.Path(__file__).with_name("engine_boundary_ratchet.json")

FORBIDDEN_IMPORTS = ("perception.modules", "products", "app", "mtmc")
# Helper modules that still sit flat in perception/pipeline/ and belong to a use case; the engine imports them by
# bare name today. Each leaves this list when it moves into perception/modules/.
USE_CASE_FLAT_MODULES = (
    "fire_alerts", "zone_dedup", "zone_config", "overcrowding_dedup", "box_color", "face_owner", "face_recognizer",
    "face_enhancer", "face_sample_gate", "face_regions", "plate_hold", "plate_reader", "appearance", "customer_faces",
    "vehicle_gallery", "vlm_arbiter", "mtmc", "mtmc_palette", "sgie_gallery", "box_stabilizer", "mask_outline",
)
FORBIDDEN_WORDS = ("attendance", "barber", "multicam", "fire", "smoke", "plate", "plates", "overcrowding")
_WORDS = re.compile(r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b", re.IGNORECASE)
# What objects are called is the detector's and the modules' business: the engine passes labels through and never
# compares one, so a quoted label in it is use-case knowledge leaking in (modules agree on these among themselves).
FORBIDDEN_LABELS = ("person", "persons", "people", "face", "faces", "vehicle", "car", "truck", "bus", "plate", "fire",
                    "smoke")
_LABELS = re.compile(r"[\"'](" + "|".join(FORBIDDEN_LABELS) + r")[\"']")


def _engine_files():
    return sorted(ENGINE.rglob("*.py")) if ENGINE.is_dir() else []


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or ""


def _is_use_case_import(mod: str) -> bool:
    return any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_IMPORTS + USE_CASE_FLAT_MODULES)


def violations() -> dict[str, dict[str, int]]:
    out = {}
    for path in _engine_files():
        text = path.read_text(encoding="utf-8")
        imports = sum(_is_use_case_import(m) for m in _imported_modules(ast.parse(text)))
        names = sum(bool(_WORDS.search(line) or _LABELS.search(line)) for line in text.splitlines())
        if imports or names:
            out[str(path.relative_to(ROOT))] = {"imports": imports, "names": names}
    return out


def _check(kind: str):
    current = violations()
    recorded = json.loads(RATCHET.read_text()) if RATCHET.exists() else {}
    worse, better = [], []
    for f in sorted(set(current) | set(recorded)):
        now = current.get(f, {}).get(kind, 0)
        was = recorded.get(f, {}).get(kind, 0)
        if now > was:
            worse.append(f"{f}: {now} {kind} violations, ratchet allows {was}")
        elif now < was:
            better.append(f"{f}: down to {now} (ratchet says {was}) -- run: python3 tests/test_engine_boundary.py --write")
    assert not worse, "use-case code added to the engine:\n" + "\n".join(worse)
    assert not better, "progress not locked in:\n" + "\n".join(better)


def test_engine_imports_no_use_case_code():
    _check("imports")


def test_engine_names_no_product():
    _check("names")


if __name__ == "__main__" and "--write" in sys.argv:
    RATCHET.write_text(json.dumps(violations(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {RATCHET}")
