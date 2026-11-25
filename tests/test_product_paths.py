"""Paths inside products/<name>/ that were written for the old one-level-deep folders.

`attendance-service/` became `products/attendance/`, one level deeper. Code that said
`Path(__file__).parent.parent / "data"` then pointed at `products/data/`: the stream
start times were never found (every replayed camera skipped, so no attendance rows),
photos and uploads went to the wrong folder, and the face-registration `docker run`
bind-mounted a `sidecar/` folder that no longer exists.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def product_sources():
    for path in (REPO / "products").rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        yield path


def test_no_product_climbs_to_the_repo_root_by_counting_parents():
    offenders = [str(p.relative_to(REPO)) for p in product_sources()
                 if re.search(r"Path\(__file__\)(\.resolve\(\))?\.parent\.parent\b", p.read_text())]
    assert not offenders, ("use .resolve().parents[2] for the repo root from products/<name>/: "
                           + ", ".join(offenders))


def test_no_product_names_the_repo_root_through_its_own_folder_variable():
    """`REPO_DIR = SERVICE_DIR.parent` was the same mistake spelled with a variable: barber then looked for
    data/stream_starts.json in products/, found no camera streaming, and never processed a single frame."""
    offenders = [str(p.relative_to(REPO)) for p in product_sources()
                 if re.search(r"^\s*REPO\w*\s*=\s*\w*DIR\.parent\s*$", p.read_text(), re.M)]
    assert not offenders, "REPO_DIR must be two levels above products/<name>/: " + ", ".join(offenders)


def test_no_product_mounts_the_old_sidecar_folder():
    offenders = [str(p.relative_to(REPO)) for p in product_sources()
                 if "/workspace/iris/sidecar" in p.read_text()]
    assert not offenders, "the pipeline now lives in perception/pipeline/: " + ", ".join(offenders)


def test_the_stream_start_file_is_the_one_the_streamer_writes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("att_cameras", REPO / "products/attendance/cameras.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._STREAM_STARTS_FILE == REPO / "data" / "stream_starts.json"
    streamer = (REPO / "test_streamer" / "stream_recordings.py").read_text()
    assert 'os.path.join(REPO, "data", "stream_starts.json")' in streamer
