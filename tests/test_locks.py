"""The locks (DEPENDENCIES.md): each was made from its pyproject.toml's current groups and every file it names carries a
hash. The sidecar's groups are what the requirements.txt files and the Dockerfile's module lists say; the root's
requirements*.txt are exports of its lock. A requirements file or group edited without `just deps-lock` fails here,
before an image or a CI environment is built from a stale lock."""
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROJECTS = {"perception": REPO / "perception", "root": REPO, "onnx-export": REPO / "tools" / "onnx-export",
            "mtmc-research": REPO / "tools" / "mtmc-research", "attendance": REPO / "products" / "attendance",
            "barber": REPO / "products" / "barber", "fall_detection": REPO / "products" / "fall_detection",
            "mlflow-server": REPO / "tools" / "mlflow-server"}


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO / "tools" / "deps" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toml(path):
    return tomllib.loads(path.read_text())


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement(text):
    """'onnxruntime-gpu>=1.19' -> ('onnxruntime-gpu', '>=1.19'): enough for the plain lines the groups hold."""
    m = re.match(r"\s*([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(.*)", text)
    return _norm(m.group(1)), m.group(2).replace(" ", "")


def _flatten(groups, name):
    out = set()
    for item in groups[name]:
        out |= _flatten(groups, item["include-group"]) if isinstance(item, dict) else {_requirement(item)}
    return out


@pytest.mark.parametrize("project", PROJECTS)
def test_the_lock_was_made_from_the_current_dependencies_and_groups(project):
    pyproject = _toml(PROJECTS[project] / "pyproject.toml")
    groups = pyproject.get("dependency-groups", {})
    lock = _toml(PROJECTS[project] / "uv.lock")
    (root,) = [p for p in lock["package"] if "." in (p["source"].get("virtual"), p["source"].get("editable"))]
    deps = {(_norm(r["name"]), r.get("specifier", "")) for r in root.get("metadata", {}).get("requires-dist", [])}
    assert deps == {_requirement(r) for r in pyproject["project"]["dependencies"]}, \
        f"{project}: uv.lock is out of date: run `just deps-lock`"
    recorded = root.get("metadata", {}).get("requires-dev", {})
    assert set(recorded) == {_norm(g) for g in groups}, f"{project}: uv.lock is out of date: run `just deps-lock`"
    for g in groups:
        locked = {(_norm(r["name"]), r.get("specifier", "")) for r in recorded[_norm(g)]}
        assert locked == _flatten(groups, g), f"{project}: uv.lock's group {g!r} is out of date: run `just deps-lock`"


@pytest.mark.parametrize("project", PROJECTS)
def test_every_locked_file_has_a_hash(project):
    for p in _toml(PROJECTS[project] / "uv.lock")["package"]:
        if "registry" in p["source"]:
            files = p.get("wheels", []) + ([p["sdist"]] if "sdist" in p else [])
            assert files and all(f.get("hash", "").startswith("sha256:") for f in files), f"{project}: {p['name']}"


@pytest.mark.parametrize("project", PROJECTS)
def test_one_uv_minor_everywhere(project):
    """Any uv 0.13.x (one lock format); the Dockerfiles and CI pin exactly 0.13.6."""
    assert _toml(PROJECTS[project] / "pyproject.toml")["tool"]["uv"]["required-version"] == ">=0.13.6,<0.14"


@pytest.mark.parametrize("project, index", [("perception", "https://download.pytorch.org/whl/cu130"),
                                            ("mtmc-research", "https://download.pytorch.org/whl/cu130"),
                                            ("root", "https://download.pytorch.org/whl/cpu")])
def test_torch_comes_only_from_its_index(project, index):
    for p in _toml(PROJECTS[project] / "uv.lock")["package"]:
        if p["name"] in ("torch", "torchvision"):
            assert p["source"]["registry"] == index, f"{project}: {p['name']} {p['version']}"


# --- the sidecar: groups generated from the requirements files ---

def test_the_sidecar_groups_are_generated_from_the_requirements_files_and_the_dockerfile():
    mod = load("perception_groups")
    text, i, j = mod.current()
    assert text[i:j] == mod.block(), "perception/pyproject.toml is out of date with the requirements.txt files: run `just deps-lock`"


def test_slim_is_the_engine_and_base_modules_and_mtmc_adds_cuda_torch():
    groups = _toml(PROJECTS["perception"] / "pyproject.toml")["dependency-groups"]
    slim, mtmc = {n for n, _ in _flatten(groups, "slim")}, {n for n, _ in _flatten(groups, "mtmc")}
    assert {"redis", "insightface", "onnxruntime-gpu"} <= slim and not {"torch", "torchvision", "ollama"} & slim
    assert slim < mtmc and {"torch", "torchvision", "ollama", "fvcore"} <= mtmc
    for p in _toml(PROJECTS["perception"] / "uv.lock")["package"]:
        if p["name"] in ("torch", "torchvision"):
            assert p["version"].endswith("+cu130")


# --- the root: requirements*.txt are exports of the lock ---

def test_the_requirements_files_are_exports_of_the_root_lock():
    mod = load("root_exports")
    groups = _toml(REPO / "pyproject.toml")["dependency-groups"]
    versions = mod.locked_versions()
    for path, group in mod.EXPORTS.items():
        assert (REPO / path).read_text() == mod.render(group, groups, versions), f"{path} is stale: run `just deps-lock`"


def test_the_test_group_stays_light_and_the_app_image_holds_the_app():
    groups = _toml(REPO / "pyproject.toml")["dependency-groups"]
    test = {n for n, _ in _flatten(groups, "test")}
    assert not {"ultralytics", "torch", "torchvision", "insightface"} & test, "the test group must stay CI-light"
    assert _flatten(groups, "app") <= _flatten(groups, "app-image")
    assert {"torch", "insightface", "opencv-python-headless"} <= {n for n, _ in _flatten(groups, "app-image")}


def _app_dockerfile_install(marker):
    """The app Dockerfile's `uv pip install` line that contains `marker`."""
    lines = [l for l in (REPO / "Dockerfile").read_text().splitlines()
             if "uv pip install" in l and marker in l and not l.lstrip().startswith("#")]
    assert len(lines) == 1, f"the app Dockerfile has {len(lines)} `uv pip install` lines naming {marker!r}"
    return lines[0]


def _assert_pinned_to_the_root_lock(line, names):
    locked = {}
    for p in _toml(REPO / "uv.lock")["package"]:
        if p["name"] in names:
            locked.setdefault(p["name"], set()).add(p["version"].split("+")[0])
    assert set(locked) == set(names), f"uv.lock has no {sorted(set(names) - set(locked))}"
    for name, versions in locked.items():
        assert len(versions) == 1, f"uv.lock has several versions of {name}: {sorted(versions)}"
        (version,) = versions
        assert f" {name}=={version} " in f" {line} ", f"the app Dockerfile's {name} pin is not the lock's {version}"


def test_the_app_dockerfile_installs_the_locked_cpu_torch():
    """The app image installs CPU torch before requirements.txt (which leaves it out); its pin must be the lock's."""
    _assert_pinned_to_the_root_lock(_app_dockerfile_install("download.pytorch.org/whl/cpu"), ("torch", "torchvision"))


def test_the_app_dockerfile_installs_the_locked_insightface_and_onnxruntime():
    """InsightFace and its CPU runtime are installed after requirements.txt (which leaves them out); pinned to the lock."""
    _assert_pinned_to_the_root_lock(_app_dockerfile_install("insightface"), ("insightface", "onnxruntime"))


def test_the_mtmc_research_tools_share_the_sidecars_cuda_torch():
    """Re-ID runs on the GPU: the research tools' torch is the sidecar's, not the app's CPU build."""
    def torch_of(project):
        return {p["name"]: p["version"] for p in _toml(PROJECTS[project] / "uv.lock")["package"]
                if p["name"] in ("torch", "torchvision")}
    assert torch_of("mtmc-research") == torch_of("perception")
