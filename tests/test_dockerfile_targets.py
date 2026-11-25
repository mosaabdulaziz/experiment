import pathlib
"""What each stage of the pipeline Dockerfile must (and must not) contain.

`slim` is what attendance, barber and fire run; `mtmc` adds torch for multi-camera Re-ID. Torch brings its
own cuDNN (nvidia-cudnn-cu13 9.24) next to the base image's 9.17; without the loader path below, ONNX Runtime
(face recognition) fails its conv engines and silently runs on the CPU, ~16x slower.
"""
import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "perception" / "pipeline" / "Dockerfile"


def stages():
    out, cur = {}, None
    for line in DOCKERFILE.read_text().splitlines():
        m = re.match(r"FROM\s+\S+(?:\s+AS\s+(\S+))?", line, re.I)
        if m:
            cur = m.group(1) or f"stage{len(out)}"
            out[cur] = []
        elif cur and not line.lstrip().startswith("#"):
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def test_the_default_last_stage_is_slim():
    last = list(stages())[-1]
    assert last == "slim", "an un-targeted build must give the slim image the other products run"


def test_slim_and_base_have_no_torch_or_vlm_clients():
    s = stages()
    for name in ("base", "slim"):
        assert not re.search(r"\btorch|ollama|fvcore|yacs", s[name]), f"{name} must stay free of the multicam dependencies"


def test_mtmc_installs_the_locks_mtmc_group_which_holds_cuda_torch_and_the_vlm_clients():
    body = stages()["mtmc"]
    assert re.search(r'MTMC_MODULES="[^"]*\bmtmc\b', body) and "install_locked.sh /req mtmc" in body
    reqs = (DOCKERFILE.parents[1] / "modules" / "mtmc" / "requirements.txt").read_text()
    assert "ollama" in reqs and "openai" in reqs and "fvcore" in reqs and "yacs" in reqs
    pyproject = (DOCKERFILE.parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^mtmc = \[.*"torch"', pyproject, re.M), "the lock's mtmc group must hold CUDA torch"


def test_base_installs_the_locks_slim_group_hash_checked():
    body = stages()["base"]
    assert "install_locked.sh /req slim" in body and "uv.lock" in body
    script = (DOCKERFILE.parent / "install_locked.sh").read_text()
    installs = [line for line in script.splitlines() if "uv pip install" in line and not line.lstrip().startswith("#")]
    assert "--frozen" in script and len(installs) == 3 and all("--require-hashes" in line for line in installs)
    # the GPU runtime and the GUI cv2 are installed last, in this order (they overwrite their CPU/headless twins)
    assert 'LATE="opencv-python onnxruntime-gpu"' in script


def test_the_cuda_index_of_the_install_is_the_locks():
    script = (DOCKERFILE.parent / "install_locked.sh").read_text()
    pyproject = (DOCKERFILE.parents[1] / "pyproject.toml").read_text()
    url = re.search(r"^CUDA_INDEX=(\S+)", script, re.M).group(1)
    assert f'url = "{url}"' in pyproject


# The two buffalo_l models face recognition loads, as insightface's v0.7 release has them (the image used to download
# and check exactly these; the repo's DVC copy must be the same bytes).
BUFFALO_L_SHA256 = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def test_the_base_image_pyds_and_buffalo_l_are_pinned():
    text = DOCKERFILE.read_text()
    assert re.search(r"^FROM nvcr\.io/nvidia/deepstream:[\w.-]+@sha256:[0-9a-f]{64} AS base$", text, re.M)
    assert re.search(r"^ARG PYDS_COMMIT=[0-9a-f]{40}$", text, re.M) and 'git checkout --quiet "$PYDS_COMMIT"' in text
    # buffalo_l is not in the image: the sidecar loads the repo's DVC copy (perception/insightface_pack.py), so the pin
    # lives in the import catalog, which tests/test_insightface_pack.py checks against the pulled files.
    import yaml
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "FaceAnalysis" not in body, "the sidecar image must not download buffalo_l"
    catalog = yaml.safe_load((DOCKERFILE.parents[2] / "mtmc" / "tracking" / "imported_models.yaml").read_text())
    (entry,) = [e for e in catalog["entries"] if e["id"] == "face-recognizer/buffalo_l"]
    pinned = {Path(f).name: h for f, h in entry["expected_sha256"].items()}
    assert {n: pinned.get(n) for n in BUFFALO_L_SHA256} == BUFFALO_L_SHA256


def test_mtmc_puts_torchs_cudnn_where_the_loader_can_complete_it():
    body = stages()["mtmc"]
    assert re.search(r"ENV\s+LD_LIBRARY_PATH=\S*site-packages/nvidia/cudnn/lib", body)
    # and it must come BEFORE the torch install is irrelevant (ENV), but the path must keep the base image's entries
    assert "${LD_LIBRARY_PATH}" in body


def test_mtmc_gives_the_cuda_wheels_time_to_download():
    assert re.search(r"ENV\s+UV_HTTP_TIMEOUT=\d{3,}", stages()["mtmc"])


def test_the_sidecar_image_copies_every_package_the_engine_loads():
    """The compose bind mount hides a missing COPY on a dev box; an image built without it (the scale and
    production composes) would fail at import. The entry point's packages must all be in the image."""
    text = (pathlib.Path(__file__).resolve().parents[1] / "perception" / "pipeline" / "Dockerfile").read_text()
    for package in ("perception/engine/", "perception/modules/", "perception/pipeline/"):
        assert f"COPY {package} ./{package}" in text, f"the sidecar image does not COPY {package}"
    assert "COPY perception/__init__.py ./perception/" in text


# --- a module-list --build-arg the lock cannot honour stops the build ---

INSTALL = DOCKERFILE.parent / "install_locked.sh"


def _install(tmp_path, group, arg, value):
    """Run install_locked.sh against the real perception/pyproject.toml, with a stand-in `uv` that exits 7: reaching it
    means the module-list check passed (nothing is installed)."""
    import os
    import subprocess
    (tmp_path / "uv").write_text("#!/bin/sh\nexit 7\n")
    (tmp_path / "uv").chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(["bash", str(INSTALL), str(DOCKERFILE.parents[1]), group, arg, value],
                          env=env, capture_output=True, text=True)


def test_each_install_passes_its_module_list_to_the_check():
    s = stages()
    assert 'install_locked.sh /req slim BASE_MODULES "$BASE_MODULES"' in s["base"]
    assert 'install_locked.sh /req mtmc MTMC_MODULES "$MTMC_MODULES"' in s["mtmc"]


def test_the_lock_records_the_module_lists_its_groups_were_generated_from():
    text, pyproject = DOCKERFILE.read_text(), (DOCKERFILE.parents[1] / "pyproject.toml").read_text()
    for group, arg in (("slim", "BASE_MODULES"), ("mtmc", "MTMC_MODULES")):
        default = re.search(rf'^ARG {arg}="([^"]*)"', text, re.M).group(1)
        assert f'# {group}: {arg}="{default}"\n' in pyproject


def test_the_dockerfile_defaults_pass_the_check(tmp_path):
    text = DOCKERFILE.read_text()
    for group, arg in (("slim", "BASE_MODULES"), ("mtmc", "MTMC_MODULES")):
        default = re.search(rf'^ARG {arg}="([^"]*)"', text, re.M).group(1)
        r = _install(tmp_path, group, arg, f"  {default}  ".replace(" ", "  "))   # extra whitespace is not a change
        assert r.returncode == 7, r.stderr


def test_a_different_module_list_stops_the_build_with_a_clear_message(tmp_path):
    for group, arg, value in (("slim", "BASE_MODULES", "face_recognition appearance"),
                              ("mtmc", "MTMC_MODULES", "mtmc extra_module"),
                              ("slim", "BASE_MODULES", "")):
        r = _install(tmp_path, group, arg, value)
        assert r.returncode == 1, (value, r.stderr)
        assert f'error: {arg}="{value}", but perception/uv.lock\'s \'{group}\' group was generated from' in r.stderr
        assert "just deps-lock" in r.stderr and f"--build-arg {arg}" in r.stderr
