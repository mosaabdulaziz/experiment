"""Onboarding a module must not need a Dockerfile edit: what a module imports is declared in a requirements.txt (its own,
or the engine's), and every parser directory builds through the one generic script."""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ROOT / "perception" / "modules"

# import name -> the distribution that provides it
DISTRIBUTION = {"onnxruntime": "onnxruntime-gpu", "cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn"}
# imports the image provides without a requirements line: numpy arrives with opencv; Pillow with torchvision; torch and
# torchvision are installed from NVIDIA's CUDA index by the Dockerfile's mtmc target; the gfpgan enhancer backend is
# opt-in (face_enhancement_backend: gfpgan) and is not shipped
PROVIDED = {"numpy", "PIL", "torch", "torchvision", "gfpgan"}


def _declared(path: pathlib.Path) -> set:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[<>=!~\[ ]", line)[0].lower().replace("_", "-"))
    return names


def _imports(module_dir: pathlib.Path) -> set:
    found = set()
    for f in module_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {n for n in found if n not in sys.stdlib_module_names and n not in ("perception", "app", "products")}


def test_every_third_party_import_of_a_module_is_declared_somewhere_the_image_installs():
    engine = _declared(ROOT / "perception" / "engine" / "requirements.txt")
    missing = []
    for d in sorted(p for p in MODULES.iterdir() if p.is_dir() and not p.name.startswith("__")):
        own = _declared(d / "requirements.txt") if (d / "requirements.txt").exists() else set()
        for name in sorted(_imports(d)):
            dist = DISTRIBUTION.get(name, name).lower().replace("_", "-")
            if name in PROVIDED or dist in engine or dist in own:
                continue
            missing.append(f"{d.name} imports {name}: add {dist} to perception/modules/{d.name}/requirements.txt")
    assert not missing, "\n".join(missing)


def test_the_dockerfile_installs_requirements_files_and_builds_parsers_without_naming_them():
    text = (ROOT / "perception" / "pipeline" / "Dockerfile").read_text()
    assert "build_parsers.sh" in text and "requirements.txt" in text
    for lib in ("Yolo_seg", "Yolo_face", "impl_Yolo"):
        assert lib not in "".join(l for l in text.splitlines() if not l.lstrip().startswith("#")), \
            f"the Dockerfile must not name the {lib} parser: build_parsers.sh finds it"


def test_every_parser_directory_is_a_buildable_one_and_the_yolo_parser_keeps_its_opencv_flag():
    dirs = sorted((ROOT / "perception" / "models").glob("*/nvdsinfer_custom_impl_*/"))
    assert len(dirs) >= 3
    for d in dirs:
        assert (d / "Makefile").exists(), d
    flags = (ROOT / "perception" / "models" / "parser_flags").read_text()
    assert re.search(r"^nvdsinfer_custom_impl_Yolo\s+OPENCV=1$", flags, re.M)
