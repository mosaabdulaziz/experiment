"""Every per-camera Redis key the engine and the safety modules write goes through the key id (role_scoped_keys).

Most of this code only imports inside the DeepStream image, so this reads the source: a key such as
f"iris:svr_done:{source_id}" built from a bare camera id would be written under the camera even in a role-scoped
process, and two role sidecars on one camera would collide on it again."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# f"iris:<name>:{<expr>}" -- the expression is what the key is named by.
KEY = re.compile(r"""f["']iris:([a-z_]+):\{([^}]*)\}""")

# Keys that are deliberately per camera, not per stream key:
ALLOWED = {
    # A camera's person-scale calibration is the camera's, whichever process measured it.
    ("perception/modules/proximity/__init__.py", "proximity"),
}
# Modules a role-scoped process refuses (sources.unscoped_modules); their keys stay camera-keyed.
CAMERA_KEYED_DIRS = ("perception/modules/face_recognition", "perception/modules/customer_faces",
                     "perception/modules/appearance", "perception/modules/face_detect")


def _files():
    for base in ("perception/engine", "perception/modules"):
        for path in sorted((ROOT / base).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if "/tests/" in rel or rel.startswith(CAMERA_KEYED_DIRS):
                continue
            yield rel, path.read_text()


def test_no_per_camera_key_is_named_by_a_bare_camera_id():
    offenders = []
    for rel, text in _files():
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, expr in KEY.findall(line):
                if (rel, name) in ALLOWED or name in ("ctrl", "analytics"):
                    continue
                if "key" not in expr:          # key_id(...), st.key_id, key -- anything else is a camera id
                    offenders.append(f"{rel}:{lineno}: iris:{name}:{{{expr}}}")
    assert offenders == []
