"""The pose parser, proved on a plain host.

The parser is C++ that normally runs inside the DeepStream container, where nvinfer hands it a
TensorRT output buffer. There is no DeepStream and no GPU on a dev box, so the parser is
compiled here against a STUB of DeepStream's header (the stub carries the real declarations the
parser uses, under their real names) and fed a tensor whose contents are known.

This is design §11's own proof for the parser -- the parser is fed a tensor and its output
checked -- and it is also what keeps the parser honest about DeepStream's API: a field that does
not exist in the real header (or a type that is not what the header declares) is a compile error
here, long before the container build.

The test skips where there is no C++ compiler; it never skips a broken parser.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PARSER_DIR = REPO / "perception" / "models" / "DeepStream-Yolo-Pose" / "nvdsinfer_custom_impl_Yolo_pose"
PARSER = PARSER_DIR / "nvdsparsepose_Yolo.cpp"
HARNESS = PARSER_DIR / "test" / "parse_pose_test.cpp"
STUB = PARSER_DIR / "test" / "stub"

# The compilers that can build it. g++ first: the MSYS2 one is on the dev box's PATH.
COMPILERS = ("g++", "c++", "clang++")


def _compiler():
    for name in COMPILERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def _build(tmp_path):
    """Compile the parser and the harness. Returns the binary, or None when there is no compiler."""
    compiler = _compiler()
    if compiler is None:
        return None
    binary = tmp_path / ("parse_pose_test.exe" if sys.platform == "win32" else "parse_pose_test")
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-I", str(STUB), str(PARSER), str(HARNESS), "-o", str(binary)],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"the pose parser does not build:\n{result.stdout}\n{result.stderr}")
    return binary


def test_the_pose_parser_parses_a_tensor_whose_contents_are_known(tmp_path):
    """The parser's whole contract, on a host without DeepStream: the threshold, the NMS, the
    payload it packs and the shape it insists on."""
    binary = _build(tmp_path)
    if binary is None:
        pytest.skip("no C++ compiler on this machine (g++, c++ or clang++)")
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"the pose parser's own checks failed:\n{result.stdout}\n{result.stderr}")
    assert "all checks passed" in result.stdout


def test_the_parser_compiles_against_the_stub_of_deepstreams_header():
    """The stub carries DeepStream's declarations under their real names, so the parser's use of
    them is checked here rather than in the container. This test says which declarations the stub
    makes, so a DeepStream upgrade that changes them is a visible diff and not a container build
    failure."""
    stub = (STUB / "nvdsinfer_custom_impl.h").read_text(encoding="utf-8")
    for declaration in (
        "#define NVDSINFER_MAX_DIMS 8",  # DeepStream 8.0's name for the dims constant
        "const char* layerName;",       # the layer's name (NOT `name`)
        "unsigned int numDims;",
        "unsigned int d[NVDSINFER_MAX_DIMS];",
        "int bindingIndex;",
        "void* buffer;",
        "unsigned int width;",          # NvDsInferNetworkInfo
        "unsigned int height;",
        "unsigned int channels;",
        "unsigned int numClassesConfigured;",  # NOT `numClasses`
        "std::vector<float> perClassPreclusterThreshold;",   # a vector, NOT a float*
        "unsigned int classId;",
        "float* mask;",
        "unsigned int mask_width;",
        "unsigned int mask_height;",
        "unsigned int mask_size;",
    ):
        assert declaration in stub, f"the stub no longer declares {declaration!r}"
