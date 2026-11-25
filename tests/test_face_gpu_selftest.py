"""The recognizer says so when it is not really on the GPU.

ONNX Runtime retries on the CPU when cuDNN's conv engines fail to load and only logs it, so a
recognizer built for the GPU can run ~16x slower with nothing saying so (measured: 225 ms vs 14 ms
per batch of 8 faces). verify_gpu_inference runs one real inference and reports where it landed.
"""
import sys
import types
from pathlib import Path

for name in ("insightface", "insightface.app", "insightface.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["insightface.app"].FaceAnalysis = object
sys.modules["insightface.utils"].face_align = types.SimpleNamespace()

from perception.modules.face_recognition.recognizer import verify_gpu_inference  # noqa: E402


class Session:
    def __init__(self, providers, raises=None):
        self._p, self._raises, self.ran = providers, raises, 0

    def run(self, _out, feed):
        self.ran += 1
        assert list(feed.values())[0].shape == (1, 3, 112, 112)
        if self._raises:
            raise self._raises

    def get_providers(self):
        return self._p


def app_with(session):
    rec = types.SimpleNamespace(session=session, input_size=(112, 112), input_name="input.1")
    return types.SimpleNamespace(models={"detection": object(), "recognition": rec})


def test_a_session_on_cuda_is_ok():
    s = Session(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert verify_gpu_inference(app_with(s)) == (True, "CUDAExecutionProvider") and s.ran == 1


def test_a_session_that_fell_back_to_the_cpu_is_reported():
    """ONNX Runtime drops CUDA from the session's providers after the first failing run."""
    ok, detail = verify_gpu_inference(app_with(Session(["CPUExecutionProvider"])))
    assert not ok and "CPUExecutionProvider" in detail


def test_an_inference_that_raises_is_reported_with_the_reason():
    ok, detail = verify_gpu_inference(app_with(Session(["CUDAExecutionProvider"], RuntimeError("cudnn: sublibrary loading failed"))))
    assert not ok and "sublibrary" in detail


def test_no_recognition_model_is_not_an_error():
    assert verify_gpu_inference(types.SimpleNamespace(models={}))[0] is True
