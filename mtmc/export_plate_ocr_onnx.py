"""Export the fine-tuned plate OCR (Paddle inference model) to ONNX for the sidecar.

Needs paddlepaddle, paddle2onnx and setuptools in a throwaway venv (Python 3.11):
    python -m mtmc.export_plate_ocr_onnx
Writes <model dir>/inference/plate_rec.onnx; the sidecar reads it with onnxruntime.
"""
import subprocess
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "products" / "el-mohandes" / "models" / "v5_arabic_mobile-fine-tuned"

if __name__ == "__main__":
    sys.exit(subprocess.call([
        "paddle2onnx", "--model_dir", str(MODEL / "inference"),
        "--model_filename", "inference.json", "--params_filename", "inference.pdiparams",
        "--save_file", str(MODEL / "inference" / "plate_rec.onnx"), "--opset_version", "14",
    ]))
