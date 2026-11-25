# Third-party models the research scripts can use (not versioned here)

The multicam pipeline does not load these, and they are not ours, so they are not in git or DVC (removed
2026-10-28, issue #83). The research scripts in `mtmc/` that name them need the files at the path shown.
What the pipeline runs, and our own finetunes, are under `products/el-mohandes/models/` (see DATA.md).

| folder | what it is | licence | used by |
|---|---|---|---|
| `yolov11-license-plate-detection/` | `license-plate-finetune-v1{n,s,m,l,x}` YOLO11 plate detectors, `.pt` and `.onnx`; Hugging Face `morsetechlab/yolov11-license-plate-detection`. `v1s.pt` is the base the plate finetune starts from. | AGPL-3.0 | `mtmc/detect_plates.py`, `batch_detect_plates.py`, `plate_ocr.py`, `benchmark_models.py`, `finetune_plate_detector.py` |
| `egyptian-license-plate-detector/` | YOLOv8n trained on the EALPR Egyptian vehicles dataset (`best.pt`, ONNX export, `yolov8n.pt`, `yolo26n.pt`) | MIT | `benchmark_models.py` |
| `paddleocr-onnx/` | PaddleOCR PP-OCRv3 arabic mobile recognition, ONNX (legacy) | see PaddleOCR | `batch_ocr_plates.py`, `eval_ocr_quality.py`, `plate_ocr.py` |
| `arabic_PP-OCRv3_mobile_rec/` | PaddleOCR arabic mobile recognition (model card says PP-OCRv5), Paddle and ONNX | Apache-2.0 | `eval_ocr_quality.py` |

Put the files back at `mtmc/models/<folder>/` to run those scripts; git ignores the folders. Copies exist in the old
commits of `main` and, until a cleanup of the store, in the DVC remote (`dvc get` is not needed: check out the commit
before this change and run `dvc pull`).
