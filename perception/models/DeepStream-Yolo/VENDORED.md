# Vendored: DeepStream-Yolo

Parts of this folder come from [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo).
The rest of the upstream checkout is ignored (`.gitignore`, `perception/models/DeepStream-Yolo/**`).

| | |
|---|---|
| upstream | https://github.com/marcoslucianops/DeepStream-Yolo |
| commit | `2894babce8e75c49115dbe0c7b516289ed853565` ("Add YOLOv6 and YOLO-Master + minor fixes", 2026-01-25) |
| licence | MIT, [`LICENSE.md`](LICENSE.md) (upstream's, unchanged; sha256 `2c713d812a53960b2f368ff3c1027a2f96739ee5847d84b5646ebf60b4a53a42`). It covers every file below |
| vendored | 2026-11-04, from the training server's checkout (`/data/IRIS/repo/DeepStream-Yolo`, at that commit); both files checked byte-identical to GitHub at that commit |

## Files

| file | state |
|---|---|
| `utils/export_yolo11.py` | unmodified upstream at the commit above. sha256 `46d9dcb897eeb013b5d47ab1917f7694c7b8a0d706dbeb1f875a344611bb551a`, pinned in `mtmc/export_vehicle_onnx.py` (`UPSTREAM_SHA256`), which wraps it and warns when the file no longer matches. Do not edit it: change the wrapper instead, or re-vendor a new upstream commit and update this note and the pin together |
| `LICENSE.md` | unmodified upstream |
| `nvdsinfer_custom_impl_Yolo/` (the custom YOLO parser and engine builder) | vendored before this note (first commit in this repo: `49e019d`, 2026-09-23). Which upstream commit it came from is not recorded. Compared with upstream `2894bab`, these four files differ; the rest match: |

Parser files that differ from upstream `2894bab` (origin of the changes unknown; left as they are):

| file | lines differing (diff `<`/`>` count) |
|---|---:|
| `nvdsinfer_custom_impl_Yolo/Makefile` | 8 |
| `nvdsinfer_custom_impl_Yolo/calibrator.h` | 4 |
| `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp` | 36 |
| `nvdsinfer_custom_impl_Yolo/yolo.cpp` | 7 |

They may be local fixes, or an older or newer upstream version. Find out before re-vendoring the parser, so that a
re-vendor does not silently drop them.
