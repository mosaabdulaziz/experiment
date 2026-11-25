"""The published mmaction2 wheel cannot be imported as it stands (open-mmlab/mmaction2#2714).

`mmaction/models/__init__.py` imports every model family eagerly, and two of them are broken in
the wheel: `localizers` imports a `drn` package the wheel omits — upstream's directory carries no
`__init__.py`, so `find_packages()` skipped it, and the one-file fix is still an unmerged PR
(#2721) — and `multimodal` needs `timm`, which the wheel does not declare. Either one makes
`import mmaction.models` raise, and the copy imports three submodules of that package.

So this installs a bare `mmaction.models` package whose `__path__` is the real directory: the
submodules import normally, and the broken family imports never run. The product's own
dependency surface stays the pinned `mmaction2` wheel, and the behaviour is the same however the
wheel was installed.

Importing this module installs the shim, so it must be imported before `fall_detection.tsstg_model`.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def install() -> bool:
    """Make `mmaction.models.<submodule>` importable. True when the shim was needed."""
    if "mmaction.models" in sys.modules:
        return False
    import mmaction

    models_dir = Path(mmaction.__file__).parent / "models"
    package = types.ModuleType("mmaction.models")
    package.__path__ = [str(models_dir)]
    package.__package__ = "mmaction.models"
    sys.modules["mmaction.models"] = package
    mmaction.models = package

    # GCNHead builds its loss through the model registry, so the losses must be registered
    import mmaction.models.losses  # noqa: F401

    return True


install()
