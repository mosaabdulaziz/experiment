"""The mmaction import shim: the published wheel's package import is broken (mmaction2#2714).

ACs: AC-N11.1 — the copied classifier really builds against the product's environment.
Design: FALL_DETECTION_DESIGN.md §8 (the classifier is mandatory, the copy is the vendor's).

`mmaction/models/__init__.py` imports every model family eagerly; the wheel omits the `drn`
package `localizers` imports and does not declare the `timm` that `multimodal` needs, so
`import mmaction.models` raises. The copy needs three submodules of that package, which
`mmaction_compat` makes importable.
"""
import pytest

import mmaction_compat  # noqa: F401  (installs the shim before the copy is imported)


def test_the_models_package_the_copy_needs_is_importable():
    import torch

    from mmaction.models.backbones.stgcn import STGCN
    from mmaction.models.heads.gcn_head import GCNHead
    from mmaction.models.utils.graph import Graph, normalize_digraph

    graph = Graph(layout="coco", mode="stgcn_spatial")
    assert graph.num_node == 17 and graph.A.shape == (3, 17, 17)
    # the symbols are the real ones: the normalizer computes, and both classes are torch modules
    assert normalize_digraph(graph.A[0]).shape == (17, 17)
    assert issubclass(STGCN, torch.nn.Module) and issubclass(GCNHead, torch.nn.Module)


def test_the_copied_model_builds_and_runs_against_the_product_environment():
    import torch
    from fall_detection.tsstg_model import RecognizerGCN

    model = RecognizerGCN()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2, 100, 17, 3))
    assert tuple(out.shape) == (2, 60) and torch.isfinite(out).all()


def test_the_shim_is_a_no_op_once_the_package_is_imported():
    assert mmaction_compat.install() is False       # already installed by the import above
