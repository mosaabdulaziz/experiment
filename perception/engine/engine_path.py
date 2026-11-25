"""Where nvinfer will actually serialize each GIE's TensorRT engine.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU, same
convention as box_color.py and zone_dedup.py.

nvinfer picks one of two engine-naming branches, and which one it takes is a
property of the CONFIG FILE, not of the caller:

  - `engine-create-func-name` present (with a value) -> TrtModelBuilder::buildModel()
    takes the custom-engine branch and derives
    `<cwd>/model_b<batch>_gpu<id>_<mode>.engine`. This name contains nothing that
    identifies the model, so two configs on this branch at the same batch and
    precision COLLIDE -- the second GIE silently deserializes the first one's
    engine, with no error at any level.
  - absent (or present with an empty value, which nvinfer treats the same as
    absent) -> the ONNX branch, deriving
    `<onnx-path>_b<batch>_gpu<id>_<mode>.engine` beside the ONNX itself. That name
    carries both the config's directory and the ONNX basename, so two ONNX-branch
    GIEs cannot collide.

Deriving the branch from the config here (rather than taking it as an argument)
is deliberate: the caller cannot get it wrong, so a config that copies the
pgie's `engine-create-func-name` line cannot be mapped into the pgie's namespace.
`assert_distinct_engine_paths` then catches the remaining case -- two configs
that genuinely both take the custom branch -- at startup instead of at runtime.
"""

import logging
import os

from .model_manifest import file_sha256

logger = logging.getLogger("iris.sidecar")

_MODES = {0: "fp32", 1: "int8", 2: "fp16"}


def _parse(gie_config):
    """(mode, onnx_file, custom_engine, read_ok) read from a GIE config file.

    read_ok is False only when the file itself could not be opened (missing,
    unreadable, permissions) -- distinct from a file that opened fine but
    declares neither onnx-file= nor engine-create-func-name=. gie_engine_path
    needs that distinction to tell an operator debugging a boot failure whether
    to look for a missing file or a bad one."""
    mode, onnx_file, custom_engine, read_ok = "fp32", None, False, False
    try:
        with open(gie_config) as f:
            read_ok = True
            for line in f:
                line = line.strip()
                if line.startswith("network-mode="):
                    mode = _MODES.get(int(line.split("=", 1)[1].strip()), "fp32")
                elif line.startswith("onnx-file="):
                    onnx_file = line.split("=", 1)[1].strip()
                elif line.startswith("engine-create-func-name="):
                    # nvinfer only takes the custom-engine branch when this key
                    # carries an actual function name; an empty value is a no-op,
                    # same as the key being absent.
                    if line.split("=", 1)[1].strip():
                        custom_engine = True
    except Exception as exc:
        logger.warning("could not parse %s for engine path (%s); using defaults",
                       gie_config, exc)
    return mode, onnx_file, custom_engine, read_ok


def gie_engine_path(gie_config, batch_size, gpu_id=0):
    """Path nvinfer will serialize this GIE's engine to. Deletes a cached engine
    that looks stale before returning, so nvinfer rebuilds rather than silently
    deserializing a truncated or outdated file."""
    mode, onnx_file, custom_engine, read_ok = _parse(gie_config)
    if custom_engine:
        engine_path = f"{os.getcwd()}/model_b{batch_size}_gpu{gpu_id}_{mode}.engine"
    else:
        if not onnx_file:
            if not read_ok:
                raise ValueError(
                    f"{gie_config} could not be opened (missing or unreadable); "
                    "cannot derive an engine path")
            raise ValueError(
                f"{gie_config} has no onnx-file= and no engine-create-func-name=; "
                "cannot derive an engine path")
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(gie_config)), onnx_file)
        engine_path = f"{onnx_path}_b{batch_size}_gpu{gpu_id}_{mode}.engine"
    invalidate_stale_engine(engine_path, gie_config, onnx_file)
    claim_engine(engine_path, gie_config, onnx_file)
    return engine_path


def claim_engine(engine_path, gie_config, onnx_file):
    """Make an engine belong to ONE exact ONNX file: delete it if it was built from anything else.

    A marker beside the engine (`<engine>.src`) records the ONNX it was built from: real path, size and sha256.
    An engine whose marker does not match the ONNX the config names now is deleted, so nvinfer rebuilds it (a
    one-off cost) instead of deserializing the wrong network. Two cases this catches:
      - the custom-engine name (`<cwd>/model_b<batch>_gpu<id>_<mode>.engine`) says nothing about the model, so two
        products run from one checkout at the same batch size and precision would share the file, whichever started
        second running the first one's network with no error at any level (`assert_distinct_engine_paths` only sees
        one process's GIEs);
      - the ONNX at the same path was replaced by another of the same size (a retrain or a rollback of the same
        architecture: `dvc checkout` of an older pointer), which path+size alone, and an mtime the checkout may not
        move, cannot see.
    An engine with no marker, or a marker without a sha256 (written before 2026-11), cannot be tied to its ONNX and
    is rebuilt once. The sha256 is the one the startup model manifest check already computed when it checked this
    file (model_manifest.file_sha256), so the ONNX is not hashed twice. Best-effort like invalidate_stale_engine: any
    error leaves everything in place."""
    if not onnx_file:
        return
    try:
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(gie_config)), onnx_file)
        if not os.path.isfile(onnx_path):
            return                                    # nothing to claim; nvinfer reports the missing ONNX itself
        claim = f"{os.path.realpath(onnx_path)}\n{os.path.getsize(onnx_path)}\nsha256:{file_sha256(onnx_path)}\n"
        marker = engine_path + ".src"
        if os.path.isfile(engine_path):
            built_from = ""
            if os.path.isfile(marker):
                with open(marker) as f:
                    built_from = f.read()
            if built_from != claim:
                if not built_from or "sha256:" not in built_from:
                    why = "has no marker naming the ONNX sha256 it was built from (built before 2026-11)"
                else:
                    why = f"was built from {built_from.splitlines()[0]} ({built_from.splitlines()[-1][:19]}...)"
                logger.warning("cached engine %s %s; this config uses %s (%s...) -- deleting it so nvinfer builds "
                               "the right network", engine_path, why, onnx_path, claim.splitlines()[-1][:19])
                os.remove(engine_path)
        with open(marker, "w") as f:
            f.write(claim)
    except Exception as exc:
        logger.warning("engine ownership check failed for %s (%s); leaving file as-is",
                       engine_path, exc)


def assert_distinct_engine_paths(entries):
    """Raise if any two GIEs would serialize to the same engine file.

    `entries` is a list of (gie_name, config_path, engine_path). A collision means
    one GIE deserializes another's model with no error at any level -- boxes for the
    wrong classes, or none -- so this fails the boot loudly rather than shipping a
    pipeline that looks healthy and is not.

    Paths are compared after os.path.realpath, so two configs that reach the same
    file via different relative traversals (e.g. "a/b.engine" vs "a/./b.engine"),
    or via a relative spelling against cwd vs an absolute one (os.path.normpath is
    purely lexical and misses that case -- a relative and an absolute spelling of
    the same file slip past it as textually different strings), still collide
    instead of slipping past undetected. The error message quotes the original,
    un-resolved path -- the one the operator actually wrote and will recognise."""
    seen = {}
    for name, config_path, engine_path in entries:
        key = os.path.realpath(engine_path)
        if key in seen:
            other_name, other_config = seen[key]
            raise RuntimeError(
                f"GIE engine path collision: '{name}' ({config_path}) and "
                f"'{other_name}' ({other_config}) both serialize to {engine_path}. "
                "One would silently deserialize the other's model. Remove "
                "'engine-create-func-name' from one config so it takes nvinfer's "
                "ONNX branch and derives a name beside its own ONNX.")
        seen[key] = (name, config_path)


def invalidate_stale_engine(engine_path, gie_config, onnx_file):
    """Delete engine_path if it looks stale or corrupt, forcing nvinfer to rebuild
    it fresh. Two conservative checks, either is enough to delete -- never delete a
    plausibly-good engine:
      - implausibly small (< 1 MB): a real TRT engine at any batch size worth running
        is tens of MB; this only catches a write killed almost immediately (an
        interrupted rebuild leaving a truncated file at the derived path).
      - older than the onnx that's supposed to have built it: the onnx was
        re-exported/replaced after this engine was last built.
    Best-effort: any error here just leaves the file in place -- nvinfer's own
    deserialize probe is the real fallback, not this."""
    try:
        if not os.path.isfile(engine_path):
            return
        size = os.path.getsize(engine_path)
        if size < 1_000_000:
            logger.warning("cached engine %s is suspiciously small (%d bytes) -- "
                           "treating as a truncated build, deleting to force a rebuild",
                           engine_path, size)
            os.remove(engine_path)
            return
        if onnx_file:
            # Same abspath-based resolution gie_engine_path uses for its own onnx
            # derivation -- otherwise the two can disagree about which onnx file is
            # "the config's onnx" whenever cwd isn't the config's own directory, and
            # this check silently no-ops instead of catching a real staleness case.
            onnx_path = os.path.join(os.path.dirname(os.path.abspath(gie_config)), onnx_file)
            if os.path.isfile(onnx_path) and os.path.getmtime(onnx_path) > os.path.getmtime(engine_path):
                logger.warning("cached engine %s is older than %s -- onnx changed since "
                               "the engine was last built, deleting to force a rebuild",
                               engine_path, onnx_path)
                os.remove(engine_path)
    except Exception as exc:
        logger.warning("engine staleness check failed for %s (%s); leaving file as-is",
                       engine_path, exc)
