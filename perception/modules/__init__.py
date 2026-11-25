"""Use-case modules for the DeepStream engine. One package per capability, named for what it does.

The engine never imports from here by name: a product lists the modules it wants under `modules:` in its sidecar
yaml and the engine loads them (perception/engine/modules.py). To add one, copy `_template/`.
"""
