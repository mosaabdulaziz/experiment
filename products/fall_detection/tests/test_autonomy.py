"""The product is autonomous and its dependency set is the ruled one.

ACs: AC-N6.1, AC-N6.2, AC-N11.2 (dependency ruling of record, Phase 1).
Design: FALL_DETECTION_DESIGN.md §5 ("nothing in the product imports another product's code — the
shapes are re-declared", products/barber/acl.py:1-16); MULTI_PRODUCT_PLAN.md decision 8.
Pure source scans — no imports of the product's own modules, so this file collects even before
the product exists (and fails on the missing files, which is the right red).
"""
import ast
import re
import tomllib
from pathlib import Path

PRODUCT = Path(__file__).resolve().parents[1]
OTHER_PRODUCTS = ("barber", "attendance", "multicam")


def _product_py_files():
    # The product's own files only: not an installed environment inside it (`uv sync` makes .venv/, whose packages
    # need not even parse under this Python), nor any hidden directory.
    def own(p):
        parts = p.relative_to(PRODUCT).parts[:-1]
        return not any(d in ("node_modules", "site-packages") or d.startswith(".") for d in parts)
    return sorted(p for p in PRODUCT.rglob("*.py") if own(p))


# ---------------------------------------------------------------- AC-N6.1
def test_no_python_file_imports_another_product():
    files = _product_py_files()
    assert files, "the product has no python files yet"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root == "products" or root in OTHER_PRODUCTS:
                    offenders.append(f"{path.relative_to(PRODUCT)}: {name}")
    assert offenders == []


def test_no_source_file_mentions_another_product():
    # the ACL rule is stronger than imports: the shapes are re-declared, not referenced.
    # Comments/docstrings naming the pattern's origin ("forked from products/barber/...") are
    # the repo's own idiom and allowed; code paths are not.
    offenders = []
    for path in _product_py_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\b(products\.(barber|attendance|multicam)|from (barber|attendance|multicam)\b)", code):
                offenders.append(f"{path.relative_to(PRODUCT)}:{i}")
    assert offenders == []


# ---------------------------------------------------------------- AC-N6.2
def test_the_config_loader_loads_only_the_products_own_yaml():
    config_py = PRODUCT / "config.py"
    assert config_py.is_file(), "config.py does not exist yet"
    text = config_py.read_text(encoding="utf-8")
    assert "fall_detection.yaml" in text
    for word in OTHER_PRODUCTS:
        assert word not in text, f"config.py references {word}"
    # the borrowed pattern is _merge/_read only; no path may reach outside the product
    assert not re.search(r"products[/\\.](barber|attendance|multicam)", text)


# ---------------------------------------------------------------- AC-N11.2
def _all_dependency_lists(pyproject):
    project = pyproject.get("project", {})
    lists = {"runtime": project.get("dependencies", [])}
    for group, deps in (pyproject.get("dependency-groups") or {}).items():
        lists[group] = deps
    for extra, deps in (project.get("optional-dependencies") or {}).items():
        lists[extra] = deps
    return lists


def test_pyproject_pins_exactly_one_torch_and_keeps_ultralytics_out_of_runtime():
    pyproject_path = PRODUCT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml does not exist yet"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    lists = _all_dependency_lists(pyproject)
    assert "runtime" in lists and lists["runtime"], "no runtime dependencies declared"

    torches = [(where, d) for where, deps in lists.items() for d in deps
               if re.match(r"^torch([=<>!~\s\[]|$)", d.strip())]
    assert len(torches) == 1, f"expected exactly one torch requirement, got {torches}"

    ultra = [(where, d) for where, deps in lists.items() for d in deps if d.strip().startswith("ultralytics")]
    assert ultra, "ultralytics is missing entirely — the vendor's CLI tests need it"
    for where, _ in ultra:
        assert where != "runtime", "ultralytics must not be a runtime dependency (ruling: test/dev group only)"

    # the deployment machine is an RTX 5090 (sm_120): the pinned index must be the CUDA build
    sources = (pyproject.get("tool", {}).get("uv", {}).get("sources") or {})
    torch_source = sources.get("torch")
    assert torch_source, "torch is not pinned to an index in [tool.uv.sources]"
    assert "cu130" in str(torch_source), f"torch is pinned to an index other than cu130: {torch_source}"
