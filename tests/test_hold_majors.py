"""tools/deps/hold_majors.py: what counts as a major version change in an automated update, and that every lock in the
repository is one it (and the other lock checks) knows about."""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("hold_majors_under_test", REPO / "tools" / "deps" / "hold_majors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("version, expected", [("12.3.0", (12,)), ("2.14.1+cu130", (2,)), ("0.141.1", (0, 141)),
                                               ("0.5", (0, 5)), ("2026.10.23", None), ("1.0.1", (1,))])
def test_major_is_the_first_number_or_the_first_two_below_1(version, expected):
    assert load().major(version) == expected


def test_majors_and_held_packages_are_caught_and_go_back_to_the_base_version():
    mod = load()
    base = {"pillow": {"12.3.0"}, "fastapi": {"0.141.1"}, "anyio": {"4.15.1"}, "certifi": {"2026.7.22"},
            "torch": {"2.14.0", "2.14.0+cpu"}, "insightface": {"1.0.1"}}
    new = {"pillow": {"13.0.0"}, "fastapi": {"0.162.0"}, "anyio": {"4.16.0"}, "certifi": {"2027.1.5"},
           "torch": {"2.14.1", "2.14.1+cpu"}, "insightface": {"1.0.1"}, "opentelemetry-api": {"1.45.2"}}
    found = {name: back for name, back, _ in mod.violations(base, new)}
    # a minor, a new calendar year, an unchanged and a new package are fine; torch is held whatever moved
    assert found == {"pillow": "12.3.0", "fastapi": "0.141.1", "torch": "2.14.0"}


def test_a_downgrade_is_not_held():
    assert load().violations({"pillow": {"13.0.0"}}, {"pillow": {"12.3.0"}}) == []


def _locks_on_disk():
    """The repository's locks, as directories relative to it ("." for the root's). Found on disk, not with git (CI's
    image has none): an installed environment (.venv, site-packages) or a hidden directory is never one of ours."""
    found = []
    for lock in REPO.rglob("uv.lock"):
        parts = lock.relative_to(REPO).parts[:-1]
        if any(d in (".venv", "node_modules", "site-packages") or d.startswith(".") for d in parts):
            continue
        found.append(str(Path(*parts)) if parts else ".")
    return sorted(found)


def test_every_lock_in_the_repository_is_checked_everywhere():
    """A new lock must be added to hold_majors.LOCKS, the CI lock check, `just check`, `just deps-lock` and
    tests/test_locks.py; this fails until it is."""
    on_disk = _locks_on_disk()
    assert sorted(load().LOCKS) == on_disk
    ci = (REPO / ".gitlab-ci.yml").read_text()
    loop = re.search(r"for d in (\. perception.*?); do", re.sub(r"\s+", " ", ci)).group(1).split()
    assert sorted(loop) == on_disk, "the deps-locks job's loop"
    just = (REPO / "justfile").read_text()
    check = re.search(r"for d in (\. perception.*?); do", just).group(1).split()
    assert sorted(check) == on_disk, "`just check`'s loop"
    relock = re.search(r"^deps-lock:\n((?:    .*\n)+)", just, re.M).group(1)
    for d in on_disk:
        assert (d == "." and "\n    env -u VIRTUAL_ENV uv lock\n" in "\n" + relock) or f"cd {d} &&" in relock, d
    tests = (REPO / "tests" / "test_locks.py").read_text()
    for d in on_disk:
        assert d == "." or f'"{d.split("/")[-1]}"' in tests or f"/ \"{d.split('/')[-1]}\"" in tests, d
