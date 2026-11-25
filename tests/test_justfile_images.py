"""The recipes must use the image compose uses.

Compose names its images from the top-level `name:` (iris). The recipes that `docker run` the sidecar image
(face registration for attendance and barber) used to build the name from COMPOSE_PROJECT_NAME with a different
default, so on a checkout without that variable in .env they ran an OLDER image -- one whose torch brings a
second cuDNN -- and GPU face registration failed.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
JUSTFILE = (REPO / "justfile").read_text()


def test_no_recipe_names_the_sidecar_image_from_a_different_default_than_compose():
    assert "COMPOSE_PROJECT_NAME:-surveillance-camera-sys}-deepstream-sidecar" not in JUSTFILE


def test_the_registration_recipe_asks_compose_for_the_image():
    assert JUSTFILE.count("docker compose config --images") >= 1


def test_the_fallback_name_is_composes_own_project_name():
    name = yaml.safe_load((REPO / "docker-compose.yml").read_text())["name"]
    assert f"COMPOSE_PROJECT_NAME:-{name}}}-deepstream-sidecar" in JUSTFILE


def _has_docker_compose() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "compose", "version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_docker_compose(), reason="no docker compose")
def test_what_compose_reports_matches_the_pattern_the_recipes_grep_for():
    out = subprocess.run(["docker", "compose", "config", "--images"], cwd=REPO, capture_output=True, text=True).stdout
    assert re.search(r"deepstream-sidecar:latest$", out, re.M), out
