"""Every third-party image in the compose files is pinned by version AND digest (DEPENDENCIES.md, "Compose images"):
`latest` or a bare major tag means two machines running the same file can run different software."""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMPOSE = sorted(REPO.glob("docker-compose*.yml"))
PINNED = re.compile(r"^[a-z0-9./_-]+:(?!latest@)[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$")


def images(path):
    """(line number, image) for every `image:` line, except the images this repo builds itself (named from variables)."""
    for n, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(r"\s*image:\s*(\S+)", line)
        if m and "${" not in m.group(1):
            yield n, m.group(1)


@pytest.mark.parametrize("path", COMPOSE, ids=lambda p: p.name)
def test_third_party_images_are_pinned_by_version_and_digest(path):
    loose = [f"{path.name}:{n}: {image}" for n, image in images(path) if not PINNED.match(image)]
    assert not loose, "pin as name:version@sha256:<digest> (DEPENDENCIES.md):\n" + "\n".join(loose)


def test_the_check_sees_the_images():
    assert sum(1 for p in COMPOSE for _ in images(p)) >= 12
