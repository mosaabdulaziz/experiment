"""What the app image (the root Dockerfile) must carry from perception/: the compose bind mount hides a missing COPY
on a dev box, an image run without it does not."""
import pathlib

DOCKERFILE = pathlib.Path(__file__).resolve().parents[1] / "Dockerfile"


def test_the_app_image_carries_the_proximity_zone_defaults():
    """app/zones.py reads them on every zone GET and save (the zone editor's proximity defaults and checks)."""
    text = DOCKERFILE.read_text()
    assert ("COPY perception/modules/proximity/zone_defaults.json ./perception/modules/proximity/zone_defaults.json"
            in text)


SAFETY_APP_MODULES = ("config", "service", "api", "notifier", "camera_health")


def test_the_app_image_carries_the_safety_product_s_app_code():
    """The safety product's app-side code lives in its own folder (MULTI_PRODUCT_PLAN.md decision 3); the app imports
    it when IRIS_PRODUCT names safety, so the image must carry every module of it."""
    text = DOCKERFILE.read_text()
    copy = next(line for line in text.splitlines() if line.startswith("COPY products/safety/"))
    for name in SAFETY_APP_MODULES:
        assert f"products/safety/{name}.py" in copy, name
    assert copy.endswith(" ./products/safety/")
    on_disk = {p.stem for p in (DOCKERFILE.parent / "products" / "safety").glob("*.py")}
    assert on_disk == set(SAFETY_APP_MODULES), "a module added to products/safety must be copied into the image too"


def test_the_app_loads_the_safety_code_only_for_the_safety_product():
    """Every other product runs this app too: nothing safety-specific is imported when the module loads."""
    import re
    server = (DOCKERFILE.parent / "app" / "server.py").read_text()
    assert not re.search(r"^from products\.safety", server, re.M)
