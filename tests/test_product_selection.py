"""Which product the dashboard opens with.

The app reports its products from IRIS_PRODUCT (an empty value means "every product", and the shell opens the
first one: attendance). Attendance needs its own backend and a Vite proxy for /attendance/ws, so a surveillance
preset that leaves the variable unset lands on a page that never connects. The compose service therefore passes
the variable through explicitly, and the presets default it to the surveillance product.
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_the_app_container_receives_iris_product_from_the_shell_or_env():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    env = compose["services"]["iris"]["environment"]
    assert any(str(e).startswith("IRIS_PRODUCT=${IRIS_PRODUCT") for e in env), \
        "without this, a recipe cannot choose the product for one run (only .env can)"


def test_the_presets_open_on_the_surveillance_ui_unless_the_env_says_otherwise():
    script = (REPO / "scripts" / "run_setup.sh").read_text()
    assert 'export IRIS_PRODUCT="${IRIS_PRODUCT:-iris}"' in script   # `:-` keeps an explicit .env/shell value


def test_the_shell_moves_a_bare_address_into_the_active_products_slice():
    """The surveillance product mounts a router under basename="/iris": an address outside that slice (a bare URL
    on first load) matches none of its routes and the page is blank. The shell must move it, as switchTo does.
    (The frontend has no unit tests yet; CI only type-checks and builds it.)"""
    shell = (REPO / "frontend" / "src" / "shared" / "Shell.tsx").read_text()
    after_base = shell.split("setProductBase(product.id)", 1)[1].split("product.load()", 1)[0]
    assert "location.hash" in after_base and "navigate('/')" in after_base


def test_the_app_container_receives_the_same_pipeline_settings_file_as_the_sidecar():
    """The sidecar's command and the app both name SIDECAR_CONFIG; the app only got it through .env, so a shell
    `SIDECAR_CONFIG=... just up` left it reading the default file: no camera roles, an empty attendance page."""
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    assert any(str(e).startswith("SIDECAR_CONFIG=${SIDECAR_CONFIG") for e in compose["services"]["iris"]["environment"])
