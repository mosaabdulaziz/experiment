import logging
import os
import sys

import uvicorn
import yaml
from dotenv import load_dotenv


def load_config() -> dict:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    api_key = os.getenv("QWEN_API_KEY", "")
    config.setdefault("api", {})["api_key"] = api_key
    return config


_NO_KEY_PLACEHOLDER = "not-configured"


def ensure_api_key(config: dict) -> bool:
    """True when the app may start. A missing key only stops it if a VLM is in use.

    The key is consumed by the VLM client (and the embedding search), and the client
    refuses to be built without one. With monitoring.vlm_enabled off -- attendance and
    barber run that way -- nothing calls it, so demanding a DashScope key just to start
    blocked every product that does not describe alerts. A placeholder lets the client
    be built; a call that really needs it fails with the provider's own 401.
    """
    if config.get("api", {}).get("api_key"):
        return True
    if config.get("monitoring", {}).get("vlm_enabled", False):
        return False
    logging.getLogger("iris.main").warning(
        "QWEN_API_KEY is not set; starting without it because monitoring.vlm_enabled is "
        "false. Alert descriptions and VLM chat will not work until a key is set.")
    config.setdefault("api", {})["api_key"] = _NO_KEY_PLACEHOLDER
    return True


def main():
    load_dotenv()
    from app.log_setup import configure_logging

    config = load_config()
    configure_logging(config)

    if not ensure_api_key(config):
        print(
            "\nERROR: QWEN_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your DashScope API key\n"
            "  3. Re-run the app\n"
        )
        sys.exit(1)

    from app.server import create_app

    app = create_app(config)
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{display_host}:{port}"
    line1 = "  IRIS is running"
    line2 = f"  Open in browser:  {url}"
    width = max(len(line1), len(line2)) + 2
    bar = "-" * width
    print(f"\n  +{bar}+")
    print(f"  |{line1.ljust(width)}|")
    print(f"  |{line2.ljust(width)}|")
    print(f"  +{bar}+\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
