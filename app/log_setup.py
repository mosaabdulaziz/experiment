import logging


def configure_logging(config: dict) -> None:
    """Configure root logging from the `logging` config section. Idempotent."""
    level_name = str(config.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # re-apply level on repeated calls (tests + reload)
    )
    logging.getLogger().setLevel(level)
