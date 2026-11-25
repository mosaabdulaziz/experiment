import logging
from app.log_setup import configure_logging


def test_configure_logging_sets_level_from_config():
    configure_logging({"logging": {"level": "WARNING"}})
    assert logging.getLogger().level == logging.WARNING
    configure_logging({"logging": {"level": "INFO"}})
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_defaults_to_info_when_absent():
    configure_logging({})
    assert logging.getLogger().level == logging.INFO
