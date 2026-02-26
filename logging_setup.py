import logging
import os

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_format = os.environ.get("LOG_FORMAT", DEFAULT_LOG_FORMAT)
    logging.basicConfig(level=level, format=log_format)
