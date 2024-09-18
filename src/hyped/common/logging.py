"""Logging configuration module.

This module sets up a custom logging configuration and provides utility functions to
retrieve loggers. It uses environment variables to dynamically set the log level and
applies custom formatting to both console and file log outputs. The module is intended
to be used across the entire project, allowing consistent logging behavior.
"""

import logging
import logging.config
import os
from logging import Logger  # noqa: F401


def setup_logging() -> None:
    """Set up the logging configuration.

    This function configures the logging system with a custom formatter and log handlers.
    It checks for an environment variable :code:`LOG_LEVEL` to set the default log level,
    which defaults to :code:`'INFO'` if not provided. Two handlers are defined: one for
    console output and one for logging to a file. Both use the same custom format, including
    the timestamp, log level, logger name, and message.
    """
    default_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging_config = {
        "version": 1,
        "formatters": {
            "custom": {
                "()": "hyped.common._formatter.CustomFormatter",
                "format": "%(custom)s[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "custom",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "custom",
                "filename": "app.log",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": default_level,
            },
        },
    }

    logging.config.dictConfig(logging_config)


setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger by name.

    This function returns a logger instance for the provided name. The logger will inherit
    the root logging configuration defined in :code:`setup_logging`.

    Args:
        name (str): The name of the logger to retrieve.

    Returns:
        logging.Logger: The logger instance associated with the given name.
    """
    return logging.getLogger(name)


def get_cls_logger(cls: type) -> logging.Logger:
    """Retrieve a logger for a specific class.

    This function returns a logger instance using the module and class name as the logger name.
    This allows for easy identification of log messages associated with a specific class.

    Args:
        cls (type): The class for which the logger should be created.

    Returns:
        logging.Logger: The logger instance associated with the class.
    """
    return get_logger(f"{cls.__module__}.{cls.__qualname__}")
