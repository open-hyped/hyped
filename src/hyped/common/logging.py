"""This module provides a utility function to configure logging for the application.

The logging setup supports both console and file logging, with a customizable
formatter that integrates worker rank information for distributed environments.
"""

import logging
import logging.config
from typing import Any

from ._worker import get_worker_info

# ANSI colors
RESET = "\033[0m"
LOG_COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
}
# Colors to cycle for different ranks (can add more)
RANK_COLORS = [
    "\033[96m",  # Cyan
    "\033[93m",  # Yellow
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[95m",  # Magenta
    "\033[94m",  # Blue
    "\033[90m",  # Gray
]

GREY = "\033[90m"
RANK_FORMAT = "%(rank_color)s[RANK %(rank)s/%(world_size)s]%(reset_color)s "
DEFAULT_FORMAT = (
    f"{GREY}[%(asctime)s] [%(name)s]{RESET} "
    "%(level_color)s[%(levelname)s]%(reset_color)s "
    "%(message)s"
)


class RankAwareFormatter(logging.Formatter):
    """Rank Aware Logging Formatter.

    This formatter extends the standard :class:`logging.Formatter` to include
    information about the process rank and world size when running in a
    distributed environment. If the worker information is available, the log
    message will be formatted to include the rank and world size, along with
    a color specific to the rank. If the worker information is not available,
    it falls back to the standard formatting.
    """

    def __init__(self, fmt: str = DEFAULT_FORMAT, **kwargs: Any) -> None:
        """Initialize the formatter.

        Args:
            fmt (str, optional): The log message format string. This format string
                will be appended to the rank-aware format. Defaults to
                :const:`DEFAULT_FORMAT`.
            **kwargs (Any): Keyword arguments passed to the underlying
                :class:`logging.Formatter` instances.
        """
        self.rank_formatter = logging.Formatter(fmt=RANK_FORMAT + fmt, **kwargs)
        self.no_rank_formatter = logging.Formatter(fmt=fmt, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record.

        If worker information is available (e.g., in a distributed environment),
        the formatted log message will include the process rank and world size,
        along with a color specific to the rank. If the worker information is
        not available, it falls back to the standard formatting.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.
        """
        # Set Colors
        record.reset_color = RESET
        record.level_color = LOG_COLORS.get(record.levelname, "")
        # get worker info
        info = get_worker_info()

        if info is not None:
            # Assign values to record so format can access them
            record.rank = info.rank
            record.world_size = info.num_workers
            # set rank color
            record.rank_color = RANK_COLORS[info.rank % len(RANK_COLORS)]
            return self.rank_formatter.format(record)
        else:
            return self.no_rank_formatter.format(record)


def setup_logging(level: int | str, log_file: None | str = None) -> None:
    """Set up the logging configuration.

    Args:
        level (None | int | str, optional): The logging level.
        log_file (None | str, optional): The file path for logging output. If set to
            :code:`None` (default) no log file is created.

    **Note**: By default, this function is called before execution, with the values
        for :code:`level` and :code:`log_file` taken from the environment variables
        :code:`LOG_LEVEL` and :code:`LOG_FILE`, respectively. If these environment
        variables are not set, the default logging level is :code:`"WARNING"`, and no
        log file will be created.
    """
    logging_config = {
        "version": 1,
        "formatters": {
            "custom": {
                "()": "hyped.common.logging.RankAwareFormatter",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "custom",
            },
        },
        "loggers": {
            "hyped": {
                "handlers": ["console"],
                "level": level,
            },
        },
    }

    # add file handler
    if log_file is not None:
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "custom",
            "filename": log_file,
            "mode": "a",
        }
        logging_config["loggers"]["hyped"]["handlers"].append("file")

    logging.config.dictConfig(logging_config)
