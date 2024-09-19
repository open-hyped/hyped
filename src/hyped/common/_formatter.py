"""Custom logging formatter for adding worker information.

This module provides a custom log formatter that appends worker-specific information to log
messages. If the worker information is available (retrieved from :func:`get_worker_info`),
it includes the worker's rank in the log message. Otherwise, no additional information is added.
"""

import logging
import logging.config

from ._worker import get_worker_info


class CustomFormatter(logging.Formatter):
    """Custom log formatter that appends worker rank to log messages.

    This formatter extends the default logging formatter to include worker-specific information,
    such as the rank of the worker, in the log message. If no worker information is available,
    the log message is formatted without additional context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.

        This method retrieves worker information using :func:`get_worker_info`. If worker info is
        available, it adds the worker's rank to the log message. Otherwise, the log message is
        formatted as usual.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with or without worker rank.
        """
        info = get_worker_info()
        record.custom = "" if info is None else f"[Rank {info.rank}] "
        return super().format(record)
