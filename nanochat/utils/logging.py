"""Logging utilities for NanoChat."""

from __future__ import annotations

import logging
import re
from typing import Any

class ColoredFormatter(logging.Formatter):
    """Formatter that enriches log output with ANSI colors."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == "INFO":
            message = re.sub(
                r"(\d+\.?\d*\s*(?:GB|MB|%|docs))",
                rf"{self.BOLD}\1{self.RESET}",
                message,
            )
            message = re.sub(
                r"(Shard \d+)",
                rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}",
                message,
            )
        return message


def setup_default_logging(level: int = logging.INFO, **kwargs: Any) -> logging.Logger:
    """Configure the root logger with colored formatting.

    Parameters
    ----------
    level:
        Desired logging level. Defaults to :data:`logging.INFO`.
    kwargs:
        Extra keyword arguments forwarded to :class:`logging.StreamHandler`.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """

    handler = logging.StreamHandler(**kwargs)
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("nanochat")


# Initialize logging eagerly for backward compatibility with legacy modules.
LOGGER = setup_default_logging()
