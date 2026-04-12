"""Logging configuration helper for DeepLearning Performance Lab tutorials.

Provides a standardized logging setup so all tutorials produce consistent,
readable console output. Tutorials use both the logging module (for technique
output with proper log levels) and print() (for section headers and visual
explanations).
"""

import logging
import sys


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a standardized format.

    Args:
        name: Logger name, typically the tutorial or module name.
        level: Logging level (default: logging.INFO).

    Returns:
        A configured logging.Logger instance with console handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
