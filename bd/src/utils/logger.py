"""Логирование."""
import logging
import sys


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("docvec")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger
