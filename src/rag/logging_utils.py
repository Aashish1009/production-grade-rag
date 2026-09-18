"""Logging setup and small timing helpers.

Kept separate so that every module can log consistently without each one
reinventing a format string.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Install a Rich-backed root handler. Idempotent unless ``force``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=False)],
        force=True,
    )

    # These libraries are extremely chatty at INFO and drown out our own output.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "transformers",
        "unstructured",
        "pdfminer",
        "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.INFO) -> Iterator[None]:
    """Log how long a block took.

    Used around model loads and batch operations, where knowing whether a
    stage took 2s or 200s is the difference between "working" and "hung".
    """
    start = time.perf_counter()
    logger.log(level, "%s ...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s finished in %.2fs", label, elapsed)
