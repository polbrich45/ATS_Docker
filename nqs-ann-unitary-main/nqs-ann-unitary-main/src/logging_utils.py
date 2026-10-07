"""Tab-indented general_log output, display policy, and interrupt-safe loops."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import TypeVar

from general_log import Logger, log_timing_summary

# ------------------------------------------------------------------------------

T                   = TypeVar("T")
_LOGGER			    : Logger | None = None
_SILENT			    = False
_MAX_LEVEL		    : int | None = None
_DISPLAY_MAX_LEVEL  = {"plain": None, "jupyter": 3, "silent": None}


def _logger() -> Logger:
    # Initialize against the active notebook/terminal stream; PYLOGFILE is optional.
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(name="nqs", use_ts_in_cmd=False)
    return _LOGGER


def _visible(level: int) -> bool:
    return not _SILENT and (_MAX_LEVEL is None or level <= _MAX_LEVEL)


@contextmanager
def quiet() -> Iterator[None]:
    """Suppress project messages and progress bars, restoring the previous policy."""
    global _SILENT
    previous, _SILENT = _SILENT, True
    try:
        yield
    finally:
        _SILENT = previous


@contextmanager
def display_context(display: str) -> Iterator[None]:
    """Set plain, jupyter (up to three tab levels), or silent output for one run."""
    global _MAX_LEVEL
    display = str(display).strip().lower()
    if display not in _DISPLAY_MAX_LEVEL:
        raise ValueError("display must be 'plain', 'jupyter', or 'silent'")
    previous, _MAX_LEVEL = _MAX_LEVEL, _DISPLAY_MAX_LEVEL[display]
    try:
        with quiet() if display == "silent" else nullcontext():
            yield
    finally:
        _MAX_LEVEL = previous


def log(message: str, level: int = 1, *, severity: str = "info") -> None:
    """Emit one general_log record with literal tabs at the requested depth."""
    if _visible(level):
        getattr(_logger(), severity)(message, lvl=level)


def title(message: str, level: int = 0) -> None:
    """Emit a stage banner using general_log."""
    if _visible(level):
        _logger().title(message, desired_size=72, fill="=", lvl=level)


def timing_summary(phases: dict[str, float], total: float, *, label: str, level: int = 1) -> None:
    """Report disjoint phase times, including uninstrumented overhead in the total."""
    if _visible(level + 1):
        phases = dict(phases)
        phases["Other / logging"] = max(0.0, total - sum(phases.values()))
        log_timing_summary(_logger(), phases, total_duration=total, title=label, lvl=level, duration_precision=3)


def progress_disabled() -> bool:
    """Silence bars in quiet runs or when NQS_PROGRESS=0; otherwise enable them."""
    return _SILENT or os.environ.get("NQS_PROGRESS") == "0"


@contextmanager
def section(title: str, level: int = 0) -> Iterator[None]:
    """Bracket a stage with its wall time; failures propagate and are labeled."""
    log(f"{title} | started", level)
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        log(f"{title} | failed after {time.perf_counter() - started:.3f} s", level, severity="error")
        raise
    else:
        log(f"{title} | finished in {time.perf_counter() - started:.3f} s", level)


# ------------------------------------------------------------------------------
# Interrupt-safe iteration
# ------------------------------------------------------------------------------


def interruptible(iterable: Iterable[T], on_interrupt: str | None = "interrupted") -> Iterator[T]:
    """Stop cleanly on KeyboardInterrupt and let callers return accumulated results."""
    iterator = iter(iterable)
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            return
        except KeyboardInterrupt:
            if on_interrupt:
                log(f"{on_interrupt} -- returning partial result", severity="warning")
            return
        try:
            yield value
        except KeyboardInterrupt:
            if on_interrupt:
                log(f"{on_interrupt} -- returning partial result", severity="warning")
            return


# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
