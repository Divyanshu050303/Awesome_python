"""
A small pipeline framework built from closures, decorators and generators.

The @pipeline_step decorator wraps any function so that it logs its own
execution, times itself, catches exceptions and reports success or failure.
It is generator-aware: a step that yields stays lazy after decoration.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------------
# Result recording
# --------------------------------------------------------------------------

@dataclass
class StepResult:
    """One execution of one step."""
    name: str
    state: str                        # "ok" | "failed" | "partial"
    seconds: float
    items: int | None = None          # only meaningful for streaming steps
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    @property
    def status(self) -> str:
        return self.state.upper()


RUN_LOG: list[StepResult] = []


def _record(name, state, seconds, items=None, error=None) -> StepResult:
    result = StepResult(name, state, seconds, items, error)
    RUN_LOG.append(result)
    return result


def reset_log() -> None:
    RUN_LOG.clear()


def summary() -> str:
    """Render RUN_LOG as a plain-text table."""
    if not RUN_LOG:
        return "(no steps run)"

    width = max(len(r.name) for r in RUN_LOG)
    lines = [f"{'step'.ljust(width)}  {'status':<7} {'time':>9}  items"]
    lines.append("-" * (width + 28))
    for r in RUN_LOG:
        items = "-" if r.items is None else f"{r.items:,}"
        lines.append(
            f"{r.name.ljust(width)}  {r.status:<7} {r.seconds:>8.3f}s  {items}"
        )
    longest = max(r.seconds for r in RUN_LOG)
    lines.append("-" * (width + 28))
    lines.append(f"{'wall'.ljust(width)}  {'':<7} {longest:>8.3f}s")
    lines.append("(streaming steps run concurrently, so their times overlap)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The decorator
# --------------------------------------------------------------------------

def pipeline_step(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    on_error: str = "raise",
) -> Callable:
    """
    Wrap a function as an instrumented pipeline step.

    Usable bare or with arguments:

        @pipeline_step
        def load(): ...

        @pipeline_step(name="load raw CSV", on_error="skip")
        def load(): ...

    on_error:
        "raise" - log the failure, then re-raise (default)
        "skip"  - log the failure and return None / end the stream
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")

    def decorator(func: Callable) -> Callable:
        step_name = name or func.__name__.replace("_", " ")

        # A generator function needs its own wrapper. If we used the plain
        # wrapper below, calling func() would hand back a generator object
        # instantly and we would be timing the *creation* of the stream
        # rather than the work. So we iterate inside the wrapper and re-yield,
        # which keeps the wrapper lazy and makes the timing honest.
        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger.info("-> %s (streaming)", step_name)
                start = time.perf_counter()
                count = 0
                done = False
                try:
                    for item in func(*args, **kwargs):
                        count += 1
                        yield item
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _record(step_name, "failed", elapsed, count, exc)
                    done = True
                    logger.error(
                        "xx %s failed after %d items in %.3fs: %s: %s",
                        step_name, count, elapsed, type(exc).__name__, exc,
                    )
                    if on_error == "raise":
                        raise
                    return          # ends the stream cleanly
                else:
                    elapsed = time.perf_counter() - start
                    _record(step_name, "ok", elapsed, count)
                    done = True
                    logger.info(
                        "ok %s yielded %d items in %.3fs",
                        step_name, count, elapsed,
                    )
                finally:
                    # If a downstream step died, this generator is closed
                    # while still suspended at its yield. GeneratorExit is a
                    # BaseException, so the except clause above misses it and
                    # the step would silently vanish from the log.
                    if not done:
                        elapsed = time.perf_counter() - start
                        _record(step_name, "partial", elapsed, count)
                        logger.warning(
                            "~~ %s abandoned after %d items", step_name, count
                        )

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger.info("-> %s", step_name)
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _record(step_name, "failed", elapsed, error=exc)
                    logger.error(
                        "xx %s failed in %.3fs: %s: %s",
                        step_name, elapsed, type(exc).__name__, exc,
                    )
                    if on_error == "raise":
                        raise
                    return None
                else:
                    elapsed = time.perf_counter() - start
                    _record(step_name, "ok", elapsed)
                    logger.info("ok %s in %.3fs", step_name, elapsed)
                    return result

        wrapper.is_pipeline_step = True      # type: ignore[attr-defined]
        wrapper.step_name = step_name        # type: ignore[attr-defined]
        return wrapper

    # Bare @pipeline_step -> _func is the function.
    # @pipeline_step(...)  -> _func is None, return the decorator itself.
    if _func is None:
        return decorator
    return decorator(_func)


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------

class Pipeline:
    """Runs steps in order, feeding each one the previous one's output."""

    def __init__(self, *steps: Callable, name: str = "pipeline"):
        for step in steps:
            if not getattr(step, "is_pipeline_step", False):
                raise TypeError(
                    f"{step.__name__!r} is not decorated with @pipeline_step"
                )
        self.steps = list(steps)
        self.name = name

    def __or__(self, step: Callable) -> "Pipeline":
        """Allows: pipe = Pipeline(load) | clean | transform"""
        return Pipeline(*self.steps, step, name=self.name)

    def run(self, seed: Any = None) -> Any:
        logger.info("=== %s: %d steps ===", self.name, len(self.steps))
        start = time.perf_counter()
        data = seed
        stages: list = []
        try:
            for i, step in enumerate(self.steps):
                data = step() if (i == 0 and seed is None) else step(data)
                if inspect.isgenerator(data):
                    stages.append(data)
        finally:
            # If a step failed mid-stream, the upstream generators are still
            # suspended. Close them now so they log their partial counts
            # here, rather than whenever the GC happens to get to them.
            for stage in reversed(stages):
                if stage is not data:
                    stage.close()
        logger.info(
            "=== %s finished in %.3fs ===", self.name, time.perf_counter() - start
        )
        return data


# --------------------------------------------------------------------------
# Generator helpers
# --------------------------------------------------------------------------

def process_records(records: Iterable, transform: Callable) -> Iterator:
    """Apply transform to each record, one at a time."""
    for record in records:
        yield transform(record)


def batched(records: Iterable, size: int) -> Iterator[list]:
    """Group a stream into lists of `size` (the last batch may be shorter)."""
    batch: list = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def tap(records: Iterable, every: int = 10_000, label: str = "progress") -> Iterator:
    """Pass records through untouched, logging every N items."""
    for i, record in enumerate(records, 1):
        if i % every == 0:
            logger.debug("%s: %s", label, f"{i:,}")
        yield record


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )