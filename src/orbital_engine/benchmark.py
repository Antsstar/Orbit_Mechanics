"""
Microbenchmark timing core.

The engine's stated purpose is to make the cost of a modelling assumption measurable alongside its
accuracy, so timing is engine functionality rather than a script bolted on top. This module holds
only the measurement primitive; scenario definitions live in `scenarios.py` and the drivers live in
`benchmarks/`.

**Why minimum-of-batches rather than mean.** Timing noise on a desktop OS is one-sided: scheduler
preemption, interrupts and frequency scaling can only ever make a run slower, never faster. The mean
therefore estimates "typical machine load during the run", which is not the quantity of interest.
The minimum batch mean estimates the cost of the work itself, and is the standard choice for
microbenchmarks for that reason. The spread between minimum and median is reported so a noisy
measurement is visible rather than silently averaged in.

Batches, not individual calls: at sub-microsecond resolution `perf_counter` overhead is a
significant fraction of the measurement, so a batch of `inner` calls is timed as a unit and divided.
"""
from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = ["Measurement", "measure"]


@dataclass(frozen=True)
class Measurement:
    """Result of a timing run. All times in microseconds per call."""

    best: float
    median: float
    worst: float
    batches: int
    inner: int

    @property
    def noise_ratio(self) -> float:
        """
        Median divided by best. 1.0 is a perfectly quiet machine; above roughly 1.3 the measurement
        is contended enough that small differences between runs should not be trusted.
        """
        return self.median / self.best if self.best > 0.0 else float("nan")

    def __str__(self) -> str:
        return f"{self.best:9.2f} us  (median {self.median:9.2f}, noise x{self.noise_ratio:.2f})"


def measure(
    fn: Callable[[], object],
    *,
    batches: int = 7,
    inner: int = 50,
    warmup: int = 10,
    setup: Optional[Callable[[], object]] = None,
) -> Measurement:
    """
    Time `fn` and return microseconds per call.

    `warmup` calls run first and are discarded, which matters for any path that compiles or caches
    on first use - a JIT-compiled kernel would otherwise charge its entire compilation to the first
    batch. `setup`, if given, runs before each batch and is not timed; use it to reset state that
    `fn` mutates, so batch N does not start from the state batch N-1 left behind.

    The garbage collector is disabled for the duration and restored afterwards, so a collection
    triggered by unrelated allocation cannot land inside a timed batch.
    """
    if batches < 1 or inner < 1:
        raise ValueError(f"batches and inner must both be >= 1, got {batches} and {inner}")

    for _ in range(warmup):
        fn()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        per_call: list[float] = []
        for _ in range(batches):
            if setup is not None:
                setup()
            start = time.perf_counter()
            for _ in range(inner):
                fn()
            elapsed = time.perf_counter() - start
            per_call.append(elapsed / inner * 1e6)
    finally:
        if gc_was_enabled:
            gc.enable()

    return Measurement(
        best=min(per_call),
        median=statistics.median(per_call),
        worst=max(per_call),
        batches=batches,
        inner=inner,
    )
