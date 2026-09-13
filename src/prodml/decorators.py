"""Custom decorators for prodml performance monitoring and logging."""

from collections.abc import Callable
from functools import wraps
import logging
import time
from typing import ParamSpec, TypeVar

logger = logging.getLogger("prodml.timed")

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to measure and log the execution time of a function.

    Logs execution time at INFO level in milliseconds.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            func_name = getattr(func, "__qualname__", func.__name__)
            logger.info("%s executed in %.2f ms", func_name, elapsed_ms)

    return wrapper
