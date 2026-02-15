import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timed(name: str, out_ms: Dict[str, float]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        out_ms[name] = (time.perf_counter() - t0) * 1000.0
