import time
from contextlib import contextmanager
from line_profiler import LineProfiler


@contextmanager
def simple_perf(desc: str = ""):
    s = time.perf_counter()
    yield
    e = time.perf_counter()
    print(f"{desc}: {(e - s) * 1000.0} ms")


def lprofile(func):
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lpw = lp(func)
        res = lpw(*args, **kwargs)
        lp.print_stats()
        return res

    return wrapper
