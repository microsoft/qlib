import time
from contextlib import contextmanager
from typing import Callable, Generator

from line_profiler import LineProfiler


@contextmanager
def simple_perf(desc: str = "", out_path: str = None) -> Generator[None, None, None]:
    s = time.perf_counter()
    yield
    e = time.perf_counter()
    msg = f"{desc}: {(e - s) * 1000.0:.4f} ms"
    
    if out_path is not None:
        with open(out_path, "a") as fstream:
            fstream.write(msg + "\n")        
    else:
        print(msg)


def lprofile(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lpw = lp(func)
        res = lpw(*args, **kwargs)
        lp.print_stats()
        return res

    return wrapper
