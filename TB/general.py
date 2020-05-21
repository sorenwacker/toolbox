from time import time, perf_counter
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print('[{}] done in {:.3f} s'.format(name, t1-t0))