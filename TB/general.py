from time import time, perf_counter
from contextlib import contextmanager
import numpy as np


@contextmanager
def timer(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print("[{}] done in {:.3f} s".format(name, t1 - t0))


def intersects(group_a, group_b):
    intersect = list(np.intersect1d(group_a, group_b))
    only_a = list(group_a)
    only_b = list(group_b)
    for el in intersect:
        only_a.remove(el)
        only_b.remove(el)
    output = dict(intersect=intersect, only_a=only_a, only_b=only_b)
    return output
