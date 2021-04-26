import time
import numba
import numpy as np


def foo_slow(a, b):
    c = np.zeros_like(a)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):

            val = 0

            for k in range(a.shape[1]):
                val += a[i, k] * b[k, j]

            c[i, j] = val

    return c


@numba.njit
def foo_fast(a, b):
    c = np.zeros_like(a)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):

            val = 0

            for k in range(a.shape[1]):
                val += a[i, k] * b[k, j]

            c[i, j] = val

    return c


l = 100
a = np.random.random((l, l))
b = np.random.random((l, l))


for i in range(10):
    t0 = time.time()
    c = foo_fast(a, b)
    t1 = time.time()
    print(f"Time (fast): {t1 - t0} sec")

for i in range(10):
    t0 = time.time()
    c = np.dot(a, b)
    t1 = time.time()
    print(f"Time (numpy): {t1 - t0} sec")


for i in range(10):
    t0 = time.time()
    c = foo_slow(a, b)
    t1 = time.time()
    print(f"Time (slow): {t1 - t0} sec")
