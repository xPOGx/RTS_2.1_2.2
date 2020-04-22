import random
import math
import time

import numpy as np

# Var 11
N = 256
omega = 1500
n = 10

def signal(n, omega, min_value=0, max_value=1):
    """Return signal function."""
    A = [min_value + (max_value - min_value) * random.random() for _ in range(n)]
    phi = [min_value + (max_value - min_value) * random.random() for _ in range(n)]

    def f(t):
        x = 0
        for i in range(n):
            x += A[i]*math.sin(omega/n*t*i + phi[i])
        return x
    return f

# Additional task (Table method)
def make_table(signal):
    start = time.time()
    N = len(signal)
    table = np.cos(2 * math.pi / N * np.linspace(0, N-1, N)) \
            -1j * np.sin(2 * math.pi / N * np.linspace(0, N-1, N))
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        indicies = np.linspace(0, N-1, N, dtype=np.int32) * p % N
        spectre[p] = np.dot(signal, table[indicies])
    print(f"Execution MakeTable time: {time.time() - start}")
    return spectre

range_min = 0
range_max = 1

x_gen = signal(n, omega, range_min, range_max)
x = [x_gen(i) for i in range(N)]

spectr = make_table(x)
