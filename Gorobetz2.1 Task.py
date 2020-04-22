import matplotlib.pyplot as plt
import math
from random import *
import numpy as np
import time

# REGION
# Gorobetz2.1
start = time.time()


# Var 9
counts = 256
frequency = 2000
harmonics = 8


def freg_gen(frequency, harmonics):
    Freg = []
    step = frequency / harmonics
    for i in range(harmonics):
        Freg.append(frequency - step * i)
    return Freg


def expectation(x, counts):
    mx = 0.0
    for i in range(counts):
        mx += x[i]
    return mx / counts


def dispersion(x, counts, mx):
    dx = 0.0
    for i in range(counts):
        dx += math.pow(x[i] - mx, 2)
    return dx / (counts - 1)


def xt_gen(counts, harmonics, Arraray_new, Freg, alpha):
    x = [0] * counts
    for j in range(counts):
        for i in range(harmonics):
            x[j] += Arraray_new[i] * math.sin(Freg[i] * j + alpha[i])
    return x


def arr_gen(harmonics, min, max):
    arr = [0] * harmonics
    for i in range(harmonics):
        arr[i] = randint(min, max)
    return arr


def dpf(signal):
    harmonics = len(signal)
    p = np.arange(harmonics)
    k = p.reshape((harmonics, 1))
    frequency = np.exp(-2j * np.pi * p * k / harmonics)
    return np.dot(frequency, signal)

# 2.1
Arraray_new = arr_gen(harmonics, 0, 5)
alpha = arr_gen(harmonics, 0, 5)
Freg = freg_gen(frequency, harmonics)
x = xt_gen(counts, harmonics, Arraray_new, Freg, alpha)
x_dpf = dpf(x)


print(f"Execution time: {time.time() - start}")
# ENDREGION
