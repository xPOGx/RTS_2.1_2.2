import matplotlib.pyplot as plt
import math, time
from random import *
import numpy as np

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


def fft(signal):
    signal = np.asanyarray(signal, dtype=float)
    counts = len(signal)
    if counts <= 2:
        return dpf(signal)
    else:
        signal_even = fft(signal[::2])
        signal_odd = fft(signal[1::2])
        terms = np.exp(-2j * np.pi * np.arange(counts) / counts)
        try:
            return np.concatenate([signal_even + terms[:counts // 2] * signal_odd, signal_even + terms[counts // 2:] * signal_odd])
        except:
            return 0
timeFFT = []
timeNPFFT = []
t = np.linspace(0, counts, counts)
for i in range(counts):
    start = time.time()
    Arraray_new = arr_gen(harmonics, 0, 5)
    alpha = arr_gen(harmonics, 0, 5)
    Freg = freg_gen(frequency, harmonics)
    x = xt_gen(i+1, harmonics, Arraray_new, Freg, alpha)
    x_fft = fft(x)
    t = np.linspace(0, i+1, counts)
    x_fft_real = x_fft.real
    x_fft_img = x_fft.imag
    tim = time.time() - start
    timeFFT.append(tim)

    start = time.time()
    Arraray_new = arr_gen(harmonics, 0, 5)
    alpha = arr_gen(harmonics, 0, 5)
    Freg = freg_gen(frequency, harmonics)
    x = xt_gen(i + 1, harmonics, Arraray_new, Freg, alpha)
    sp = np.fft.fft(x)
    tim = time.time() - start
    timeNPFFT.append(tim)

d = []
for i in range(len(timeNPFFT)):
    o = timeNPFFT[i] - timeFFT[i]
    d.append(o)

d = np.asarray(d)
plt.title("Відхилення FFT від numpy FFT по часу для коного N")
plt.plot(t, d.real, 'b', t, d.imag, 'r')
plt.show()
