import numpy as np
from numba import jit
from math import log
from scipy import optimize as op


def arithmetic_first_depth_average(b, w, a, v):
    return (b[0] + a[0]) / 2


def weighted_first_depth_average(b, w, a, v):
    return (b[0] * v[0] + a[0] * w[0]) / (w[0] + v[0])


def weighted_max_first_depth_average(b, w, a, v, z):
    return (b[0] * min(v[0], z) + a[0] * min(w[0], z)) / (min(w[0], z) + min(v[0], z))


@jit(nopython=True)
def single_sided_ida(b, w, z):
    sum_w = w.cumsum()
    z = sum_w[-1] if z > sum_w[-1] else z
    n = len(np.where(z > sum_w)[0])
    if n == 0:
        return b[0]
    else:
        sum_bw = (b * w).cumsum()
        p = b[n] * (z - sum_w[n - 1]) + sum_bw[n - 1]
        return p / z


def intrinsic_depth_average(b, w, a, v, z):
    p_bida = single_sided_ida(b, w, z)
    p_aida = single_sided_ida(a, v, z)
    return (p_bida + p_aida) / 2


@jit(nopython=True)
def single_sided_mida(b, w, t):
    sum_w = w.cumsum()
    t = sum_w[-1] if t > sum_w[-1] else t
    n = len(np.where(t > sum_w)[0])
    if n == 0:
        return b[0]
    else:
        sum_bw = (b * w).cumsum()
        p = b[0] * sum_w[0]
        for j in range(1, n):
            p += b[j] * (sum_w[j] - sum_w[j-1])
            p += (sum_bw[j-1] - b[j] * sum_w[j-1]) * (log(sum_w[j]) - log(sum_w[j-1]))
        p += b[n] * (t - sum_w[n-1])
        p += (sum_bw[n-1] - b[n] * sum_w[n-1]) * (log(t) - log(sum_w[n-1]))
        return p / t


def mean_intrinsic_depth_average(b, w, a, v, t):
    p_mbida = single_sided_mida(b, w, t)
    p_maida = single_sided_mida(a, v, t)
    return (p_mbida + p_maida) / 2
