import numpy as np
from bookirds.dual import solve
from datetime import datetime, timedelta


def bsplev_single(x, i, k, t, org_k=None):
    ## Enpoint support
    org_k = org_k or k  # original_k adds right point support for derivative recursion
    if x == t[0] and i == 0:
        return 1
    elif x == t[-1] and i >= len(t) - org_k - 1:
        return 1

    ## Recursion
    if k == 1:
        if t[i] <= x < t[i+1]:
            return 1
        return 0
    else:
        left, right = 0, 0
        if t[i] != t[i+k-1]:
            left = (x - t[i]) / (t[i+k-1] - t[i]) * bsplev_single(x, i, k-1, t)
        if t[i+1] != t[i+k]:
            right = (t[i+k] - x) / (t[i+k] - t[i+1]) * bsplev_single(x, i+1, k-1, t)
        return left + right


def bspldnev_single(x, i, k, t, n, og_k=None):
    if n == 0:
        return bsplev_single(x, i, k, t)
    elif k == 1 or n >= k:
        return 0

    og_k = og_k or k
    r, div1, div2 = 0, t[i+k-1] - t[i], t[i+k] - t[i+1]
    if isinstance(div1, timedelta):
        div1 = div1 / timedelta(days=1)
    if isinstance(div2, timedelta):
        div2 = div2 / timedelta(days=1)

    if n == 1:
        if div1 != 0:
            r += bsplev_single(x, i, k-1, t, og_k) / div1
        if div2 != 0:
            r -= bsplev_single(x, i+1, k-1, t, og_k) / div2
        r *= k - 1
    else:
        if div1 != 0:
            r += bspldnev_single(x, i, k-1, t, n-1, og_k) / div1
        if div2 != 0:
            r -= bspldnev_single(x, i+1, k-1, t, n-1, og_k) / div2
        r *= k - 1
    return r


class BSpline:
    def __init__(self, k, t):
        self.t = t
        self.k = k
        self.n = len(t) - k

    def __copy__(self):
        ret = BSpline(self.k, self.t)
        ret.c = getattr(self, "c", None)
        return ret

    def bsplev(self, x, i, otypes=["float64"]):
        """Evaluate `x` coordinates on the `i`th B-Spline. Returns 1d array."""
        func = np.vectorize(bsplev_single, excluded=["k", "t"], otypes=otypes)
        return func(x, i=i, k=self.k, t=self.t)

    def bspldnev(self, x, i, n, otypes=["float64"]):
        """
        Evaluate `x` coordinates on the `n`th  derivative of the `i`th B-Spline.
        Returns 1d array.
        """
        func = np.vectorize(bspldnev_single, excluded=["k", "t"], otypes=otypes)
        return func(x, i=i, k=self.k, t=self.t, n=n)

    def bsplmatrix(self, tau, left_n=0, right_n=0):
        """Evaluate `x` coordinates on all B-Splines. Returns 2d array."""
        B_ji = np.zeros(shape=(len(tau), self.n))
        for i in range(self.n):
            B_ji[0, i] = bspldnev_single(tau[0], i, self.k, self.t, left_n)
            B_ji[1:-1, i] = self.bsplev(tau[1:-1], i=i)
            B_ji[-1, i] = bspldnev_single(tau[-1], i, self.k, self.t, right_n)
        return B_ji

    def bsplsolve(self, tau, y, left_n, right_n):
        """Evaluate the B-Spline coeffs `c` that parametrise the pp."""
        if len(tau) != self.n:
            raise ValueError(f"`tau` must have length equal to pp dimension, "
                             f"`tau`: {len(tau)}, `n`: {self.n}")
        if len(tau) != len(y):
            raise ValueError(
                f"`tau` and `y` must have the same length, "
                f"`tau`: {len(tau)}, `y`: {len(y)}"
            )
        B_ji = self.bsplmatrix(tau, left_n, right_n)
        c = solve(B_ji, y[:, np.newaxis])
        self.c = c[:, 0]

    def ppev_single(self, x):
        """Evaluate `x` coordinate on the pp."""
        sum = 0
        for i, c_ in enumerate(self.c):
            sum += c_ * bsplev_single(x, i, self.k, self.t)
        return sum

    def ppev(self, x):
        """Evaluate `x` coordinates on the pp."""
        func = np.vectorize(self.ppev_single)
        return func(x)
