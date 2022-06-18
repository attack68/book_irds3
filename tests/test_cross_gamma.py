import pytest
import numpy as np

import context
from modules.dual import Dual
from modules.curves import Swap, datetime, SolvedCurve, Curve, Portfolio, Swap2


def test_cross_gamma():
    nodes = {
        datetime(2022, 1, 1): Dual(1, {"v0": 1}),
        datetime(2023, 1, 1): Dual(1, {"v1": 1}),
        datetime(2024, 1, 1): Dual(1, {"v2": 1}),
        datetime(2025, 1, 1): Dual(1, {"v3": 1}),
        datetime(2026, 1, 1): Dual(1, {"v4": 1}),
        datetime(2027, 1, 1): Dual(1, {"v5": 1}),
    }
    swaps = {
        Swap(datetime(2022, 1, 1), 12 * 1, 12, 12): 1.00,
        Swap(datetime(2022, 1, 1), 12 * 2, 12, 12): 1.10,
        Swap(datetime(2022, 1, 1), 12 * 3, 12, 12): 1.20,
        Swap(datetime(2022, 1, 1), 12 * 4, 12, 12): 1.25,
        Swap(datetime(2022, 1, 1), 12 * 5, 12, 12): 1.27,
    }
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear",
        obj_rates=list(swaps.values()),
        swaps=list(swaps.keys()),
        algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    portfolio = Portfolio([
        Swap2(datetime(2022, 1, 1), 5 * 12, 12, 12, fixed_rate=25.0, notional=-8.3e8),
    ])
    ss, sz, zz = portfolio.cross_gamma(s_cv)
    exp_ss = np.array([
        [18.63, 7.36, 5.48, 3.62, 1.80],
        [7.36, 14.71, 5.47, 3.61, 1.79],
        [5.48, 5.48, 10.95, 3.61, 1.79],
        [3.62, 3.61, 3.61, 7.20, 1.79],
        [1.80, 1.79, 1.79, 1.79, 3.58],
    ])
    exp_sz = np.array([
        [8.13, 8.04, 7.95, 7.81, 7.71],
        [0, 8.02, 7.93, 7.80, 7.69],
        [0, 0, 7.94, 7.80, 7.70],
        [0, 0, 0, 7.78, 7.68],
        [0, 0, 0, 0, 7.68]
    ])
    exp_zz = np.zeros((5, 5))

    assert np.all(np.abs(zz - exp_zz) < 1e-1)
    assert np.all(np.abs(ss - exp_ss) < 1e-1)
    assert np.all(np.abs(sz - exp_sz) < 1e-1)