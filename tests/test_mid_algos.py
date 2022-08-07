import pytest
from pandas import DataFrame

import context
from bookirds.mid_market import intrinsic_depth_average, mean_intrinsic_depth_average
from bookirds.curves import *


@pytest.fixture()
def orderbook():
    return {
        "b": np.array([5, 4, 3, 2, 1]),
        "w": np.array([10, 5, 10, 5, 10]),
        "a": np.array([6, 7, 8, 9, 10]),
        "v": np.array([5, 10, 15, 10, 5]),
    }


def test_ida(orderbook):
    b, w, a, v = orderbook["b"],  orderbook["w"], orderbook["a"], orderbook["v"]

    assert intrinsic_depth_average(b, w, a, v, 1) == 5.5
    assert intrinsic_depth_average(b, w, a, v, 5) == 5.5
    assert intrinsic_depth_average(b, w, a, v, 10) == (5 + 6.5) / 2
    assert intrinsic_depth_average(b, w, a, v, 40) == (3 + 7.75) / 2
    assert intrinsic_depth_average(b, w, a, v, 1e6) == (3 + 8) /2


def test_mida(orderbook):
    b, w, a, v = orderbook["b"], orderbook["w"], orderbook["a"], orderbook["v"]

    assert mean_intrinsic_depth_average(b, w, a, v, 1) == 5.5
    assert mean_intrinsic_depth_average(b, w, a, v, 5) == 5.5
    assert mean_intrinsic_depth_average(b, w, a, v, 10) > 5.5
    ida = intrinsic_depth_average(b, w, a, v, 10)
    assert mean_intrinsic_depth_average(b, w, a, v, 10) < ida
    assert abs(mean_intrinsic_depth_average(b, w, a, v, 40) - 5.5651978329242) < 1e-10


def test_model_margin():
    nodes = {
        datetime(2022, 1, 1): Dual(1, {"v0": 1}),
        datetime(2024, 1, 1): Dual(1, {"v1": 1}),
        datetime(2027, 1, 1): Dual(1, {"v2": 1}),
        datetime(2032, 1, 1): Dual(1, {"v3": 1}),
        datetime(2052, 1, 1): Dual(1, {"v4": 1})
    }
    swaps = {
        Swap(datetime(2022, 1, 1), 12 * 2, 12, 12): 1.635,
        Swap(datetime(2022, 1, 1), 12 * 5, 12, 12): 1.885,
        Swap(datetime(2022, 1, 1), 12 * 10, 12, 12): 1.930,
        Swap(datetime(2022, 1, 1), 12 * 30, 12, 12): 1.980,
    }
    s_cv = SolvedCurve(
        nodes=nodes,
        swaps=list(swaps.keys()),
        obj_rates=list(swaps.values()),
        interpolation="log_linear",
        algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    pf = Portfolio([
        Swap(datetime(2027, 1, 1), 12 * 5, 12, 12, fixed_rate=1.9797, notional=100e6)
    ])
    df = DataFrame({
        "2Y": [1, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, 0],
        "5Y": [0, 1, 0, 0, 1, 0, 0, -1, -1, 0, 2, -1],
        "10Y": [0, 0, 1, 0, 0, 1, 0, 1, 0, -1, -1, 2],
        "30Y": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, -1],
        "margin": [0.4, 0.5, 0.5, 0.6, 0.25, 0.45, 0.85, 0.25, 0.7, 0.55, 0.5, 0.6],
    },
        index=["2Y", "5Y", "10Y", "30Y", "2s5s", "2s10s", "2s30s", "5s10s", "5s30s",
              "10s30s", "2s5s10s", "5s10s30s"]
    )
    c = df["margin"].to_numpy()[:, np.newaxis]
    A = df[["2Y", "5Y", "10Y", "30Y"]].to_numpy()
    ret = pf.model_margin(c, A.T, pf.risk(s_cv))
    assert ret[2].success
    assert abs(ret[0] - 33405.031106437) <= 1e-5
