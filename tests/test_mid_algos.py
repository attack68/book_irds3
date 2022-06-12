import pytest
import numpy as np

import context
from modules.mid_market import intrinsic_depth_average, mean_intrinsic_depth_average


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