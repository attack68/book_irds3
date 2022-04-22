import pytest
import numpy as np

import context
from modules.dual import Dual
from modules.curves import Swap, datetime, SolvedCurve


@pytest.fixture()
def market_rates():
    return {
        Swap(datetime(2022, 1, 1), 12 * 1, 12, 12): 1.000,
        Swap(datetime(2022, 1, 1), 12 * 2, 12, 12): 1.200,
        Swap(datetime(2022, 1, 1), 12 * 3, 12, 12): 1.300,
        Swap(datetime(2022, 1, 1), 12 * 4, 12, 12): 1.3500,
    }


@pytest.fixture()
def nodes(market_rates):
    return {
        datetime(2022, 1, 1): Dual(1, {"v0": 1}),
        datetime(2023, 1, 1): Dual(1, {"v1": 1}),
        datetime(2024, 1, 1): Dual(1, {"v2": 1}),
        datetime(2025, 1, 1): Dual(1, {"v3": 1}),
        datetime(2026, 1, 1): Dual(1, {"v4": 1}),

    }


def test_solved_curve_reprice(market_rates, nodes):
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear", obj_rates=list(market_rates.values()),
        swaps=list(market_rates.keys()), algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    assertions = [
        abs(s_cv.swaps[i].rate(s_cv).real - s_cv.obj_rates[i]) < 1e-6
        for i in range(len(market_rates))
    ]
    assert all(assertions)


def test_solved_curve_overspecified(market_rates):
    nodes = {
        datetime(2022, 1, 1): Dual(1.00, {"v0": 1}),
        datetime(2024, 1, 1): Dual(1.00, {"v1": 1}),
        datetime(2026, 1, 1): Dual(1.00, {"v2": 1})
    }
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear", obj_rates=list(market_rates.values()),
        swaps=list(market_rates.keys()), algorithm="levenberg_marquardt"
    )
    ret = s_cv.iterate()
    assert "tolerance reached" in ret
    assert "7 iterations" in ret
    assert s_cv.f.real > 1e-3