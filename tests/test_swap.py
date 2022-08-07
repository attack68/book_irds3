import pytest
import numpy as np

import context
from bookirds.dual import Dual
from bookirds.curves import Swap, datetime, SolvedCurve, Curve, Portfolio, Swap2


@pytest.fixture()
def market_rates():
    return {
        Swap(datetime(2022, 1, 1), 12 * 1, 12, 12): 1.000,
        Swap(datetime(2022, 1, 1), 12 * 2, 12, 12): 1.200,
        Swap(datetime(2022, 1, 1), 12 * 3, 12, 12): 1.300,
        Swap(datetime(2022, 1, 1), 12 * 4, 12, 12): 1.3500,
    }


@pytest.fixture()
def nodes():
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


def test_basic_scaling(nodes):
    curve = Curve(nodes=nodes, interpolation="log_linear")
    swap = Swap(datetime(2022, 1, 1), 2*12, 12, 12, fixed_rate=1.00)
    assert swap.analytic_delta(curve).real == 200
    assert swap.rate(curve).real == 0
    assert swap.npv(curve).real == -20000

    curve = Curve(interpolation="log_linear", nodes={
        datetime(2022, 1, 1): 1.0, datetime(2032, 1, 1): 0.90
    })
    assert abs(swap.analytic_delta(curve).real - 196.868462959) < 1e-7
    assert abs(swap.rate(curve).real - 1.058592008) < 1e-7
    assert abs(swap.npv(curve).real - 1153.49187188) < 1e-7


def test_swap_creation_notional(nodes):
    curve = Curve(nodes=nodes, interpolation="log_linear")
    swap = Swap(datetime(2022, 1, 1), 2 * 12, 12, 12, fixed_rate=1.00, notional=100e6)
    assert swap.npv(curve).real == -2000000


def test_portfolio_npv(nodes):
    crv = Curve(nodes=nodes, interpolation="log_linear")
    swap1 = Swap(datetime(2022, 1, 1), 2 * 12, 12, 12, fixed_rate=1.00, notional=100e6)
    swap2 = Swap(datetime(2022, 1, 1), 3 * 12, 12, 12, fixed_rate=2.00, notional=10e6)
    swap3 = Swap(datetime(2022, 1, 1), 4 * 12, 12, 12, fixed_rate=3.00, notional=1e6)
    portfolio = Portfolio(objects=[swap1, swap2, swap3])
    assert portfolio.npv(crv) == swap1.npv(crv) + swap2.npv(crv) + swap3.npv(crv)


def test_swap2(market_rates, nodes):
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear", obj_rates=list(market_rates.values()),
        swaps=list(market_rates.keys()), algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    swap = Swap(datetime(2022, 1, 1), 2 * 12, 12, 12, fixed_rate=1.00, notional=100e6)
    swap2 = Swap2(datetime(2022, 1, 1), 2 * 12, 12, 12, fixed_rate=1.00, notional=100e6)

    assert abs(swap.rate(s_cv) - swap2.rate(s_cv)) < 1e-8
    assert abs(swap.npv(s_cv) - swap2.npv(s_cv)) < 1e-8
    dpdz, dpds = swap2.risk_fwd_zero_rates(s_cv)
    assert abs(swap2.analytic_delta(s_cv) - dpdz.sum()) < 1e-4
    assert abs(swap2.npv(s_cv) * 2 / 10000 + dpds.sum()) < 1


def test_curve_grad_r_v(market_rates, nodes):
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear", obj_rates=list(market_rates.values()),
        swaps=list(market_rates.keys()), algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    expected = np.array([
        [-0.98030, -0.96673, -0.95237, -0.93826],
        [0, -0.96289, -0.94859, -0.93453],
        [0, 0, -0.95020, -0.93612],
        [0, 0, 0, -0.93359]
    ])
    assert np.all(np.abs(s_cv.grad_r_v - expected) < 1e-4)


def test_curve_grad_v_r(market_rates, nodes):
    s_cv = SolvedCurve(
        nodes=nodes, interpolation="log_linear", obj_rates=list(market_rates.values()),
        swaps=list(market_rates.keys()), algorithm="levenberg_marquardt"
    )
    s_cv.iterate()
    result = np.matmul(s_cv.grad_v_r, s_cv.grad_r_v)
    expected = np.eye(4)
    assert np.all(np.abs(result - expected) < 1e-4)