import pytest

import context
from modules.swaptions import (
    swaption_price,
    swaption_implied_vol,
    swaption_delta,
    swaption_greeks,
)


@pytest.mark.parametrize("dist, expected", [
    ("normal", 0.381616106359),
    ("log", 0.5357485125062)
])
def test_swaption_price_p(dist, expected):
    result = swaption_price(1.325, 1.5, 5, 52, "payer", dist)
    assert abs(result - expected) < 1e-9


@pytest.mark.parametrize("dist, expected", [
    ("normal", 0.300000025495),
    ("log", 0.3048195840646)
])
def test_swaption_price_r(dist, expected):
    result = swaption_price(2.80, 3.10, 0.25, 13, "receiver", dist)
    assert abs(result - expected) < 1e-9


def test_swaption_implied_vol():
    result = swaption_implied_vol(0.15, 1.3, 1.3, 1.0, 20, "straddle", "normal")
    expected = 18.799712059
    assert abs(result - expected) < 1e-9


def test_swaption_delta():
    result = swaption_delta(1.1, 1.3, 1.0, 50, "payer", "normal")
    expected = 0.34457828294098825
    assert abs(result - expected) < 1e-9


@pytest.mark.parametrize("args, expected", [
    ((1.10, 1.30, 1.0, 50, "payer", "normal"), [0.34458, 0.00737, 0.00368, -0.00037]),
    ((1.10, 1.10, 1.0, 50, "straddle", "normal"), [0.00000, 0.01596, 0.00798, -0.00079])
])
def test_swaption_greeks(args, expected):
    result = swaption_greeks(*args)
    for i, value in enumerate(result):
        assert abs(expected[i] - value) < 1e-5
