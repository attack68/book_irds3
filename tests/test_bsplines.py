import pytest
import numpy as np

import context
from modules.bsplines import *


@pytest.fixture()
def t():
    return np.array([1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4])


@pytest.fixture()
def x():
    return np.linspace(1, 4, 7)


@pytest.mark.parametrize("i, expected", [
    (0, np.array([1., 0.125, 0., 0., 0., 0., 0.])),
    (1, np.array([0., 0.375, 0., 0., 0., 0., 0.])),
    (2, np.array([0., 0.375, 0., 0., 0., 0., 0.])),
    (3, np.array([0., 0.125, 1., 0.125, 0., 0., 0.])),
    (4, np.array([0., 0., 0., 0.59375, 0.25, 0.03125, 0.])),
    (5, np.array([0., 0., 0., 0.25, 0.5, 0.25, 0.])),
    (6, np.array([0., 0., 0., 0.03125, 0.25, 0.59375, 0.])),
    (7, np.array([0., 0., 0., 0., 0., 0.125, 1.])),
])
def test_individual_bsplines(t, x, i, expected):
    bs = BSpline(k=4, t=t)
    result = bs.bsplev(x, i=i)
    assert (result == expected).all()


@pytest.mark.parametrize("i, expected", [
    (0, np.array([-3., -0.75, 0., 0., 0., 0., 0.])),
    (1, np.array([3., -0.75, 0., 0., 0., 0., 0.])),
    (2, np.array([0., 0.75, 0., 0., 0., 0., 0.])),
    (3, np.array([0., 0.75, -3., -0.75, 0., 0., 0.])),
    (4, np.array([0., 0., 3., -0.1875, -0.75, -0.1875, 0.])),
    (5, np.array([0., 0., 0., 0.75, 0., -0.75, 0.])),
    (6, np.array([0., 0., 0., 0.1875, 0.75, 0.1875, -3.])),
    (7, np.array([0., 0., 0., 0., 0., 0.75, 3.])),
])
def test_first_derivative_endpoint_support(t, x, i, expected):
    bs = BSpline(k=4, t=t)
    result = bs.bspldnev(x, i=i, n=1)
    assert (result == expected).all()


@pytest.mark.parametrize("i, expected", [
    (0, np.array([6., 3., 0., 0., 0., 0., 0.])),
    (1, np.array([-12., -3., 0., 0., 0., 0., 0.])),
    (2, np.array([6., -3., 0., 0., 0., 0., 0.])),
    (3, np.array([0., 3., 6., 3., 0., 0., 0.])),
    (4, np.array([0., 0., -9., -3.75, 1.5, 0.75, 0.])),
    (5, np.array([0., 0., 3., 0., -3., 0., 3.])),
    (6, np.array([0., 0., 0., 0.75, 1.5, -3.75, -9.])),
    (7, np.array([0., 0., 0., 0., 0., 3., 6.])),
])
def test_second_derivative_endpoint_support(t, x, i, expected):
    bs = BSpline(k=4, t=t)
    result = bs.bspldnev(x, i=i, n=2)
    assert (result == expected).all()


@pytest.mark.parametrize("i, expected", [
    (0, np.array([-6., -6., 0., 0., 0., 0., 0.])),
    (1, np.array([18., 18., 0., 0., 0., 0., 0.])),
    (2, np.array([-18., -18., 0., 0., 0., 0., 0.])),
    (3, np.array([6., 6., -6., -6., 0., 0., 0.])),
    (4, np.array([0., 0., 10.5, 10.5, -1.5, -1.5, -1.5])),
    (5, np.array([0., 0., -6., -6., 6., 6., 6.])),
    (6, np.array([0., 0., 1.5, 1.5, -10.5, -10.5, -10.5])),
    (7, np.array([0., 0., 0., 0., 6., 6., 6.])),
])
def test_third_derivative_endpoint_support(t, x, i, expected):
    bs = BSpline(k=4, t=t)
    result = bs.bspldnev(x, i=i, n=3)
    assert (result == expected).all()


def test_fourth_derivative_endpoint_support(t, x):
    bs = BSpline(k=4, t=t)
    expected = np.array([0., 0., 0., 0., 0., 0., 0.])
    for i in range(8):
        test = bs.bspldnev(x, i=i, n=4) == expected
        assert test.all()
