import pytest
import numpy as np
from scipy.linalg import lu

import context
from modules.dual import Dual, solve, plu_decomp


@pytest.fixture()
def z_1():
    return Dual(1, {"z1": 1})


@pytest.fixture()
def z_2():
    return Dual(99, {"z1": 1, "z2": 99})


@pytest.fixture()
def A():
    return np.random.randn(25).reshape(5, 5)


@pytest.fixture()
def A_sparse():
    return np.array([
        [24, -36, 12, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.25, 0.583333333333, 0.1666666666, 0, 0, 0, 0, 0],
        [0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0, 0, 0],
        [0, 0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0, 0],
        [0, 0, 0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0],
        [0, 0, 0, 0, 0, 0.1666666666, 0.583333333333, 0.25, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 12, -36, 24],
    ])


@pytest.fixture()
def A_sparse2():
    return np.array([
        [0.6666666666, 0.1666666666, 0, 0],
        [0.1666666666, 0.583333333333, 0.25, 0],
        [0, 0, 0, 1],
        [0, 12, -36, 24],
    ])


@pytest.fixture()
def b():
    return np.random.randn(5).reshape(5, 1)


def test_neg(z_2):
    result = -z_2
    expected = Dual(-99, {"z1": -1, "z2": -99})
    assert result == expected


def test_eq_ne(z_1):
    assert z_1 != 10
    assert z_1 != Dual(1, {"z2": 10})
    assert z_1 != Dual(2, {"z1": 1})
    assert z_1 == Dual(1, {"z1": 1})


def test_conjugate(z_1, z_2):
    result = z_1.conjugate()
    expected = Dual(1, {"z1": -1})
    assert result == expected

    result = z_2.conjugate()
    expected = Dual(99, {"z1": -1, "z2": -99})
    assert result == expected


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_dual_immutable(z_1, z_2, op):
    _ = getattr(z_1, op)(z_2)
    assert z_1 == Dual(1, {"z1": 1})
    assert z_2 == Dual(99, {"z1": 1, "z2": 99})


def test_dual_add(z_1, z_2):
    result = z_1 + z_2
    expected = Dual(100, {"z1": 2, "z2": 99})
    assert result == expected


def test_dual_sub(z_1, z_2):
    result = z_1 - z_2
    expected = Dual(-98, {"z1": 0, "z2": -99})
    assert result == expected


def test_dual_sub_float(z_1):
    result = z_1 - 10
    expected = Dual(-9, {"z1": 1})
    assert result == expected


def test_dual_rsub_float(z_1):
    result = 10 - z_1
    expected = Dual(9, {"z1": -1})
    assert result == expected


def test_dual_mul(z_1, z_2):
    result = z_1 * z_2
    expected = Dual(99, {"z1": 100, "z2": 99})
    assert result == expected


def test_dual_div(z_1, z_2):
    result = z_1 / z_2
    expected = Dual(99 / 99**2, {"z1": 98 / 99**2, "z2": -99 / 99**2})
    assert result == expected


def test_dual_rdiv(z_2):
    result = 10 / z_2
    expected = Dual(10, {}) / z_2
    assert result == expected


def test_sparse_plu():
    A = np.array([
        [0, 1, 0],
        [0, 1, 1],
        [2, 3, 0],
    ])
    P, L, U = plu_decomp(A)

    diff = np.matmul(P, A) - np.matmul(L, U)
    assertions = [abs(diff[i, j]) < 1e-10 for i in range(3) for j in range(3)]
    assert all(assertions)


@pytest.mark.parametrize("sparse", [False, True])
def test_plu(A, A_sparse, sparse):
    if sparse:
        A = A_sparse
    n = A.shape[0]

    P, L, U = plu_decomp(A)

    assertions = [abs(L[i, j]) < 1e-10 for i in range(n) for j in range(i+1, n)]
    assert all(assertions)

    assertions = [abs(U[i, j]) < 1e-10 for j in range(n) for i in range(j+1, n)]
    assert all(assertions)

    diff = np.matmul(P, A) - np.matmul(L, U)
    assertions = [abs(diff[i, j]) < 1e-10 for i in range(n) for j in range(n)]
    assert all(assertions)


def test_solve(A, b):
    x = solve(A, b)
    x_np = np.linalg.solve(A, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A.shape[0])]
    assert all(assertions)


def test_solve_dual():
    A = np.array([[1, 0], [0, 1]], dtype="object")
    b = np.array([Dual(2, {"x": 1}), Dual(5, {"x": 1, "y": 1})])[:, np.newaxis]
    x = solve(A, b)
    assertions = (abs(b - x) < 1e-10)
    assert all(assertions)


def test_sparse_solve(A_sparse):
    b = np.array([0, 0.90929743, 0.14112001, -0.7568025, -0.95892427,
                  -0.2794155, 0.6569866, 0.98935825, 0])
    b = b[:, np.newaxis]
    x = solve(A_sparse, b)
    x_np = np.linalg.solve(A_sparse, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A_sparse.shape[0])]
    assert all(assertions)