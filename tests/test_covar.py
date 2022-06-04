import numpy as np

import context
from modules.curves import Portfolio


def test_covar_squared():
    mu1, mu2 = 1, 10
    si1, si2 = 1, 2
    p = 0.75
    Q = np.array([[si1**2, si1 * si2 * p], [si1 * si2 * p, si2 ** 2]])
    mu = np.array([mu1, mu2])[:, np.newaxis]
    Q2 = Portfolio.covar_squared(Q, mu)
    assert np.all(Q2.T == Q2)
    assert Q2[0, 0] == Q[0, 0]
    assert Q2[0, 1] == Q[0, 1]
    assert Q2[1, 1] == Q[1, 1]
    assert Q2[0, 2] == 2 * mu1 * si1 ** 2
    assert Q2[0, 3] == 2 * mu2 * si1 * si2 * p
    assert Q2[1, 2] == 2 * mu1 * si1 * si2 * p
    assert Q2[2, 2] == 2 * si1 ** 2 * (2 * mu1 ** 2 + si1 ** 2)
    assert Q2[2, 3] == 2 * si1 * si2 * p * (2 * mu1 * mu2 + si1 * si2 * p)
    assert Q2[3, 3] == 2 * si2 ** 2 * (2 * mu2 ** 2 + si2 ** 2)
