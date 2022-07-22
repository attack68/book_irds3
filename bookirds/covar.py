import numpy as np
from scipy.stats import norm


class Covar_:

    def covar(self, curve, Q, alpha: float=None, S_ini=None):
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        c = np.sqrt(np.matmul(S.T, np.matmul(Q, S)))[0, 0]
        if alpha is not None:
            return norm.ppf(1-alpha) * c
        return c

    def covar_smt(self, curve, Q, S_ini=None):
        """single instrument minimising trade"""
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        Q_inv = np.diag(-1 / np.diagonal(Q))
        return np.matmul(Q_inv, np.matmul(Q, S))

    def covar_smt_impact(self, curve, Q, S_ini=None):
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        c = self.covar(curve, Q, S_ini=S)
        S_trade = self.covar_smt(curve, Q, S_ini=S)
        S_min = S + np.diag(S_trade[:, 0])  # tensor
        c_impact = np.sqrt(np.matmul(S_min.T, np.matmul(Q, S_min))) - c
        return np.diagonal(c_impact)[:, np.newaxis]

    def covar_alloc(self, curve, Q, S_ini=None):
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        c = self.covar(curve, Q, S_ini=S)
        S_diag = np.diag(S[:, 0])
        return 1 / c * np.matmul(S_diag, np.matmul(Q, S))

    def covar_mmt(self, curve, Q, instruments, S_ini=None):
        """multi-instrument minimising trade"""
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        S_hat, Q_hat = S[instruments, :], Q[instruments, :]
        Q_hat_hat = Q[np.ix_(instruments, instruments)]
        S_trade_hat = np.linalg.solve(Q_hat_hat, -np.matmul(Q_hat, S))
        S_trade = np.zeros_like(S)
        for ix, val in zip(instruments, S_trade_hat[:, 0]):
            S_trade[ix, 0] = val
        return S_trade

    def covar_mmt_impact(self, curve, Q, instruments, S_ini=None):
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        c = self.covar(curve, Q, S_ini=S)
        S_min = S + self.covar_mmt(curve, Q, instruments, S_ini=S)
        return np.sqrt(np.matmul(S_min.T, np.matmul(Q, S_min)))[0, 0] - c

    @staticmethod
    def covar_squared(Q, mu):
        Q_12 = 2 * np.matmul(Q, np.diag(mu[:, 0]))
        Q_22 = 2 * Q * (2 * np.matmul(mu, mu.T) + Q)
        return np.block([[Q, Q_12], [Q_12.T, Q_22]])