import numpy as np
from scipy.stats import norm


class Covar_:

    def covar(self, curve, Q, alpha: float=None):
        S = self.risk(curve=curve)
        c = np.sqrt(np.matmul(S.T, np.matmul(Q, S)))[0, 0]
        if alpha is not None:
            return norm.ppf(1-alpha) * c
        return c

    def covar_smt(self, curve, Q):
        """single instrument minimising trade"""
        S = self.risk(curve=curve)
        Q_inv = np.diag(-1 / np.diagonal(Q))
        return np.matmul(Q_inv, np.matmul(Q, S))

    def covar_smt_impact(self, curve, Q):
        S, c = self.risk(curve=curve), self.covar(curve, Q)
        S_trade = self.covar_smt(curve, Q)
        S_min = S + np.diag(S_trade[:, 0])  # tensor
        c_impact = np.sqrt(np.matmul(S_min.T, np.matmul(Q, S_min))) - c
        return np.diagonal(c_impact)[:, np.newaxis]

    def covar_alloc(self, curve, Q):
        S, c = self.risk(curve=curve), self.covar(curve, Q)
        S_diag = np.diag(S[:, 0])
        return 1 / c * np.matmul(S_diag, np.matmul(Q, S))

    def covar_mmt(self, curve, Q, instruments):
        """multi-instrument minimising trade"""
        S = self.risk(curve=curve)
        S_hat, Q_hat = S[instruments, :], Q[instruments, :]
        Q_hat_hat = Q[np.ix_(instruments, instruments)]
        S_trade_hat = np.linalg.solve(Q_hat_hat, -np.matmul(Q_hat, S))
        S_trade = np.zeros_like(S)
        for ix, val in zip(instruments, S_trade_hat[:, 0]):
            S_trade[ix, 0] = val
        return S_trade

    def covar_mmt_impact(self, curve, Q, instruments):
        S, c = self.risk(curve=curve), self.covar(curve, Q)
        S_min = S + self.covar_mmt(curve, Q, instruments)
        return np.sqrt(np.matmul(S_min.T, np.matmul(Q, S_min)))[0, 0] - c