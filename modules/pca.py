import numpy as np


class PCA_:

    @staticmethod
    def pca(Q):
        lambd, E = np.linalg.eigh(Q)
        return lambd[::-1], E[:, ::-1]

    @classmethod
    def historical_multipliers(cls, Q, data):
        lambd, E = cls.pca(Q)
        centralised_data = data - data.mean(axis=0)
        return np.matmul(centralised_data, E)

    def pca_risk(self, curve, Q, S_ini=None):
        S = S_ini if S_ini is not None else self.risk(curve)
        lambd, E = self.pca(Q)
        return np.matmul(E.T, S)

    def pca_covar_alloc(self, curve, Q, S_ini=None):
        S_tilde = self.pca_risk(curve, Q, S_ini)
        lambd, E = self.pca(Q)
        c = self.covar(curve, Q)
        return (S_tilde[:, 0]**2 * lambd / c)[:, np.newaxis]

    def pca_hedge_adjustment(self, curve, Q, S_ini=None, H=[0], L=None):
        """defaults to hedging directionality: PC1 is set to zero"""
        S = S_ini if S_ini is not None else self.risk(curve=curve)
        lambd, E = self.pca(Q)
        n, n2 = len(lambd), len(H)
        E_H = E[:, H]
        if L is not None:
            n3 = len(L)
            L_ = np.zeros(shape=(n3, n))
            for row, col in enumerate(L):
                L_[row, col] = 1.0
            A = np.block([[np.eye(n), E_H, L_.T],
                          [E_H.T, np.zeros((n2, n2)), np.zeros((n2, n3))],
                          [L_, np.zeros((n3, n2)), np.zeros((n3, n3))]])
            b = np.block([[np.zeros((n, 1))],
                         [-np.matmul(E_H.T, S)],
                         [np.zeros((n3, 1))]])
            return np.linalg.solve(A, b)[:n, :]
        else:
            return -np.matmul(E_H, np.matmul(E_H.T, S))
