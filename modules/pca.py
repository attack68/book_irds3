import numpy as np


class PCA_:

    def pca(self, Q):
        lambd, E = np.linalg.eigh(Q)
        return lambd[::-1], E[:, ::-1]

    def historical_multipliers(self, Q, data):
        lambd, E = self.pca(Q)
        centralised_data = data - data.mean(axis=0)
        return np.matmul(centralised_data, E)

    def pca_risk(self, curve, Q):
        S = self.risk(curve)
        lambd, E = self.pca(Q)
        return np.matmul(E.T, S)

    def pca_covar_alloc(self, curve, Q):
        S_tilde = self.pca_risk(curve, Q)
        lambd, E = self.pca(Q)
        c = self.covar(curve, Q)
        return (S_tilde[:, 0]**2 * lambd / c)[:, np.newaxis]

