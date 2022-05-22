import numpy as np


class PCA_:

    def pca(self, Q):
        lambd, E = np.linalg.eigh(Q)
        return lambd[::-1], E[:, ::-1]

    def historical_multipliers(self, Q, data):
        lambd, E = self.pca(Q)
        centralised_data = data - data.mean(axis=0)
        return np.matmul(centralised_data, E)
