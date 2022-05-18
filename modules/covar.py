import numpy as np
from scipy.stats import norm


class Covar_:

    def covar(self, curve, Q, alpha: float=None):
        S = self.risk(curve=curve)
        c = np.sqrt(np.matmul(S.T, np.matmul(Q, S)))
        if alpha is not None:
            return norm.ppf(1-alpha) * c
        return c