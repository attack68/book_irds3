import numpy as np


class Gamma_:

    def cross_gamma(self, curve, disc_curve=None, swaps=False):
        disc_curve = disc_curve or curve
        v, dsds, d, n = disc_curve.var_collection
        dz, ds = self.risk_fwd_zero_rates(curve, disc_curve)
        U = np.matmul(np.triu(np.ones((n, n)) * dsds * d), np.diag(ds[:, 0]))
        dP_dsds = -(U + U.T) / 10000
        U = np.matmul(np.triu(np.ones((n, n)) * dsds * d), np.diag(dz[:, 0]))
        dP_dsdz = -U / 10000
        dP_dzdz = np.zeros((n, n))

        if swaps:
            J = np.matmul(curve.grad_s_v, curve.grad_v_r) * 100
            transform = lambda G: np.matmul(np.matmul(J, G), J.T)
            return transform(dP_dsds), transform(dP_dsdz), transform(dP_dzdz)
        return dP_dsds, dP_dsdz, dP_dzdz
