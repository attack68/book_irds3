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

    def exp_pnl(self, curve, mu, Q=None, order=1, S=None, G=None):
        S = S if S is not None else self.risk(curve)
        ret = np.matmul(S.T, mu)
        if order == 2:
            if G is None:
                ss, sz, zz = self.cross_gamma(curve, swaps=True)
                G = ss + sz + sz.T + zz
            ret += 0.5 * np.einsum('ij, ji', G, Q)
            ret += 0.5 * np.einsum('ix, ij, jx', mu, G, mu)
        return ret

    def var_pnl(self, curve, mu, Q, order=1, S=None, G=None):
        S = S if S is not None else self.risk(curve)
        ret = np.matmul(np.matmul(S.T, Q), S)
        if order == 2:
            if G is None:
                ss, sz, zz = self.cross_gamma(curve, swaps=True)
                G = ss + sz + sz.T + zz
            ret += 0.5 * np.einsum('ij, jk, kl, li', G, Q, G, Q)
            ret += np.einsum('ix, ij, jk, kl, lx', mu, G, Q, G, mu)
            ret += 2 * np.einsum('ix, ij, jk, kx', S, Q, G, mu)
        return ret

    def sharpe(self, curve, mu, Q, order=1, S=None, G=None):
        exp = self.exp_pnl(curve, mu, Q, order, S, G),
        vol = np.sqrt(self.var_pnl(curve, mu, Q, order, S, G))
        return exp / vol