from datetime import datetime, timedelta
import math
from math import ceil
from copy import deepcopy, copy
import numpy as np
from bookirds.dual import Dual
from bookirds.bsplines import BSpline
from bookirds.covar import Covar_
from bookirds.pca import PCA_
from bookirds.mid_market import Margin_
from bookirds.cross_gamma import Gamma_


def exp(x):
    if isinstance(x, Dual):
        return x.__exp__()
    return math.exp(x)


def log(x):
    if isinstance(x, Dual):
        return x.__log__()
    return math.log(x)


def interpolate(x, x_1, y_1, x_2, y_2, interpolation, start=None):
    if interpolation == "linear":
        op = lambda z: z
    elif interpolation == "log_linear":
        op, y_1, y_2 = exp, log(y_1), log(y_2)
    elif interpolation == "linear_zero_rate":
        y_1 = log(y_1) / ((start - x_1) / timedelta(days=365))
        y_2 = log(y_2) / ((start - x_2) / timedelta(days=365))
        op = lambda z: exp((start-x)/timedelta(days=365) * z)
    ret = op(y_1 + (y_2 - y_1) * (x - x_1) / (x_2 - x_1))
    return ret


class Curve:

    def __init__(self, nodes: dict, interpolation: str, **kwargs):
        self.nodes = deepcopy(nodes)
        self.interpolation = interpolation

    def __getitem__(self, date: datetime):
        node_dates = list(self.nodes.keys())
        for i, node_date_1 in enumerate(node_dates[1:]):
            if date <= node_date_1 or i == len(node_dates) - 2:
                node_date_0 = node_dates[i]
                return interpolate(
                    date,
                    node_date_0,
                    self.nodes[node_date_0],
                    node_date_1,
                    self.nodes[node_date_1],
                    self.interpolation,
                    node_dates[0]
                )

    def __repr__(self):
        output = ""
        for k, v in self.nodes.items():
            output += f"{k.strftime('%Y-%b-%d')}: {v:.6f}\n"
        return output

    def __copy__(self):
        W = getattr(self, "W", None)
        return type(self)(
            nodes=self.nodes,
            interpolation=self.interpolation,
            swaps=getattr(self, "swaps", None),
            algorithm=getattr(self, "algo", None),
            obj_rates=getattr(self, "obj_rates", None),
            w=None if W is None else np.diagonal(W),
            t=getattr(self, "t", None),
        )

    def rate(self, start: datetime, months: int = None, days: int = None):
        if months is not None:
            end = add_months(start, months)
        elif days is not None:
            end = start + timedelta(days=days)
        else:
            end = None
        df_ratio = self[start] / self[end]
        rate = (df_ratio - 1) * timedelta(days=365) / (end - start)
        return rate * 100

    @property
    def var_collection(self):
        def val(v):
            return v.real if isinstance(v, Dual) else v
        v = np.array([val(v) for v in self.nodes.values()])
        n = v.shape[0] - 1
        dsds = np.array([v[i + 1] / v[i] for i in range(n)])
        dates = [k for k in self.nodes.keys()]
        d = np.array([
            (dates[i + 1] - dates[i]) / timedelta(days=365) for i in range(n)
        ])
        return v[:, np.newaxis], dsds[:, np.newaxis], d[:, np.newaxis], n

    @property
    def grad_r_v(self):
        if getattr(self, "grad_r_v_", None) is None:
            v, dsds, d, n = self.var_collection
            v = v[1:, :]
            alpha = np.triu(np.ones((n, n)))
            self.grad_r_v_ = -np.matmul(np.diag(dsds[:, 0]), np.matmul(d, v.T)) * alpha
        return self.grad_r_v_

    @property
    def grad_v_r(self):
        if getattr(self, "grad_v_r_", None) is None:
            v, dsds, d, n = self.var_collection
            diag0 = -v[:n, 0] / (d[:, 0] * v[1:, 0] ** 2)
            diag1 = 1 / (d[:, 0] * v[1:, 0])
            self.grad_v_r_ = np.diag(diag0) + np.diag(diag1[1:], 1)
        return self.grad_v_r_


def add_months(start: datetime, months: int) -> datetime:
    """add a given number of months to an input date with a modified month end rule"""
    year_roll = int((start.month + months - 1) / 12)
    month = (start.month + months) % 12
    month = 12 if month == 0 else month
    try:
        end = datetime(start.year + year_roll, month, start.day)
    except ValueError:  # day is out of range for month
        return add_months(datetime(start.year, start.month, start.day-1), months)
    else:
        return end


def add_days(start: datetime, days: int) -> datetime:
    return start + timedelta(days=days)


class Schedule:

    def __init__(self, start: datetime, tenor: int, period: int, days=False):
        self.add_op = add_days if days else add_months
        self.start = start
        self.end = self.add_op(start, tenor)
        self.tenor = tenor
        self.period = period
        self.dcf_conv = timedelta(days=365)
        self.n_periods = ceil(tenor / period)

    def __repr__(self):
        output = "period start | period end | period DCF\n"
        for period in self.data:
            output += f"{period[0].strftime('%Y-%b-%d')} | " \
                      f"{period[1].strftime('%Y-%b-%d')} | {period[2]:3f}\n"
        return output

    @property
    def data(self):
        schedule = []
        period_start = self.start
        for i in range(self.n_periods - 1):
            period_end = self.add_op(period_start, self.period)
            schedule.append(
                [period_start, period_end, (period_end - period_start) / self.dcf_conv]
            )
            period_start = period_end
        schedule.append(
            [period_start, self.end, (self.end - period_start) / self.dcf_conv]
        )
        return schedule


class SolvedCurve(Curve):
    def __init__(self, nodes: dict, interpolation: str, swaps: list, obj_rates: list,
                 algorithm: str = "gauss_newton", w: list = None, **kwargs):
        super().__init__(nodes=nodes, interpolation=interpolation)
        self.swaps, self.obj_rates, self.algo = swaps, obj_rates, algorithm
        self.n, self.m = len(self.nodes.keys()) - 1, len(self.swaps)
        self.s = np.array([self.obj_rates]).transpose()
        self.W = None if w is None else np.diag(w)
        self.lam = 1000

    def calculate_metrics(self):
        self.r = np.array([[swap.rate(self) for swap in self.swaps]]).transpose()
        self.v = np.array([[v for v in list(self.nodes.values())[1:]]]).transpose()
        x = self.r - self.s
        Wx = x if self.W is None else np.matmul(self.W, x)
        self.f = np.matmul(x.transpose(), Wx)[0][0]
        self.grad_v_f = np.array(
            [[self.f.dual.get(f"v{i+1}", 0) for i in range(self.n)]]
        ).transpose()
        self.J = np.array([
            [rate.dual.get(f"v{j+1}", 0) for rate in self.r[:, 0]]
            for j in range(self.n)
        ])

    def update_step_gradient_descent(self):
        y = np.matmul(self.J.transpose(), self.grad_v_f)
        alpha = np.matmul(y.transpose(), self.r - self.s) / np.matmul(y.transpose(), y)
        alpha = alpha[0][0].real
        v_1 = self.v - self.grad_v_f * alpha
        return v_1

    def update_step_gauss_newton(self):
        J_T = self.J.transpose()
        A = np.matmul(self.J, J_T if self.W is None else np.matmul(self.W, J_T))
        b = -0.5 * self.grad_v_f
        delta = np.linalg.solve(A, b)
        v_1 = self.v + delta
        return v_1

    def update_step_levenberg_marquardt(self):
        self.lam *= 2 if self.f_prev < self.f.real else 0.5
        J_T = self.J.transpose()
        WJ_T = J_T if self.W is None else np.matmul(self.W, J_T)
        A = np.matmul(self.J, WJ_T) + self.lam * np.eye(self.J.shape[0])
        b = -0.5 * self.grad_v_f
        delta = np.linalg.solve(A, b)
        v_1 = self.v + delta
        return v_1

    def iterate(self, max_i=2000, tol=1e-10):
        ret, self.f_prev, self.f_list = None, 1e10, []
        for i in range(max_i):
            self.calculate_metrics()
            self.f_list.append(self.f.real)
            if self.f.real < self.f_prev and (self.f_prev - self.f.real) < tol:
                ret = f"tolerance reached ({self.algo}) after {i} iterations, "
                ret += f"func: {self.f.real}"
                break
            v_1 = getattr(self, f"update_step_{self.algo}")()
            for i, (k, v) in enumerate(self.nodes.items()):
                if i == 0:
                    continue
                self.nodes[k] = v_1[i - 1, 0]
            self.f_prev = self.f.real
        self.lam = 1000
        return f"max iterations ({self.algo}), f: {self.f.real}" if ret is None else ret

    @property
    def grad_s_v(self):
        if getattr(self, "grad_s_v_", None) is None:
            self.grad_s_v_numeric()
        return self.grad_s_v_

    def grad_s_v_numeric(self, **kwargs):
        kwargs = {
            "interpolation": self.interpolation,
            "nodes": self.nodes,
            "algorithm": "gauss_newton",
            "swaps": self.swaps,
            "obj_rates": self.obj_rates,
            "w": None if self.W is None else np.diagonal(self.W),
            **kwargs
        }
        grad_s_v = np.zeros(shape=(self.m, self.n))
        ds = 1e-2
        s_cv_fwd = type(self)(**kwargs)
        s_cv_bck = type(self)(**kwargs)
        s_cv_fwd.not_iterated = False
        s_cv_bck.not_iterated = False
        for s in range(self.m):
            s_cv_fwd.nodes, s_cv_fwd.s = deepcopy(self.nodes), self.s.copy()
            s_cv_bck.nodes, s_cv_bck.s = deepcopy(self.nodes), self.s.copy()
            s_cv_fwd.s[s, 0] += ds
            s_cv_bck.s[s, 0] -= ds
            print("fwd", s_cv_fwd.iterate())
            print("bck", s_cv_bck.iterate())
            dvds_fwd = np.array([v.real for v in (s_cv_fwd.v[:, 0] - self.v[:, 0])/ds])
            dvds_bck = np.array([v.real for v in (s_cv_bck.v[:, 0] - self.v[:, 0])/ds])
            grad_s_v[s, :] = (dvds_fwd - dvds_bck) / 2
        self.grad_s_v_ = grad_s_v


class Swap(Covar_, PCA_, Margin_):

    def __init__(
        self,
        start: datetime,
        tenor: int,
        period_fix: int,
        period_float: int,
        days: bool = False,
        fixed_rate: float = None,
        notional: float = None,
    ):
        self.add_op = add_days if days else add_months
        self.start = start
        self.end = self.add_op(start, tenor)
        self.schedule_fix = Schedule(start, tenor, period_fix, days=days)
        self.schedule_float = Schedule(start, tenor, period_float, days=days)
        self.fixed_rate = fixed_rate
        self.notional = 1e6 if notional is None else notional

    def __repr__(self):
        return f"<Swap: {self.start.strftime('%Y-%m-%d')} -> " \
               f"{self.end.strftime('%Y-%m-%d')}>"

    def analytic_delta(self, curve: Curve, leg: str = "fix"):
        delta = 0
        for period in getattr(self, f"schedule_{leg}").data:
            delta += curve[period[1]] * period[2]
        return delta * self.notional / 10000

    def rate(self, curve: Curve):
        if self.notional == 0:
            self.notional = 1
            analytic_delta = self.analytic_delta(curve) * 10000 / self.notional
            rate = (curve[self.start] - curve[self.end]) / analytic_delta
            self.notional = 0
        else:
            analytic_delta = self.analytic_delta(curve) * 10000 / self.notional
            rate = (curve[self.start] - curve[self.end]) / analytic_delta
        return rate * 100

    def npv(self, curve: Curve):
        self.set_fixed_rate(fixed_rate=self.fixed_rate, curve=curve)
        npv = (self.rate(curve) - self.fixed_rate) * self.analytic_delta(curve)
        return npv * 100

    def risk(self, curve: SolvedCurve):
        grad_v_P = np.array([
            [self.npv(curve).dual.get(f"v{i+1}", 0)
             for i in range(curve.n)]
        ]).transpose()
        grad_s_P = np.matmul(curve.grad_s_v, grad_v_P)
        return grad_s_P / 100

    def set_fixed_rate(self, fixed_rate: float = None, *args, **kwargs):
        if fixed_rate is None:
            fixed_rate = self.rate(*args, **kwargs)
            if isinstance(fixed_rate, Dual):
                fixed_rate = fixed_rate.real
        self.fixed_rate = fixed_rate


class Swap2(Swap, Gamma_):
    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def rate(self, curve: Curve, disc_curve: Curve = None):
        disc_curve = disc_curve or curve
        if self.notional == 0:
            self.notional = 1
            fixed_delta = self.analytic_delta(disc_curve) * 10000 / self.notional
            self.notional = 0
        else:
            fixed_delta = self.analytic_delta(disc_curve) * 10000 / self.notional

        floating_leg = 0
        for period in getattr(self, f"schedule_float").data:
            _ = (curve[period[0]] / curve[period[1]] - 1)
            floating_leg += _ * disc_curve[period[1]]

        return floating_leg / fixed_delta * 100

    def npv(self, curve: Curve, disc_curve: Curve = None):
        disc_curve = disc_curve or curve
        self.set_fixed_rate(self.fixed_rate, curve, disc_curve)
        rate_diff = (self.rate(curve, disc_curve) - self.fixed_rate)
        return rate_diff * self.analytic_delta(disc_curve) * 100

    def risk_fwd_zero_rates(self, curve: SolvedCurve, disc_curve: SolvedCurve = None):
        disc_curve = disc_curve or curve
        stat_crv, stat_disc_crv = copy(curve), copy(disc_curve)
        for crv in [stat_crv, stat_disc_crv]:
            crv.nodes = {k: v.real for (k, v) in crv.nodes.items()}
        n = len(curve.nodes.keys())
        npv_fore = self.npv(curve, stat_disc_crv)
        grad_zv_p = np.array([npv_fore.dual.get(f"v{i}", 0) for i in range(1, n)])
        npv_disc = self.npv(stat_crv, disc_curve)
        grad_sv_p = np.array([npv_disc.dual.get(f"v{i}", 0) for i in range(1, n)])

        grad_z_p = np.matmul(curve.grad_r_v, grad_zv_p[:, np.newaxis])
        grad_s_p = np.matmul(disc_curve.grad_r_v, grad_sv_p[:, np.newaxis])
        return grad_z_p / 10000, grad_s_p / 10000


class AdvancedCurve(SolvedCurve):
    def __init__(self, nodes: dict, interpolation: str, swaps: list, obj_rates: list,
                 t: list, algorithm: str = "gauss_newton", w: list = None):
        super().__init__(nodes, interpolation, swaps, obj_rates, algorithm, w=w)
        self.t = t
        self.not_iterated = True
        self.bs = BSpline(4, t)

    def __getitem__(self, date: datetime):
        if date <= self.t[0]:
            return super().__getitem__(date)
        else:
            return self.bs.ppev_single(date).__exp__()

    def __copy__(self):
        ret = super().__copy__()
        ret.bs = copy(self.bs)
        return ret

    def solve_bspline(self):
        tau = [k for k in self.nodes.keys() if k >= self.t[0]]
        y = [v.__log__() for k, v in self.nodes.items() if k >= self.t[0]]

        # add second derivative endpoints
        tau.insert(0, self.t[0])
        tau.append(self.t[-1])
        y.insert(0, 0)
        y.append(0)

        self.bs.bsplsolve(np.array(tau), np.array(y), 2, 2)

    def calculate_metrics(self):
        self.solve_bspline()
        super().calculate_metrics()

    @property
    def grad_s_v(self):
        if getattr(self, "grad_s_v_", None) is None:
            self.grad_s_v_numeric(t=self.t)
        return self.grad_s_v_

    def iterate(self):
        if self.not_iterated:
            w = None if self.W is None else np.diagonal(self.W)
            base_solve = SolvedCurve(
                self.nodes, self.interpolation, self.swaps,
                self.obj_rates, algorithm=self.algo, w=w
            )
            print("basic solve: ", base_solve.iterate())
            self.nodes = base_solve.nodes
            self.not_iterated, self.algo = False, "gauss_newton"
        return super().iterate()


class SwapSpread:
    def __init__(self, swap1: Swap, swap2: Swap):
        self.swap1 = swap1
        self.swap2 = swap2

    def rate(self, curve: Curve):
        return self.swap2.rate(curve) - self.swap1.rate(curve)


class Portfolio(Covar_, PCA_, Margin_, Gamma_):
    def __init__(self, objects: list = []):
        self.objects = objects

    def risk(self, curve: SolvedCurve):
        risk = self.objects[0].risk(curve)
        for obj in self.objects[1:]:
            risk += obj.risk(curve)
        return risk

    def npv(self, curve: SolvedCurve):
        npv = self.objects[0].npv(curve)
        for obj in self.objects[1:]:
            npv += obj.npv(curve)
        return npv

    def risk_fwd_zero_rates(self, curve: SolvedCurve, disc_curve: SolvedCurve = None):
        disc_curve = disc_curve or curve
        risk_z, risk_s = self.objects[0].risk_fwd_zero_rates(curve, disc_curve)
        for obj in self.objects[1:]:
            risk = obj.risk_fwd_zero_rates(curve, disc_curve)
            risk_z += risk[0]
            risk_s += risk[1]
        return risk_z, risk_s
