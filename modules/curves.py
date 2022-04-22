from datetime import datetime, timedelta
import math
from math import ceil
from copy import deepcopy
from modules.dual import Dual


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

    def __init__(self, nodes: dict, interpolation: str):
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


class Schedule:

    def __init__(self, start: datetime, tenor: int, period: int, days=False):
        self.start = start
        self.end = add_months(start, tenor)
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
            period_end = add_months(period_start, self.period)
            schedule.append(
                [period_start, period_end, (period_end - period_start) / self.dcf_conv]
            )
            period_start = period_end
        schedule.append(
            [period_start, self.end, (self.end - period_start) / self.dcf_conv]
        )
        return schedule


class Swap:

    def __init__(
        self,
        start: datetime,
        tenor: int,
        period_fix: int,
        period_float: int,
    ):
        self.start = start
        self.end = add_months(start, tenor)
        self.schedule_fix = Schedule(start, tenor, period_fix)
        self.schedule_float = Schedule(start, tenor, period_float)

    def __repr__(self):
        return f"<Swap: {self.start.strftime('%Y-%m-%d')} -> " \
               f"{self.end.strftime('%Y-%m-%d')}>"

    def analytic_delta(self, curve: Curve, leg: str = "fix", notional: float = 1e4):
        delta = 0
        for period in getattr(self, f"schedule_{leg}").data:
            delta += curve[period[1]] * period[2]
        return delta * notional / 10000

    def rate(self, curve: Curve):
        rate = (curve[self.start] - curve[self.end]) / self.analytic_delta(curve)
        return rate * 100

    def npv(self, curve: Curve, fixed_rate: float, notional: float = 1e6):
        npv = (self.rate(curve) - fixed_rate) * self.analytic_delta(curve)
        return npv * notional / 100