from datetime import datetime, timedelta
from math import log, exp, ceil
from copy import deepcopy


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
