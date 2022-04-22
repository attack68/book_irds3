from math import exp, log


class Dual:

    def __init__(self, real, dual=None):
        """
        A dual number is denoted in the form:
            z = x + ye, such that e**2 = 0 and e != 0

        When tracking multiple variable derivatives we use a dict such that:
            z = x + y_0e_0 + y_1e_1, such that e_ie_j = 0 and e_i != 0

        real: real number
        dual: dict (key=name_index and value=value)
        """
        self.real = real
        self.dual = {} if dual is None else dual.copy()

    def __neg__(self):
        """-z = -x - y_0e_0 - y_1e_1"""
        dual = {}
        for key in self.dual:
            dual[key] = -self.dual[key]
        return Dual(-self.real, dual)

    def __eq__(self, argument):
        """equality iff real and dual components the same"""
        if not isinstance(argument, Dual):
            return False
        else:
            if self.real != argument.real:
                return False
            else:
                return self.dual == argument.dual

    def conjugate(self):
        dual = {}
        for key in self.dual:
            dual[key] = -self.dual[key]
        return Dual(self.real, dual)

    def __ne__(self, argument):
        """inequality iff not equal"""
        return not (self.__eq__(argument))

    def __add__(self, argument):
        """z_1 + z_2 = (x_1 + x_2) + (y_1_0 + y_2_0)e_0 + (y_1_1 + y_2_1)e_1"""
        if isinstance(argument, Dual):
            real = self.real + argument.real
            dual = self.dual.copy()
            for key in argument.dual:
                val = 0 if key not in dual else dual[key]
                dual[key] = val + argument.dual[key]
            return Dual(real, dual)
        else:
            return Dual(self.real + argument, self.dual)

    __radd__ = __add__

    def __sub__(self, argument):
        """z_1 - z_2 = (x_1 - x_2) + (y_1_0 - y_2_0)e_0 + (y_1_1 - y_2_1)e_1"""
        if isinstance(argument, Dual):
            real = self.real - argument.real
            dual = self.dual.copy()
            for key in argument.dual:
                val = 0 if key not in dual else dual[key]
                dual[key] = val - argument.dual[key]
            return Dual(real, dual)
        else:
            return Dual(self.real - argument, self.dual)

    def __rsub__(self, argument):
        """z_2 - z_1 = -(z_1 - z_2)"""
        return -(self - argument)

    def __mul__(self, argument):
        """z_1 * z_2 = x_1 * x_2 + (x_1 * y_2_0 + x_2 * y_1_0)e_0"""
        if isinstance(argument, Dual):
            real = self.real * argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument.real
            for key in argument.dual:
                val = 0 if key not in dual else dual[key]
                dual[key] = val + argument.dual[key] * self.real
            return Dual(real, dual)
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument
            return Dual(self.real * argument, dual)

    __rmul__ = __mul__

    def __truediv__(self, argument):
        """z_1 / z_2 = (z_1 * ^z_2) / (z_2 * ^z_2)"""
        if isinstance(argument, Dual):
            numerator = self * argument.conjugate()
            return numerator / argument.real**2
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] / argument
            return Dual(self.real / argument, dual)

    def __rtruediv__(self, argument):
        """x / z = (x * ^z) / (z * ^z)"""
        numerator = Dual(argument, {})
        return numerator / self

    def __pow__(self, power):
        """z**n = x**n + n x**(n-1)(y_0e_0 + y_1e_1)"""
        dual = {}
        for key in self.dual:
            dual[key] = power * self.dual[key] * (self.real ** (power - 1))
        return Dual(self.real ** power, dual)

    def __str__(self):
        output = f"    f = {self.real:.8f}\n"
        for k, v in self.dual.items():
            output += f"df/d{k} = {v:.6f}\n"
        return output

    def __repr__(self):
        output = f"{self.real}"
        for key in self.dual.keys():
            output += f"{self.dual[key]:+.3f}e_{key}"
        return output

    def __exp__(self):
        """exp(z) = exp(x) + exp(x) * (y_0e_0 + y_1e_1)"""
        real = exp(self.real)
        dual = {}
        for key in self.dual:
            dual[key] = real * self.dual[key]
        return Dual(real, dual)

    def __log__(self):
        """log(z) = log(x) + 1/x * (y_0e_0 + y_1e_1)"""
        real = log(self.real)
        dual = {}
        for key in self.dual:
            dual[key] = self.dual[key] / self.real
        return Dual(real, dual)
