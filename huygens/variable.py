#!/usr/bin/env python3


class Variable(object):
    "Turn into a float at the _slightest provocation."

    def __str__(self):
        #return "%s(%s)"%(self.__class__.__name__, float(self)) # ?
        return str(float(self))

    def __repr__(self):
        ks = list(self.__dict__.keys())
        ks.sort()
        ns = ", ".join("%s=%s"%(k,self.__dict__[k]) for k in ks)
        return "%s(%s, %s, %s)"%(self.__class__.__name__, id(self), float(self), ns)

    def __lt__(self, other):
        return float(self) < other

    def __le__(self, other):
        return float(self) <= other

    def __eq__(self, other):
        return float(self) == other

    def __gt__(self, other):
        return float(self) > other

    def __ge__(self, other):
        return float(self) >= other

    def __add__(self, other):
        return other + float(self)
    __radd__ = __add__

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)

    def __mul__(self, other):
        return float(self) * other
    __rmul__ = __mul__

    def __truediv__(self, other):
        return float(self) / other
    __floordiv__ = __truediv__

    def __rtruediv__(self, other):
        return other / float(self)
    __rfloordiv__ = __rtruediv__

    def __mod__(self, other):
        return float(self) % other

    def __rmod__(self, other):
        return other % float(self)

    def __pow__(self, other):
        return float(self) ** other

    def __rpow__(self, other):
        return other ** float(self)

    def __neg__(self):
        return -float(self)

    def __pos__(self):
        return float(self)

    def __abs__(self):
        return abs(float(self))



