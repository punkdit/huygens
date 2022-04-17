#!/usr/bin/env python3

from math import log, exp, floor

from huygens.tool import conv, smooth, clamp


EPSILON = 1e-6

class Variable(object):
    "Turn into a float at the _slightest provocation."
    def __init__(self, state):
        self.state = state
        self.t0 = state.t # t0 is creation time (now)

    @property
    def t(self):
        return self.state.t - self.t0

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, float(self), self.__dict__)

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


class Linear(Variable):
    def __init__(self, state, x0, dx, modulus=None):
        Variable.__init__(self, state)
        self.x0 = float(x0)
        self.dx = float(dx)
        self.modulus = modulus

    def __float__(self):
        x = self.x0 + self.dx * self.t
        if self.modulus is not None:
            x %= self.modulus
        return x


class Stepper(Variable):
    def __init__(self, state, i0, i1, dt, repeat=False):
        Variable.__init__(self, state)
        self.i0 = int(i0)
        self.i1 = int(i1)
        self.dt = float(dt)
        self.repeat = repeat

    def __index__(self):
        i = int(floor(self.t / self.dt)) + self.i0
        if self.repeat:
            i %= (self.i1 - self.i0)
        return i



class Slider(Variable):
    " got a _start and stop value over a time period"
    def __init__(self, state, x0=0., x1=1., period=1., smooth=True):
        Variable.__init__(self, state)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.period = float(period)
        self.smooth = smooth

    def is_done(self):
        return self.t >= self.period-EPSILON


class LinSlide(Slider):
    def __float__(self):
        if self.smooth:
            a = smooth(0., 1., self.t/self.period)
        else:
            a = clamp(self.t/self.period)
        x = conv(self.x0, self.x1, a)
        return x


class LogSlide(Slider):
    def __float__(self):
        state = self.state
        a = smooth(0, 1, self.t/self.period)
        r0, r1 = log(self.x0), log(self.x1)
        r = conv(r0, r1, a)
        x = exp(r)
        return x


class World(object):
    def __init__(self, state):
        self.state = state

    def linear(self, *args, **kw):
        v = Linear(self.state, *args, **kw)
        return v

    def slider(self, *args, **kw):
        v = LinSlide(self.state, *args, **kw)
        return v

    def stepper(self, *args, **kw):
        v = Stepper(self.state, *args, **kw)
        return v

    def log_slider(self,  *args, **kw):
        v = LogSlide(self.state, *args, **kw)
        return v


