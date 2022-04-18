#!/usr/bin/env python3

from math import log, exp, floor

from huygens.variable import Variable
from huygens.tool import conv, smooth, clamp


EPSILON = 1e-6


class Dynamic(Variable):
    def __init__(self, state):
        self.state = state
        self.t0 = state.t # t0 is creation time (now)

    @property
    def t(self):
        return self.state.t - self.t0

    def is_done(self):
        return False

    @property
    def done(self):
        return self.is_done()


class Linear(Dynamic):
    def __init__(self, state, x0, dx, modulus=None):
        Dynamic.__init__(self, state)
        self.x0 = float(x0)
        self.dx = float(dx)
        self.modulus = modulus

    def __float__(self):
        x = self.x0 + self.dx * self.t
        if self.modulus is not None:
            x %= self.modulus
        return x


class Stepper(Dynamic):
    def __init__(self, state, i0, i1, dt, repeat=False):
        Dynamic.__init__(self, state)
        self.i0 = int(i0)
        self.i1 = int(i1)
        self.dt = float(dt)
        self.repeat = repeat

    def __index__(self):
        i = int(floor(self.t / self.dt)) + self.i0
        if self.repeat:
            i %= (self.i1 - self.i0)
        return i



class Slider(Dynamic):
    " got a _start and stop value over a time period"
    def __init__(self, state, x0=0., x1=1., period=1., smooth=True):
        Dynamic.__init__(self, state)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.period = float(period)
        self.smooth = smooth

    def is_done(self):
        return self.t >= self.period-EPSILON


class Slew(Slider):
    def __float__(self):
        # XX do this properly...
        self.x0 = conv(self.x0, self.x1, 0.01*self.period)
        return self.x0

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

    def slide(self, *args, **kw):
        v = LinSlide(self.state, *args, **kw)
        return v

    def stepper(self, *args, **kw):
        v = Stepper(self.state, *args, **kw)
        return v

    def log_slide(self,  *args, **kw):
        v = LogSlide(self.state, *args, **kw)
        return v

    def slew(self, *args, **kw):
        v = Slew(self.state, *args, **kw)
        return v


