#!/usr/bin/env python3

from math import log, exp, floor, pi

from huygens.variable import Variable
from huygens.tool import conv, smooth, clamp


EPSILON = 1e-6


class Dynamic(Variable):
    def __init__(self, state):
        self.state = state
        self._t_start = state.t # _t_start is creation time (now)

    @property
    def t_now(self):
        return self.state.t - self._t_start

    def reset_t(self):
        #print(self.__class__.__name__, "reset_t")
        self._t_start = self.state.t # _t_start is now

    def is_done(self):
        return False

    @property
    def done(self):
        return self.is_done()

    def __lshift__(self, other):
        return Sequence(self.state, [self, other])
    #__rshift__ = __lshift__ # ?

    def repeat(self):
        return Repeat(self.state, self)


class Sequence(Dynamic):
    "Sequence of Dynamic Variable's"
    def __init__(self, state, children):
        Dynamic.__init__(self, state)
        assert len(children)
        self.children = list(children)
        #print("Sequence.__init__", id(self))
        assert not hasattr(self, "idx")
        self.idx = 0

    def reset_t(self):
        Dynamic.reset_t(self)
        self.idx = 0
        child = self.children[self.idx]
        child.reset_t()

    def _update(self):
        children = self.children
        assert self.idx < len(children)
        #print("Sequence._update: enter idx=", self.idx)
        while 1:
            child = children[self.idx]
            if not child.is_done() or self.idx+1==len(children):
                break
            #print("Sequence: idx+=1", self.idx)
            self.idx += 1
            child = children[self.idx]
            child.reset_t()
        #print("Sequence._update: exit idx=", self.idx)
        assert self.idx < len(children)

    def is_done(self):
        self._update()
        children = self.children
        child = children[self.idx]
        return self.idx+1==len(children) and child.is_done()

    def __float__(self):
        self._update()
        child = self.children[self.idx]
        return child.__float__()

    def __lshift__(self, other):
        return Sequence(self.state, self.children+[other])


class Repeat(Dynamic):
    def __init__(self, state, child):
        Dynamic.__init__(self, state)
        self.child = child

    def reset_t(self):
        Dynamic.reset_t(self)
        self.child.reset_t()

    def __float__(self):
        child = self.child
        if child.is_done():
            child.reset_t()
        value = child.__float__()
        return value


class Const(Dynamic):
    def __init__(self, state, x, dt=None):
        Dynamic.__init__(self, state)
        self.x = float(x)
        self.dt = dt

    def is_done(self):
        return self.dt is not None and self.t_now < self.dt

    def __float__(self):
        return self.x


class Linear(Dynamic):
    def __init__(self, state, x0, dx, modulus=None):
        Dynamic.__init__(self, state)
        self.x0 = float(x0)
        self.dx = float(dx)
        self.modulus = modulus

    def __float__(self):
        x = self.x0 + self.dx * self.t_now
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
        i = int(floor(self.t_now / self.dt)) + self.i0
        if self.repeat:
            i %= (self.i1 - self.i0)
        return i


class Cyclic(Dynamic):
    def __init__(self, state, mod=1.0, period=1.0):
        Dynamic.__init__(self, state)
        self.mod = mod
        self.period = period

    def __float__(self):
        x = self.t_now / self.period
        x %= self.mod
        return x


class Slider(Dynamic):
    " got a _start and stop value over a time period"
    def __init__(self, state, x0=0., x1=1., period=1., smooth=True):
        Dynamic.__init__(self, state)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.period = float(period)
        self.smooth = smooth

    def is_done(self):
        return self.t_now >= self.period-EPSILON


class Slew(Slider):
    def __float__(self):
        # XX do this properly...
        self.x0 = conv(self.x0, self.x1, 0.01*self.period)
        return self.x0

class LinSlide(Slider):
    def __float__(self):
        if self.smooth:
            a = smooth(0., 1., self.t_now/self.period)
        else:
            a = clamp(self.t_now/self.period)
        x = conv(self.x0, self.x1, a)
        return x


class LogSlide(Slider):
    def __float__(self):
        state = self.state
        a = smooth(0, 1, self.t_now/self.period)
        r0, r1 = log(self.x0), log(self.x1)
        r = conv(r0, r1, a)
        x = exp(r)
        return x


class World(object):
    def __init__(self, state):
        self.state = state

    def const(self, *args, **kw):
        v = Const(self.state, *args, **kw)
        return v

    def linear(self, *args, **kw):
        v = Linear(self.state, *args, **kw)
        return v
    lin = linear

    def slide(self, *args, **kw):
        v = LinSlide(self.state, *args, **kw)
        return v
    slider = slide

    def smooth(self, values, dt):
        assert len(values)>1
        lin = self.slide(values[0], values[1], dt)
        idx = 1
        while idx + 1 < len(values):
            lin = lin << self.slide(values[idx], values[idx+1], dt)
            idx += 1
        return lin

    def stepper(self, *args, **kw):
        v = Stepper(self.state, *args, **kw)
        return v

    def log_slide(self,  *args, **kw):
        v = LogSlide(self.state, *args, **kw)
        return v
    log_slider = log_slide

    def slew(self, *args, **kw):
        v = Slew(self.state, *args, **kw)
        return v

    def cyclic(self, *args, **kw):
        v = Cyclic(self.state, *args, **kw)
        return v


