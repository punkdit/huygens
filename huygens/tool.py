#!/usr/bin/env python3

from random import random

def rnd(a=0., b=1.):
    return (b-a)*random() + a


def conv(a, b, alpha=0.5):
    return (1.-alpha)*a + alpha*b


def smooth(val0, val1, alpha):
    #assert 0.<=alpha<=1.
    if alpha <= 0:
        return val0
    if alpha >= 1.:
        return val1
    s = -2*(alpha**3) + 3*(alpha**2) # 0.<=s<=1.
    val = (1-s)*val0 + s*val1 # conv
    return val


def bump(val0, val1, alpha):
    if alpha <= 0.5:
        val = smooth(val0, val1, 2*alpha)
    else:
        val = smooth(val0, val1, 2*(1.-alpha))
    return val


def clamp(value, vmin=0., vmax=1.):
    value = max(vmin, value)
    value = min(vmax, value)
    return value


