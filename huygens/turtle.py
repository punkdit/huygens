#!/usr/bin/env python3
"""
copied from arrowtheory repo
"""

import sys

from math import sin, cos, pi, asin, acos, atan
from huygens.front import *

# XXX TODO 
# Cleanup this mess: 
#   (1) don't just build paths out of straight lines
#   (2) construct actual Path object in Turtle instances


def mkpath(ps, closepath=False):
    ps = [path.moveto(*ps[0])]+[path.lineto(*p) for p in ps[1:]]
    if closepath:
        ps.append(path.closepath())
    p = path.path(*ps)
    return p


def dopath(ps, attrs=[], fill=[], closepath=False, smooth=0.0, stroke=True, cvs=None):
    #print("dopath:", ps)
    c = cvs
    if len(ps) < 2:
        return
    p = mkpath(ps)
    attrs = list(attrs)
    if smooth:
        attrs.append(deformer.smoothed(smooth)) # XXX not implemented
    if fill:
        c.fill(p, attrs+fill)
    if stroke:
        c.stroke(p, attrs)


class Turtle(object):
    default_attrs = [style.linewidth.thin, style.linecap.round, style.linejoin.round]

    def __init__(self, x=0.0, y=0.0, angle=0.0, cvs=None, attrs=None):
        "angle: clockwise degrees starting from angle=0.0 is up"
        self.x = x
        self.y = y
        self.theta = (angle/360.)*2*pi
        self.ps = [(x, y)]
        self.paths = []
        self.pen = True
        self._save = None
        self.cvs = cvs
        if attrs is None:
            attrs = self.default_attrs
        self.attrs = attrs

    def copy(self):
        import copy
        t = copy.deepcopy(self)
        return t

    def save(self):
        assert self._save is None
        self._save = (self.x, self.y, self.theta)

    def restore(self):
        assert self._save is not None
        self.x, self.y, self.theta = self._save
        self._save = None

    def penup(self):
        if self.ps:
            self.paths.append(self.ps)
        self.pen = False
        self.ps = None
        return self

    def pendown(self):
        if self.pen:
            return
        self.pen = True
        self.ps = [(self.x, self.y)]
        return self

    def getpos(self):
        return self.x, self.y

    def lookat(self, x, y):
        dx = x-self.x        
        dy = y-self.y        
        r = (dx**2 + dy**2)**0.5
        #assert r > 1e-8, "can't lookat self"
        if r < 1e-8:
            return self
        if dy > EPSILON:
            theta = atan(dx/dy)
        elif dy < -EPSILON:
            theta = atan(dx/dy) + pi
        elif dx > EPSILON:
            theta = 0.5*pi
        elif dx < -EPSILON:
            theta = -0.5*pi
        else:
            assert 0
        self.theta = theta
        return self

    def fwd(self, d):
        self.x += d*sin(self.theta)
        self.y += d*cos(self.theta)
        if self.pen:
            self.ps.append((self.x, self.y))
        return self

    def goto(self, x=None, y=None, angle=None):
        if x is not None and y is not None:
            self.lookat(x, y)
            self.ps.append((x, y))
            self.x = x
            self.y = y
        if angle is not None:
            self.theta = (angle/360.)*2*pi
        return self

    def reverse(self, d):
        self.fwd(-d)
        return self
    back = reverse
    rev = reverse

    def right(self, angle, r=0., N=40):
        dtheta = (angle/360.)*2*pi
        theta = self.theta
        self.theta += dtheta
        if r==0.:
            return self
        x, y = self.x, self.y
        x0 = x - r*sin(theta-pi/2)
        y0 = y - r*cos(theta-pi/2)
        for i in range(N):
            theta += (1./(N))*dtheta
            x = x0 - r*sin(theta+pi/2)
            y = y0 - r*cos(theta+pi/2)
            if self.pen:
                self.ps.append((x, y))
        self.x = x
        self.y = y
        return self

    def left(self, angle, r=0.):
        self.right(-angle, -r)
        return self

    def flat_arrow(self, size=0.15, angle=30.):
        self.penup()
        self.right(angle)
        self.back(size)
        self.pendown()
        self.fwd(size)
        self.left(2*angle)
        self.back(size)
        self.penup()
        self.fwd(size)
        self.right(angle)
        self.pendown()
        return self
        
    def dart_arrow(self, size=0.15, angle=30.):
        assert 0<angle<90, "bad angle %s"%angle
        b = size / (2*cos(2*pi*angle/360.))
        self.save()
        #self.penup()
        #self.fwd(0.1*b) # Clear the end of the line we are pointing on.
        #self.pendown()  # We don't seem to need this if we stroke it.
        self.right(angle)
        self.back(size)
        self.right(angle)
        self.fwd(b)
        self.left(180-2*angle)
        self.back(b)
        self.right(angle)
        self.fwd(size)
        self.restore()
        return self
        
    def curve_arrow(self, size=0.15, angle=30.):
        a = 0.8*angle
        b = 0.9*a
        self.save()
        self.pendown()
        self.left(180)
        self.right(b)
        self.right(2*a, size)
        self.penup()
        self.restore()

        self.save()
        self.pendown()
        self.left(180)
        self.left(b)
        self.left(2*a, size)
        self.penup()
        self.restore()
        return self

    def bar_arrow(self, size=0.15, *arg, **kw):
        return self.flat_arrow(0.6*size, 90)

    def feather_arrow(self, size=0.15, angle=30.):
        # not perfect... but works in some circumstances
        self.curve_arrow(size, angle)
        self.reverse(0.5*size)
        self.curve_arrow(size, angle)

    def hook_arrow(self, size=0.15, *arg, **kw):
        self.save()
        self.right(180., -0.5*size)
        self.reverse(0.1*size)
        self.restore()

    def arrow(self, size=0.1, angle=30., astyle="flat"):
        meth = getattr(self, astyle+"_arrow", None)
        assert meth is not None, "astyle %r not found"%astyle
        meth(size, angle)
        return self

    def mkpath(self, closepath=False):
        paths = list(self.paths)
        if self.pen:
            paths.append(self.ps)
        p = None
        for pts in paths:
            if len(pts)<=1:
                continue
            p1 = mkpath(pts, closepath)
            assert p1 is not None
            p = p1 if p is None else p+p1
            assert p is not None
        return p
        
    def _render(self, attrs=None, closepath=False, cvs=None, name="stroke", preserve=False):
        if attrs is None:
            attrs = self.attrs
        if cvs is None:
            cvs = self.cvs
        assert cvs is not None
        if self.pen:
            self.paths.append(self.ps)
        for ps in self.paths:
            if len(ps)>1:
                p = mkpath(ps, closepath)
                method = getattr(cvs, name)
                method(p, attrs)
        if not preserve:
            self.paths = []
            if self.ps is not None:
                self.ps = self.ps[-1:]
        return self

    def stroke(self, *args, **kw):
        kw["name"] = "stroke"
        self._render(*args, **kw)

    def fill(self, *args, **kw):
        kw["name"] = "fill"
        self._render(*args, **kw)

    def stroke_preserve(self, *args, **kw):
        kw["name"] = "stroke"
        kw["preserve"] = True
        self._render(*args, **kw)

    def fill_preserve(self, *args, **kw):
        kw["name"] = "fill"
        kw["preserve"] = True
        self._render(*args, **kw)


