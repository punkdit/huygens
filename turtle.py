#!/usr/bin/env python3
"""
copied from arrowtheory repo
"""

import sys

from math import sin, cos, pi, asin, acos
#from pyx import canvas, path, deco, trafo, style, text, color, deformer
from bruhat.render.front import *


def get_canvas():
    try:
        c = sys.modules["__main__"].cvs
    except:
        c = sys.modules["__main__"].c
    return c


def dopath(ps, extra=[], fill=[], closepath=False, smooth=0.0, stroke=True, cvs=None):
    #print("dopath:", ps)
    c = cvs
    if c is None:
        c = get_canvas()
    if len(ps) < 2:
        return
    ps = [path.moveto(*ps[0])]+[path.lineto(*p) for p in ps[1:]]
    if closepath:
        ps.append(path.closepath())
    p = path.path(*ps)
    extra = list(extra)
    if smooth:
        extra.append(deformer.smoothed(smooth))
    if fill:
        c.fill(p, extra+fill)
    if stroke:
        c.stroke(p, extra)


class Turtle(object):
    def __init__(self, x=0.0, y=0.0, angle=0.0, cvs=None, extra=[]):
        "angle: clockwise degrees starting from angle=0.0 is up"
        self.x = x
        self.y = y
        self.theta = (angle/360.)*2*pi
        self.ps = [(x, y)]
        self.paths = []
        self.pen = True
        self._save = None
        self.cvs = cvs
        self.extra = extra

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
        assert r > 1e-8, "can't lookat self"
        theta = asin(dx/r)
        self.theta = theta
        return self

    def fwd(self, d):
        self.x += d*sin(self.theta)
        self.y += d*cos(self.theta)
        if self.pen:
            self.ps.append((self.x, self.y))
        return self

    def moveto(self, x=None, y=None, angle=None):
        if x is not None and y is not None:
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

    def arrow(t, r, angle=30.):
        t.penup()
        t.fwd(0.2*r)
        t.right(angle)
        t.back(r)
        t.pendown()
        t.fwd(r)
        t.left(2*angle)
        t.back(r)
        
    def stroke(self, extra=[], fill=[], closepath=False, cvs=None):
        if self.pen:
            self.paths.append(self.ps)
        for ps in self.paths:
            if len(ps)>1:
                dopath(ps, extra, fill, closepath, 
                    smooth=0., cvs=cvs or self.cvs)
        self.paths = []
        self.ps = self.ps[-1:]
        return self

    def fill(self, fill, cvs=None):
        if self.pen:
            self.paths.append(self.ps)
        for ps in self.paths:
            if len(ps)>1:
                dopath(ps, [], fill, closepath=True, smooth=0., 
                    stroke=False, cvs=cvs or self.cvs)
        self.paths = []
        self.ps = self.ps[-1:]
        return self


def test():

    cvs = canvas.canvas()
    extra = [style.linewidth.THIck, color.rgb(0.2, 0.6, 0.2, 0.6),
        style.linejoin.bevel]
    turtle = Turtle(cvs=cvs, extra=extra)

    n = 8
    angle = 360. / n
    R = 3.0
    for i in range(n):
        turtle.fwd(1.*R)
        turtle.left((1./3)*angle)
        turtle.back(0.5*R)
        turtle.left((1./3)*angle)
        turtle.back(0.7*R)
        turtle.left((1./3)*angle)
        turtle.stroke(extra)

    #cvs.writePDFfile("test_turtle.pdf")

    cvs = canvas.canvas()
    turtle = Turtle(cvs=cvs, extra=extra)

    if 0:
        for i in range(24*2):
            turtle.left(320, 0.6*R)
            turtle.left(-60, 0.3*R)
            turtle.right(90, 0.6*R)
            turtle.stroke(extra)

    for i in range(1):
        turtle.fwd(2.)
        turtle.right(300, 1.)
    turtle.arrow(0.4)
    turtle.stroke(extra)

    cvs.writePDFfile("test_turtle.pdf")



    print("OK")


if __name__ == "__main__":

    test()



