#!/usr/bin/env python

"""
replacement for huygens.turtle

"""

from math import pi, cos, sin, atan

from huygens.front import path, style, Canvas, color
from huygens.back import arc_to_bezier, EPSILON


class Turtle:

    default_attrs = [style.linewidth.thin, style.linecap.round, style.linejoin.round]

    def __init__(self, x=0.0, y=0.0, angle=0.0, cvs=None, attrs=None):
        "angle: clockwise degrees starting from angle=0.0 is up"
        self.x = x
        self.y = y
        self.theta = (angle/360.)*2*pi
        self.pen = True
        self._save = None
        self.cvs = cvs
        if attrs is None:
            attrs = self.default_attrs
        self.attrs = list(attrs)
        self.reset()

    def reset(self):
        self.current = [path.moveto(self.x, self.y)]
        self.paths = []

    def startat(self):
        angle = 360*self.theta/(2*pi)
        return Turtle(self.x, self.y, angle, self.cvs, self.attrs)

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
        self.paths.append(self.current)
        self.pen = False
        self.current = None
        return self

    def pendown(self):
        if self.pen:
            return
        self.pen = True
        self.current = [path.moveto(self.x, self.y)]
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
            self.current.append(path.lineto(self.x, self.y))
        return self

    def goto(self, x=None, y=None, angle=None):
        if x is not None and y is None:
            y = self.y
        elif y is not None and x is None:
            x = self.x
        if x is not None and y is not None:
            self.lookat(x, y)
            #self.ps.append((x, y))
            self.current.append(path.lineto(x, y))
            self.x = x
            self.y = y
        if angle is not None:
            self.theta = (angle/360.)*2*pi
        return self
    moveto = goto

    def reverse(self, d):
        self.fwd(-d)
        return self
    back = reverse
    rev = reverse

    def right(self, angle, r=0.):
        if abs(r)>EPSILON:
            while angle > 30:
                self.right(30, r)
                angle -= 30
            while angle < -30:
                self.right(-30, r)
                angle += 30
        dtheta = (angle/360.)*2*pi
        theta = self.theta
        self.theta += dtheta
        if abs(r)<EPSILON or abs(dtheta)<EPSILON:
            return self
        ar = abs(r)
        x0, y0 = self.x, self.y
        t0 = theta
        t1 = theta+dtheta
        xc = x0 - r*sin(t0-pi/2)
        yc = y0 - r*cos(t0-pi/2)
        #self.cvs.fill(path.circle(xc, yc, 0.1), [color.rgb.grey])
        x1 = xc + r*sin(t1-pi/2)
        y1 = yc + r*cos(t1-pi/2)
        u = abs(angle) / 180
        #u = 0.2
        x01 = x0 + u*ar*sin(t0)
        y01 = y0 + u*ar*cos(t0)
        x10 = x1 - u*ar*sin(t1)
        y10 = y1 - u*ar*cos(t1)
        #p = arc_to_bezier(x0, y0, r, theta, theta+dtheta)
        #p = path.line(x, y, x1, y1)
        #p = path.curve(x0, y0, x01, y01, x10, y10, x1, y1)
        p = path.curveto(x01, y01, x10, y10, x1, y1)
        self.current.append(p)
        self.x = x1
        self.y = y1
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
        return self

    def hook_arrow(self, size=0.15, *arg, **kw):
        self.save()
        self.right(180., -0.5*size)
        self.reverse(0.1*size)
        self.restore()
        return self

    def arrow(self, size=0.1, angle=30., astyle="flat"):
        meth = getattr(self, astyle+"_arrow", None)
        assert meth is not None, "astyle %r not found"%astyle
        meth(size, angle)
        return self

    def _render(self, attrs=None, closepath=False, cvs=None, name="stroke", preserve=False):
        if attrs is None:
            attrs = self.attrs
        if cvs is None:
            cvs = self.cvs
        assert cvs is not None
        if closepath:
            self.current.append(path.closepath())
        if self.pen:
            self.paths.append(self.current)
        for items in self.paths:
            p = path.path(items)
            #print("Turtle._render", p)
            method = getattr(cvs, name)
            if name=="stroke":
                method(p, attrs)
            elif name=="fill":
                method(p, attrs)
            else:
                method(p)
        if not preserve:
            self.reset()
        return self

    def stroke(self, *args, **kw):
        kw["name"] = "stroke"
        self._render(*args, **kw)
        return self

    def fill(self, *args, **kw):
        kw["name"] = "fill"
        self._render(*args, **kw)
        return self

    def clip(self, *args, **kw):
        kw["name"] = "clip"
        self._render(*args, **kw)
        return self

    def stroke_preserve(self, *args, **kw):
        kw["name"] = "stroke"
        kw["preserve"] = True
        self._render(*args, **kw)
        return self

    def fill_preserve(self, *args, **kw):
        kw["name"] = "fill"
        kw["preserve"] = True
        self._render(*args, **kw)
        return self



# ----------------------------------------------------------------------------
#
#


def test():
    cvs = Canvas()

    t = Turtle(-2, 0, 0, cvs)
    for i in range(9):
        t.fwd(0.3).left(30).back(0.5).right(10)
        #t.penup()
        #t.fwd(0.2)
        #t.pendown()
    t.fill()

    #cvs.stroke(path.circle(0, 0, 1.), [color.rgb.grey])

    cvs.fill(path.circle(0, 0, 0.1), [color.rgb.red])

    t = cvs.turtle(0, 0, 0)
    t.right(30, 1.)
    cvs.fill(path.circle(t.x, t.y, 0.1), [color.rgb.blue])
    t.right(30, 1.)
    cvs.fill(path.circle(t.x, t.y, 0.1), [color.rgb.blue])
    t.left(30, 1.)
    cvs.fill(path.circle(t.x, t.y, 0.1), [color.rgb.blue])
    t.left(30, 1.)
    cvs.fill(path.circle(t.x, t.y, 0.1), [color.rgb.blue])
    t.stroke()

    for (x,y) in [(-0.5, -1.), (-1.5,-1)]:
        cvs.stroke(path.circle(x, y, 0.5), [color.rgb.red.alpha(0.5)])

    t = Turtle(2, -1, 0, cvs)
    t.left(180, 0.5)
    t.right(180, 0.5)
    t.left(180, 0.5)
    t.right(180, 0.5)
    t.right(180, 2)
    
    t.fill([color.rgb.grey.alpha(0.5)])

    x = -2
    t = cvs.turtle(x, 2).fwd(1).flat_arrow().stroke()
    x += 1
    t = cvs.turtle(x, 2).left(30, 1).curve_arrow().stroke()
    x += 1
    t = cvs.turtle(x, 2).left(30, 1).stroke().dart_arrow().stroke_preserve().fill()
    x += 1
    t = cvs.turtle(x, 2).left(30, 1).feather_arrow().stroke()
    x += 1
    t = cvs.turtle(x, 2).left(30, 1).bar_arrow().stroke()

    x = -2
    y = -2
    angle = 30
    for i in range(5):
        cvs.turtle(x, y).left(angle, 1).stroke()
        cvs.turtle(x, y-1).left(angle, 3).stroke()
        angle -= 5
        x += 0.5

    cvs.writePDFfile("turtle_test.pdf")
    


if __name__ == "__main__":
    test()

    print("OK\n")


