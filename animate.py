#!/usr/bin/env python3

import os

from bruhat.render import config
config(text="xelatex")

from bruhat.render.base import SCALE_CM_TO_POINT

from bruhat.render.front import *
from bruhat.render.box import *
from bruhat.render.turtle import Turtle
from bruhat.argv import argv


"""
For the default 16:9 aspect ratio, encode at these resolutions:

2160p: 3840x2160
1440p: 2560x1440
1080p: 1920x1080
720p: 1280x720
480p: 854x480
360p: 640x360
240p: 426x240
"""

bg = color.rgb.white
bg = color.rgb.grey
fg = color.rgb.black

W, H = 854, 480
width, height = W/SCALE_CM_TO_POINT, H/SCALE_CM_TO_POINT
#print(width, height)


def arrow(cvs, x0, y0, x1, y1, size, attrs):
    t = Turtle(x0, y0, attrs=attrs)
    t.moveto(x1, y1)
    t.arrow(size, style="curve")
    t.stroke(cvs=cvs)


def setup():
    cvs = canvas.canvas()

    p = path.path([
        path.moveto(0., 0.),
        path.lineto(0., height),
        path.lineto(width, height),
        path.lineto(width, 0),
        path.closepath()])
    cvs.clip(p)
    cvs.fill(p, [bg])

    #cvs.stroke(path.line(0., 0., width, h), [fg])
    #cvs.stroke(path.line(0., h, width, 0), [fg])

    return cvs


def text_box(s, scale=5.):
    sub = canvas.canvas()
    sub.append(fg)
    sub.append(Scale(scale))
    sub.text(0., 0., s)
    box = CanBox(sub)
    box = AlignBox(box, "center")
    return box


def title_seq():

    x0, y0 = 0.5*width, 0.6*height
    x1, y1 = 0.6*width, 0.3*height

    #x0, y0 = 1.5*width, 1.6*height
    #x1, y1 = 1.6*width, 1.3*height
    N = 100

    for i in range(N):

        cvs = canvas.canvas()
    
        text_box("What the Quantum ?!?").render(cvs, x0, y0)
        text_box("Episode 1").render(cvs, x1, y1)

        x0 += 0.2/N*width
        x1 -= 0.2/N*width
    
        yield cvs



def axis(cvs, x, y, w, h, attrs):

    attrs = [style.linejoin.round, style.linecap.round] + attrs

    size = 0.7
    arrow(cvs, x-0.1*w, y, x+1.1*w, y, size, attrs)
    arrow(cvs, x, y-0.1*h, x, y+1.1*h, size, attrs)


class Axis(Canvas):
    def __init__(self, x0, y0, dx, dy, lw=0.02):
        Canvas.__init__(self)
        attrs = [fg, style.linewidth.THIck]
        axis(self, x0, y0, dx, dy, attrs)
        self.append(trafo.translate(x0, y0))
        self.append(trafo.scale(dx, dy))
        self.append(LineWidth(lw/dx))


def lattice_balls(r, theta):
    dx, dy = 2*sin(theta), 2*cos(theta)
    balls = []
    for i in range(7): # row
      for j in range(-3, 7): # col
        balls.append(((1. + dx*i + 2.*j)*r, (1. + dy*i)*r))
    return balls


def lin(i, x0, x1, N):
    assert 0<=i<N
    alpha = i / (N-1)
    return (1-alpha)*x0 + alpha*x1


def ilin(x0, x1, N):
    for i in range(N):
        assert 0<=i<N
        alpha = i / (N-1)
        yield (1-alpha)*x0 + alpha*x1


def iconst(x0, N):
    for i in range(N):
        yield x0


class Seq(object):
    def __init__(self, N):
        self.N = N
    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.N)
    __repr__ = __str__
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        raise IndexError
    @classmethod
    def promote(cls, item):
        if isinstance(item, Seq):
            return item
        if isinstance(item, list):
            return ConcSeq(item)
        raise TypeError
    def __add__(lhs, rhs):
        return AddSeq(lhs, rhs)
    def __mul__(lhs, rhs):
        return MulSeq(lhs, rhs)

class ConcSeq(Seq):
    def __init__(self, items):
        Seq.__init__(self, len(items))
        self.items = items
    def __getitem__(self, idx):
        return self.items[idx]


class PairSeq(Seq):
    def __init__(self, lhs, rhs):
        lhs = Seq.promote(lhs)
        rhs = Seq.promote(rhs)
        Seq.__init__(self, lhs.N + rhs.N)
        self.lhs = lhs
        self.rhs = rhs

class AddSeq(PairSeq):
    def __getitem__(self, idx):
        lhs, rhs = self.lhs, self.rhs
        if 0<=idx<lhs.N:
            return lhs[idx]
        elif lhs.N<=idx<lhs.N+rhs.N:
            return rhs[idx-lhs.N]
        else:
            raise IndexError

class MulSeq(PairSeq):
    def __getitem__(self, idx):
        lhs, rhs = self.lhs, self.rhs
        return (lhs[idx], rhs[idx])


class Lin(Seq):
    def __init__(self, x0, x1, N):
        Seq.__init__(self, N)
        self.x0 = x0
        self.x1 = x1
    def __getitem__(self, idx):
        N = self.N
        if idx<0 or idx>=N:
            raise IndexError
        x0, x1 = self.x0, self.x1
        alpha = idx / (N-1)
        return (1-alpha)*x0 + alpha*x1


class Smooth(Lin):
    def __getitem__(self, idx):
        N = self.N
        if idx<0 or idx>=N:
            raise IndexError
        x0, x1 = self.x0, self.x1
        x = idx / (N-1) # 0.<=x<=1.
        y = -2*(x**3) + 3*(x**2) # 0.<=y<=1.
        return (1-y)*x0 + y*x1


def ball_seq():

    N = 100

    blue = color.rgb(0.4, 0.3, 0.9, 1.0)
    orange = color.rgb(0.8, 0.2, 0.2, 1.0)

    r = 1./12

    rs = Smooth(r, 0.5*r, 2*N)
    thetas = Smooth(0., pi/6, N) + Smooth(pi/6, 0., N)
    for (r, theta) in rs*thetas:

        x, y, w = 0.1*width, 0.1*height, 0.7*height
        axis = Axis(x, y, w, w, lw=0.04)

        p = path.rect(0, 0, 1., 1.)
        axis.stroke(p, [linestyle.dashed, orange])
        axis.clip(p)

        balls = lattice_balls(r, theta)
        for (x1, y1) in balls:
            p = path.circle(x1, y1, r)
            axis.fill(p, [blue])
            axis.stroke(p)

        yield axis


def main():

    frames = argv.get("frames", None)

    frame = 0
    seqs = [
        #title_seq,
        ball_seq,
    ]
    for seq in seqs:
      for cvs in seq():
        main = setup()
        main.append(cvs)

        name = "ep01/%.4d"%frame
        main.writePNGfile("%s.png"%name)
        if frame==0:
            main.writePDFfile("%s.pdf"%name)

        print(".", end="", flush=True)
        frame += 1

        if frames is not None and frame>frames:
            break

    print("OK")


if __name__ == "__main__":

    main()




