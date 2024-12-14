#!/usr/bin/env python3

from math import atan, pi, sqrt
from functools import reduce
from operator import mul, add
from time import sleep, time
from random import random, seed

import numpy

from huygens.back import arc_to_bezier, Compound, Translate, Empty
from huygens.front import *
from huygens.namespace import *
from huygens.turtle import Turtle
from huygens.pov import get_angle
from huygens.tool import rnd, conv, smooth, bump, clamp


#grey = color.rgb(0.8, 0.8, 0.8)
#medgrey = color.rgb(0.65, 0.65, 0.65)
#darkgrey = color.rgb(0.5, 0.5, 0.5)
#
#lgreen = color.rgb(0.2, 0.7, 0.2, 0.7)
#lblue = color.rgb(0.2, 0.2, 0.7, 0.7)


#EPSILON = 1e-8


def smoothrect(x0, y0, w, h, r):
    t = Turtle(x0, y0+r)
    t.fwd(h-2*r).right(90, r)
    t.fwd(w-2*r).right(90, r)
    t.fwd(h-2*r).right(90, r)
    t.fwd(w-2*r).right(90, r)
    p = t.mkpath(True)
    return p



def arrow(x0, y0, x1, y1, size, astyle="curve"):
    t = Turtle(x0, y0)
    t.goto(x1, y1)
    t.arrow(size, astyle=astyle)
    #t.stroke(cvs=cvs)
    p = t.mkpath()
    return p


#bgs = [
#    color.rgb(0.9, 0.8, 0.7),
#    #color.rgb(0.9, 0.7, 0.8), # pink :P
#    color.rgb(0.7, 0.7, 0.5),
#    color.rgb(0.8, 0.9, 0.7),
#    color.rgb(0.5, 0.8, 0.5),
#    color.rgb(0.7, 0.9, 0.8),
#]
#bg_idx = 0
#def get_bg():
#    global bg_idx
#    bg = bgs[bg_idx]
#    bg_idx = (bg_idx+1)%len(bgs)
#    return bg
#
#
#def i_workspace(state, child, x0, y0, w, h, bg=None, r=None):
#
#    width, height = state.width, state.height
#    if bg is None:
#        #bg = get_bg()
#        #bg = medgrey
#        bg = white
#
#    if r is None:
#        r = 0.02*w
#
#    for fg in child:
#
#        p_stroke = smoothrect(x0, y0, w, h, r)
#        attrs = [Translate(x0, y0), Scale(w/width, h/height)]
#
#        cvs = Canvas()
#        cvs.fill(p_stroke, [bg])
#
#        sub = Canvas()
#        sub.clip(p_stroke)
#        sub.extend(attrs)
#        sub.append(fg)
#        cvs.append(sub)
#        cvs.stroke(p_stroke, st_THick)
#
#        yield cvs
#
#workspace = i_workspace
#show_workspace = i_workspace
#
#
#def multi(*children):
#
#    N = len(children)
#    try:
#        while 1:
#            items = [child.__next__() for child in children]
#            if None in items:
#                break
#            #assert len(children) == N, len(children)
#            #assert len(items) == N, len(items)
#            cvs = Canvas()
#            cvs.extend(items)
#            yield cvs
#
#    except StopIteration:
#        print("multi: StopIteration")
#        return
#
#
#def show_multi(*children, accum=False):
#
#    N = len(children)
#    old = [None]*N
#    while 1:
#        items = []
#        found = False
#        for i, child in enumerate(children):
#            try:
#                item = child.__next__()
#                found = True
#            except StopIteration:
#                if not accum:
#                    return
#                item = old[i]
#            items.append(item)
#        if not found:
#            break
#        cvs = Canvas()
#        cvs.extend(items)
#        yield cvs
#        old = items
#
#
#
#def mouse_bubbles(state):
#
#    from animate import Animate
#
#    width, height = state.width, state.height
#    xc, yc = 0.5*width, 0.5*height
#
#    pts = []
#    anim = Animate(state)
#    def make_bubble(r0=0.5, r1=2.0):
#        cvs = Canvas()
#        a, b = 0.5, 1.0
#        cl = color.rgba(rnd(a, b), rnd(a, b), rnd(a, b), rnd(0., 0.5))
#        r = rnd(r0, r1)
#        x, y = rnd(-pip, pip), rnd(-pip, pip)
#        cvs.fill(path.circle(x, y, r), [cl])
#        return cvs
#
#    last = None
#    for cvs in anim.run(forever=True):
#
#        evts = state.get_mouse(True)
#        if not evts:
#            last = None
#        #if evts:
#        #    print(evts)
#        for (x, y, button) in evts:
#            #if button != 3:
#            #    continue
#            pip = 0.2
#            if last is not None:
#                x0, y0 = last
#                r = ((x-x0)**2 + (y-y0)**2)**0.5
#                #print("\tr =", r)
#                if r > 0.3:
#                    pip = 1.0
#            last = (x, y)
#            #cvs.stroke(path.circle(x, y, 1.))
#            bbl = make_bubble(0.2, pip)
#            sp = anim.make_sprite(bbl, x, y)
#            sp.popup(0., rnd(0.2, 1.0))
#            break
#
#        yield cvs
#
#show_mouse_bubbles = mouse_bubbles
#
#
#def mouse_draw(state):
#
#    from animate import Animate
#
#    width, height = state.width, state.height
#    xc, yc = 0.5*width, 0.5*height
#
#    pts = []
#    anim = Animate(state)
#    def make_line(x0, y0, x1, y1, pip=0.08):
#        #print("make_line", x0, y0, x1, y1)
#        cvs = Canvas()
#        a, b = 0.5, 1.0
#        cl = color.rgba(rnd(a, b), 0.5*rnd(a, b), 0.5*rnd(a, b), 1.)
#        cvs.stroke(path.line(x0, y0, x1, y1), [cl]+st_THICK)
#        cvs.fill(path.circle(x1, y1, pip), [cl])
#        return cvs
#
#    last = None
#    for cvs in anim.run(forever=True):
#
#        evts = state.get_mouse(True)
#        if not evts:
#            last = None
#        for (x, y, button) in evts:
#            if button != 3:
#                continue
#            pip = 0.2
#            if last is None:
#                last = (x, y)
#                continue
#            item = make_line(*last, x, y)
#            last = (x, y)
#            sp = anim.make_sprite(item)
#            sp.popup(0., 1.0)
#            break
#
#        yield cvs
#
#show_mouse_bubbles = mouse_bubbles
#
#
#def i_period(state, dt):
#    t0 = state.t
#    while state.t - t0 <= dt:
#        yield Canvas()
#
#
#def i_dropin(state, cvs, dt=0.5, src="north", keep=False):
#    t0 = state.t
#    width = state.width
#    height = state.height
#    assert src in "north south east west".split(), src
#    dx0, dy0 = {
#        "north" : (0, height),
#        "south" : (0, -height),
#        "east" : (-width, 0),
#        "west" : (+width, 0),
#    }[src]
#    while 1:
#        alpha = clamp((state.t - t0)/dt) # 0 .. 1
#        alpha = smooth(0., 1., alpha)
#        dx = conv(dx0, 0, alpha)
#        dy = conv(dy0, 0, alpha)
#        yield Canvas([Translate(dx, dy), cvs])
#        if not keep and state.t-t0 > dt:
#            break
#
#def i_dropin_keep(*args, **kw):
#    return i_dropin(*args, **kw, keep=True)
#
#
#def i_dropout(state, cvs, dt=0.5, keep=False):
#    t0 = state.t
#    height = state.height
#    while 1:
#        alpha = clamp((state.t - t0)/dt) # 0 .. 1
#        alpha = smooth(0., 1., alpha)
#        dy = conv(0, -height, alpha)
#        yield Canvas([Translate(0, dy), cvs])
#        if not keep and state.t-t0 > dt:
#            break
#
#
#def i_fade(state, fg, tgt=black, period=1.0, alpha0=0., alpha1=1., keep=False):
#    t0 = state.t
#    width = state.width
#    height = state.height
#    p = path.rect(0, 0, width, height)
#    while 1:
#        alpha = conv(alpha0, alpha1, (state.t - t0)/period)
#        alpha = clamp(alpha)
#        cvs = Canvas()
#        cvs.append(fg)
#        cvs.fill(p, [tgt.alpha(alpha)])
#        yield cvs
#        if not keep and abs(alpha - alpha1) < EPSILON:
#            break
#
#def i_fadeout(state, fg, tgt=black, period=1.0, keep=False):
#    return i_fade(state, fg, tgt, period, 0., 1., keep)
#
#def i_fadein(state, fg, tgt=black, period=1.0, keep=False):
#    return i_fade(state, fg, tgt, period, 1., 0., keep)



class Bound(object):
    def __init__(self, llx, lly, urx, ury):
        self.llx = llx
        self.lly = lly
        self.urx = urx
        self.ury = ury

    def __str__(self):
        return "[[%.2f %.2f %.2f %.2f]]" % ( self.llx, self.lly, self.urx, self.ury )

    def __repr__(self):
        return "Bound(%s)"%(self.__dict__,)

    def overlap(self, other): # with epsilon ?
        result = (self.llx <= other.urx and self.urx >= other.llx and 
             self.lly <= other.ury and self.ury >= other.lly)
        #print("%s ^ %s: %s"%(self, other, result))
        return result

    def dbg(self):
        debug.stroke(path.rect(self.llx, self.lly, self.urx-self.llx, self.ury-self.lly))

    @classmethod
    def frompath(cls, p, t0=0., t1=1., N=10):
        assert isinstance(p, Path)
        xs, ys = [], []
        for i in range(N):
            t = conv(t0, t1, i/(N-1))
            x, y, dx, dy = p.tangent(t)
            xs.append(x)
            ys.append(y)
        llx, lly, urx, ury = min(xs), min(ys), max(xs), max(ys)
        return cls(llx, lly, urx, ury)
        
        
def intersect1(pa, pb, ta0=0., ta1=1., tb0=0., tb1=1., N=10, threshold=1.0, tolerance=1e-4, dbg=False):

    count = 0
    while abs(ta1-ta0)>tolerance and abs(tb1-tb0)>tolerance:
        assert 0.<=ta0<ta1<=1.
        assert 0.<=tb0<tb1<=1.
    
        pta = []
        ptb = []
        A = numpy.zeros((N, N))
        for i in range(N):
            r = conv(ta0, ta1, i/(N-1))
            x, y, dx, dy = pa.tangent(r)
            pta.append((x, y))
            if dbg:
                debug.fill(path.circle(x, y, 0.1), [red])
            r = conv(tb0, tb1, i/(N-1))
            x, y, dx, dy = pb.tangent(r)
            ptb.append((x, y))
            if dbg:
                debug.fill(path.circle(x, y, 0.1), [red])
        for i, (x0, y0) in enumerate(pta):
          for j, (x1, y1) in enumerate(ptb):
            d = (x0-x1)**2 + (y0-y1)**2
            A[i, j] = d
        (i, j) = numpy.unravel_index(A.argmin(), A.shape)
        value = A[i, j]
        if value > threshold:
            return

        x0, y0 = pta[i]
        x1, y1 = ptb[j]
        #debug.fill(path.circle(x0, y0, 0.1), [green])
        #debug.fill(path.circle(x1, y1, 0.1), [green])
        if value < tolerance:
            ra, rb = conv(ta0, ta1, i/(N-1)),conv(tb0, tb1, j/(N-1))  
            return ra, rb
    
        a0 = max(0, i-1) / (N-1)
        a1 = min(N-1, i+1) / (N-1)
        b0 = max(0, j-1) / (N-1)
        b1 = min(N-1, j+1) / (N-1)
        ta0, ta1 = conv(ta0, ta1, a0), conv(ta0, ta1, a1)
        tb0, tb1 = conv(tb0, tb1, b0), conv(tb0, tb1, b1)

        count += 1
    

def intersect(pa, pb, ta0=0., ta1=1., tb0=0., tb1=1., stepsize=0.01, depth=0, dbg=False):

    #print("  "*depth, "intersect", ta0, ta1, tb0, tb1, dbg)
    if ta1-ta0 < stepsize and tb1-tb0 < stepsize:
        result = intersect1(pa, pb, ta0, ta1, tb0, tb1, dbg=dbg)
        if result is None:
            return []  #  <----- return
        r0, r1 = result
        x0, y0, dx, dy = pa.tangent(r0)
        pts = [(x0, y0)]
        #print("  "*depth, "intersect: return")
        return pts

    a0, a1, a2, a3 = [conv(ta0, ta1, r) for r in [0., 1/3, 2/3, 1.]]
    b0, b1, b2, b3 = [conv(tb0, tb1, r) for r in [0., 1/3, 2/3, 1.]]

    aint = [(a0, a1), (a1, a2), (a2, a3)]
    bint = [(b0, b1), (b1, b2), (b2, b3)]
    N = len(aint)
    abound = [Bound.frompath(pa, *ai) for ai in aint]
    bbound = [Bound.frompath(pb, *bi) for bi in bint]
    pts = []
    #for i in range(N):
    #    abound[i].dbg()
    #    bbound[i].dbg()
    for i in range(N):
      for j in range(N):
        if abound[i].overlap(bbound[j]):
            pts += intersect(pa, pb, *aint[i], *bint[j], depth=depth+1)
    return pts


def metric(x0, y0, x1, y1):
    d = sqrt((x0-x1)**2 + (y0-y1)**2)
    return d


def homotopy_path(pa, pb, t, N=12):
    # probably non-optimal, but _works...
    assert 0.<=t<=1.
    assert N>3
    if t<1e-8:
        return pa
    elif t>1-1e-8:
        return pb
    pts = []
    for i in range(N+1):
        s = i/N # 0. --> 1.
        xa, ya, dxa, dya = pa.tangent(s)
        xb, yb, dxb, dyb = pb.tangent(s)
        x, y = conv(xa, xb, t), conv(ya, yb, t)
        pts.append((x, y))

    return points_to_path(pts)
homotopy = homotopy_path # backward compat

def points_to_path(pts):
    assert len(pts)%2, len(pts)
    slow = 0.5
    idx = 0
    items = []
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    dx, dy = x1-x0, y1-y0
    items.append(MoveTo(x0, y0))
    while idx+3 < len(pts):
        x0, y0 = pts[idx]
        x1, y1 = pts[idx+1]
        x2, y2 = pts[idx+2]
        x3, y3 = pts[idx+3]
        dx1, dy1 = x2-x3, y2-y3
        xc1, yc1 = conv(x1, x0+slow*dx), conv(y1, y0+slow*dy)
        xc2, yc2 = conv(x1, x2+slow*dx1), conv(y1, y2+slow*dy1)
        item = CurveTo(xc1, yc1, xc2, yc2, x2, y2)
        #debug.fill(path.circle(x0, y0, 0.1), [black])
        #debug.fill(path.circle(x1, y1, 0.1), [green])
        #debug.fill(path.circle(x2, y2, 0.1), [black])
        #debug.fill(path.circle(x3, y3, 0.1), [green])
        #debug.fill(path.circle(xc1, yc1, 0.1), [orange])
        #debug.fill(path.circle(xc2, yc2, 0.1), [orange])
        items.append(item)
        dx, dy = x2-x1, y2-y1
        idx += 2

    x0, y0 = pts[idx]
    x1, y1 = pts[idx+1]
    x2, y2 = pts[idx+2]
    xc1, yc1 = conv(x1, x0+slow*dx), conv(y1, y0+slow*dy)
    xc2, yc2 = x1, y1
    item = CurveTo(xc1, yc1, xc2, yc2, x2, y2)
    items.append(item)
    #debug.fill(path.circle(x0, y0, 0.2), [black])
    #debug.fill(path.circle(x1, y1, 0.2), [green])
    #debug.fill(path.circle(x2, y2, 0.2), [black])
    return Path(items)
pts_to_path = points_to_path # backward compat

#
#def stopat(state, tms, frames, *args, **kw):
#    "stopat: tms a list of floats, frames is a callable with args and kw"
#    proxy = state.push()
#    child = frames(proxy, *args, **kw)
#
#    for cvs in child:
#        yield cvs
#
#        if tms and proxy.t >= tms[0]:
#            proxy.freeze()
#            tms.pop(0)
#            print("FREEZE")
#
#        if state.get_key(True):
#            if proxy.frozen:
#                print("RESUME")
#                proxy.resume()
#            else:
#                break
#
#
