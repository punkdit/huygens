#!/usr/bin/env python3

"""

make some shapes: pair of pants, tube's, etc.

"""

from huygens.namespace import *
from huygens.argv import argv

from huygens import pov
from huygens.pov import View, Mat

from huygens.wiggle import Cell0, Cell1, Cell2

def make_pants_rev(m=None, i=None, cone=1.0):

    if i is None:
        i = Cell0("i", stroke=None)
    if m is None:
        m = Cell0("m", fill=grey)

    mm = Cell1(m(skip=True), m(skip=True), 
        stroke=None, pip_color=None, 
        _width=0.001, skip=True)
    m_m = Cell1(m, m, stroke=None, pip_color=None)
    i_mm = Cell1(i, m@m, pip_color=None)
    mm_i = Cell1(m@m, i, pip_color=None)
    saddle = Cell2( mm_i<<i_mm, mm @ mm, pip_color=None)

    m_m = Cell1(m, m, stroke=None, pip_color=None)
    lfold = Cell2(i_mm, i_mm, cone=cone, pip_color=None)
    rfold = Cell2(mm_i, mm_i, cone=cone, pip_color=None)
    pants = lfold << saddle << rfold

    def constrain_pants(cell, system):
        #print("constrain_pants", cell)
        add = system.add
        left, saddle, right = cell
        ws = [
            left.src.src[0].pip_x - left.src.pip_x,
            left.src.src[1].pip_x - left.src.pip_x,
            left.tgt.src[0].pip_x - left.tgt.pip_x,
            left.tgt.src[1].pip_x - left.tgt.pip_x,
            saddle.tgt[0].pip_x - saddle.tgt[0].tgt[0].pip_x,
            saddle.tgt[0].pip_x - saddle.tgt[0].tgt[1].pip_x,
            saddle.tgt[1].src[0].pip_x - saddle.tgt[1].pip_x,
            saddle.tgt[1].src[1].pip_x - saddle.tgt[1].pip_x,
            right.src.pip_x - right.src.tgt[0].pip_x,
            right.src.pip_x - right.src.tgt[1].pip_x,
            right.tgt.pip_x - right.tgt.tgt[0].pip_x,
            right.tgt.pip_x - right.tgt.tgt[1].pip_x,
        ]
        w0 = ws[0]
        for w in ws[1:]:
            add(w == w0)
        center = (1/2)*(saddle.tgt[0].pip_x + saddle.tgt[1].pip_x)
        for cell in saddle.src:
            add(cell.src.pip_x == cell.tgt.pip_x)
            add(cell.src.pip_x == center) # center the waist between the legs
        add(left.pip_x == 0.5*(left.tgt.pip_x + left.src.pip_x))
        add(right.pip_x == 0.5*(right.tgt.pip_x + right.src.pip_x))
        add(saddle.tgt[1].pip_x - saddle.tgt[0].pip_x >= 2*w0) # space between legs
        add(left.tgt.pip_x - left.tgt.tgt.pip_x >= 0.5*w0) # space to the left of left leg
        add(right.tgt.src.pip_x - right.tgt.pip_x >= 0.5*w0) # space to the right of right leg
    pants = pants(on_constrain = constrain_pants, assoc=False)
    return pants


def make_pants(m=None, i=None, cone=1.0):

    if i is None:
        i = Cell0("i", stroke=None)
    if m is None:
        m = Cell0("m", fill=grey)

    mm = Cell1(m(skip=True), m(skip=True), 
        stroke=None, pip_color=None, 
        _width=0.001, skip=True)

    m_m = Cell1(m, m, stroke=None, pip_color=None)
    i_mm = Cell1(i, m@m, pip_color=None)
    mm_i = Cell1(m@m, i, pip_color=None)

    saddle = Cell2( mm@mm, mm_i << i_mm, pip_color=None)

    lfold = Cell2(i_mm, i_mm, cone=cone, pip_color=None)
    rfold = Cell2(mm_i, mm_i, cone=cone, pip_color=None)
    pants = lfold << saddle << rfold

    def constrain_pants(cell, system):
        #print("constrain_pants", cell)
        add = system.add
        left, saddle, right = cell
        ws = [
            left.tgt.src[0].pip_x - left.tgt.pip_x,
            left.tgt.src[1].pip_x - left.tgt.pip_x,
            left.src.src[0].pip_x - left.src.pip_x,
            left.src.src[1].pip_x - left.src.pip_x,
            saddle.src[0].pip_x - saddle.src[0].tgt[0].pip_x,
            saddle.src[0].pip_x - saddle.src[0].tgt[1].pip_x,
            saddle.src[1].src[0].pip_x - saddle.src[1].pip_x,
            saddle.src[1].src[1].pip_x - saddle.src[1].pip_x,
            right.tgt.pip_x - right.tgt.tgt[0].pip_x,
            right.tgt.pip_x - right.tgt.tgt[1].pip_x,
            right.src.pip_x - right.src.tgt[0].pip_x,
            right.src.pip_x - right.src.tgt[1].pip_x,
        ]
        w0 = ws[0]
        for w in ws[1:]:
            add(w == w0)
        center = (1/2)*(saddle.src[0].pip_x + saddle.src[1].pip_x)
        for cell in saddle.tgt:
            add(cell.src.pip_x == cell.tgt.pip_x)
            add(cell.src.pip_x == center) # center the waist between the legs
        add(left.pip_x == 0.5*(left.tgt.pip_x + left.src.pip_x))
        add(right.pip_x == 0.5*(right.tgt.pip_x + right.src.pip_x))
        add(saddle.src[1].pip_x - saddle.src[0].pip_x >= 2*w0) # space between legs
        add(left.src.pip_x - left.src.tgt.pip_x >= 0.5*w0) # space to the left of left leg
        add(right.src.src.pip_x - right.src.pip_x >= 0.5*w0) # space to the right of right leg
    pants = pants(on_constrain = constrain_pants, assoc=False)
    return pants


def make_tube(m=None, i=None):
    if i is None:
        i = Cell0("i", stroke=None)
    if m is None:
        m = Cell0("m", fill=grey)

    i_mm = Cell1(i, m@m, pip_color=None)
    mm_i = Cell1(m@m, i, pip_color=None)
    lfold = Cell2(i_mm, i_mm, cone=1.0, pip_color=None)
    rfold = Cell2(mm_i, mm_i, cone=1.0, pip_color=None)
    tube = lfold << rfold
    def constrain_tube(cell, system):
        #print("constrain_tube", cell)
        add = system.add
        left, right = cell
        add(left.src.pip_x == left.tgt.pip_x)
        add(right.src.pip_x == right.tgt.pip_x)
        add(left.pip_x == 0.5*(left.tgt.pip_x + left.src.pip_x))
        add(right.pip_x == 0.5*(right.tgt.pip_x + right.src.pip_x))
        #add(right.src.pip_x - left.src.pip_x == right.tgt.pip_x - left.tgt.pip_x)
    tube = tube(on_constrain = constrain_tube, assoc=False)
    return tube


def test():

    m = Cell0("m", fill=grey.alpha(0.5))
    pop = make_pants(m)
    rpop = make_pants_rev(m)
    tube = make_tube(m)
    cell = rpop * tube * pop

    cell = cell.layout(height=1.)
    cvs = cell.render_cvs(pos="north")
    cvs.writePDFfile("pants.pdf")

    print("OK\n")


if __name__ == "__main__":

    test()


