#!/usr/bin/env python3

"""


"""

from huygens.namespace import *
from huygens.argv import argv

from huygens import pov
from huygens.pov import View, Mat

from huygens.wiggle import Cell0, Cell1, Cell2

def make_pants(m=None, i=None):

    if i is None:
        i = Cell0("i", stroke=None)
    if m is None:
        m = Cell0("m", fill=grey)

    #i = Cell0("i", stroke=blue)
    m_m = Cell1(m(skip=True), m(skip=True), 
        stroke=None, pip_color=None, 
        _width=0.001, skip=True, weight=1.)
    saddle = Cell2(
        Cell1(m@m, i) << Cell1(i, m@m),
        m_m @ m_m,
        pip_color=None,
        cone=0.6,
    )
    m_m = Cell1(m, m, stroke=None, pip_color=None)
    lfold = Cell2(Cell1(i, m@m), Cell1(i, m@m), cone=1.0, pip_color=None)
    rfold = Cell2(Cell1(m@m, i), Cell1(m@m, i), cone=1.0, pip_color=None)
    pants = lfold << saddle << rfold

    def constrain_pants(cell, system):
        #print("constrain_pants", cell)
        add = system.add
        left, saddle, right = cell
        w0 = right.src.pip_x - left.src.pip_x # pant waist
        w1 = saddle.tgt[0].pip_x - left.tgt.pip_x # left pant leg
        w2 = right.tgt.pip_x - saddle.tgt[1].pip_x # right pant leg
        w3 = saddle.tgt[1].pip_x - saddle.tgt[0].pip_x # space between legs
        add(w0 == w1)
        add(w0 == w2)
        add(0.8*w0 <= w3)

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
        add(saddle.tgt[1].pip_x - saddle.tgt[0].pip_x >= 2*w0) # space between legs
        add(left.tgt.pip_x - left.tgt.tgt.pip_x >= 0.5*w0) # space to the left of left leg
        add(right.tgt.src.pip_x - right.tgt.pip_x >= 0.5*w0) # space to the right of right leg
    pants = pants(on_constrain = constrain_pants, assoc=False)
    return pants


def make_tube(m=None, i=None):
    if i is None:
        i = Cell0("i", stroke=None)
    if m is None:
        m = Cell0("m", fill=grey)

    lfold = Cell2(Cell1(i, m@m), Cell1(i, m@m), cone=1.0, pip_color=None)
    rfold = Cell2(Cell1(m@m, i), Cell1(m@m, i), cone=1.0, pip_color=None)
    tube = lfold << rfold
    def constrain_tube(cell, system):
        #print("constrain_tube", cell)
        add = system.add
        left, right = cell
        add(left.src.pip_x == left.tgt.pip_x)
        add(right.src.pip_x == right.tgt.pip_x)
        #add(right.src.pip_x - left.src.pip_x == right.tgt.pip_x - left.tgt.pip_x)
    tube = tube(on_constrain = constrain_tube, assoc=False)
    return tube


