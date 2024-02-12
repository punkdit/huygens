#!/usr/bin/env python3

"""

"""

from functools import reduce
import operator 
from math import pi, sin, cos
from time import sleep, time
start_time = time()

import warnings
warnings.filterwarnings('ignore')

from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

import huygens
from huygens.namespace import *
del conv # !
from huygens.argv import argv


huygens.config(text="pdflatex", latex_header=r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{extarrows}
\usepackage{mathtools}
\arraycolsep=3.0pt\def\arraystretch{0.6}
""")

from huygens.sat import Const, System, Listener

def conv(a, b, alpha=0.5):
    return (1.-alpha)*a + alpha*b



class Layout(Listener):
    """
        each occurance of a 
        Box in a Compound will get a Layout object when it comes time to render
    """
    def __init__(self, system, box):
        self.system = system
        self.box = box
        #self.arrays = {}

    def __str__(self):
        return str(self.__dict__)
    __repr__ = __str__

    def on_update(self, name, value): # called from the system
        if type(name) is tuple:
            name, idx = name
            #self.arrays[name][idx] = value
            getattr(self, name)[idx] = value
        elif type(name) is str:
            setattr(self, name, value)
        else:
            assert 0, repr(name)

    def get_var(self, name, weight=0.01, *args, **kw):
        # weight: relative importance w.r.t. minimize
        assert type(name) is str
        system = self.system
        var = system.listen_var(self, name, weight=weight, *args, **kw)
        setattr(self, name, var)
        return var

    def get_array(self, name, n, weight=0.01, *args, **kw):
        # weight: relative importance w.r.t. minimize
        system = self.system
        vs = [system.listen_var(self, (name, i), weight=weight, *args, **kw) for i in range(n)]
        #self.arrays[name] = vs
        setattr(self, name, vs)
        return vs

    def add(self, item, weight=None):
        self.system.add(item, weight)

    def render(self, cvs):
        Box.on_render(self.box, self, cvs)
        self.box.on_render(self, cvs)



class Box(object):
    """
        A type of "box" that can have multiple occurances in a diagram.
        Each occurance will be given a unique Layout object, that
        contains specific coordinate variables for that occurance.
    """
    DEBUG = False

    st_stroke = st_Thick
    st_lstrokes = None
    st_rstrokes = None
    #min_width = 1.0
    #min_height = 1.0
    min_width = None
    min_height = None
    fill = None # top-level bg fill
    size = None # top-level size constraint
    rotate = 0 # top-level rotate

    def __init__(self, nleft=1, nright=1, **kw):
        self.nleft = nleft
        self.nright = nright
        self.shape = (nleft, nright)
        self.__dict__.update(kw)

    def __str__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.nleft, self.nright)
    __repr__ = __str__

    def __mul__(self, other):
        assert isinstance(other, Box)
        assert self.nright == other.nleft, "%s * %s"%(self, other)
        if isinstance(self, HBox) and isinstance(other, HBox):
            boxs = self.boxs + other.boxs
        elif isinstance(self, HBox):
            boxs = self.boxs + [other]
        elif isinstance(other, HBox):
            boxs = [self] + other.boxs
        else:
            boxs = [self, other]
        return HBox(self.nleft, other.nright, boxs)

    def __add__(self, other):
        assert isinstance(other, Box)
        #return VBox(self.nleft+other.nleft, self.nright+other.nright, [self, other])
        #return VBox(other.nleft+self.nleft, other.nright+self.nright, [other, self])
        #self, other = other, self
        if isinstance(self, VBox) and isinstance(other, VBox):
            boxs = self.boxs + other.boxs
        elif isinstance(self, VBox):
            boxs = self.boxs + [other]
        elif isinstance(other, VBox):
            boxs = [self] + other.boxs
        else:
            boxs = [self, other]
        return VBox(other.nleft+self.nleft, other.nright+self.nright, boxs)
    __lshift__ = __add__
    __matmul__ = __add__

    def get_hunits(self):
        return 1

    def get_vunits(self):
        return 1

    def on_constrain(self, layout):
        # lower left coord is (x0,y0)
        x0 = layout.get_var("x0")
        y0 = layout.get_var("y0")
        width = layout.get_var("width")
        height = layout.get_var("height")
        # y positions of left and right ports (legs)
        lys = layout.get_array("lys", self.nleft)
        rys = layout.get_array("rys", self.nright)
        # tangents for the left and right ports
        # XXX
        # This is still not perfect, 
        # we really need two coordinates for a tangent.
        # Problems arise when glueing two Spiders with different widths.
        # XXX
        ldys = layout.get_array("ldys", self.nleft, weight=0.)
        rdys = layout.get_array("rdys", self.nright, weight=0.)
        add = layout.add
        add(width >= 0.)
        add(height >= 0.)
        if self.min_width is not None:
            add(width >= self.min_width)
            print(self.__class__.__name__, "width >=", self.min_width)
        if self.min_height is not None:
            add(height >= self.min_height)
        for i in range(self.nleft):
            #add(y0 <= lys[i])
            #add(lys[i] <= y0+height)
            add(lys[i] == y0 + ((i+0.5)/self.nleft) * height, 10.0)
            add(ldys[i] == 0., 0.1) # this cuts down on extra wiggle
        for i in range(self.nright):
            #add(y0 <= rys[i])
            #add(rys[i] <= y0+height)
            add(rys[i] == y0 + ((i+0.5)/self.nright) * height, 10.0)
            add(rdys[i] == 0., 0.1) # this cuts down on extra wiggle

    def on_render(self, layout, cvs):
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        if self.DEBUG:
            #cvs.stroke(path.rect(x0, y0, width, height), [grey.alpha(0.5)])
            r = 0.05
            st = st_THick+[grey.alpha(0.5)]
            cvs.stroke(path.rect(x0+r, y0+r, width-2*r, height-2*r), st)
            #cvs.stroke(path.line(x0, y0, x0+width, y0+height), st)
            #cvs.stroke(path.line(x0+width, y0, x0, y0+height), st)

    def constrain(self, system):
        layout = Layout(system, self)
        self.on_constrain(layout)
        return layout

    def render(self, x0=0., y0=0., width=None, height=None, size=None, scale=1.0, border=0.1):
        # this is the top-level render call
        system = System()
        add = system.add
        layout = self.constrain(system)
        add(layout.x0 == x0)
        add(layout.y0 == y0)

        # require external ports perpendicular to the left and right edges
        for v in layout.ldys + layout.rdys:
            add(v==0, 1.)

        # top-level size constraint
        if size is None:
            size = self.size

        if width is not None:
            add(layout.width == width)
        elif size is not None:
            add(layout.width == size)
        elif scale is not None:
            width = scale * (self.get_hunits()**0.5)
            add(layout.width == width)

        if height is not None:
            add(layout.height == height)
        elif size is not None:
            add(layout.height == size)
        elif scale is not None:
            height = scale * (self.get_vunits()**0.5)
            add(layout.height == height)

        system.solve(simplify=False, verbose=False) # TODO: simplify=True
        cvs = Canvas()
        layout.render(cvs)
        if self.fill is not None:
            bg = Canvas()
            bg.fill(path.rect(layout.x0, layout.y0, layout.width, layout.height), [self.fill])
            cvs = Canvas([bg, cvs])
        if border is not None:
            bb = cvs.get_bound_box()
            p = path.rect(bb.llx-border, bb.lly-border, 
                bb.width+2*border, bb.height+2*border)
            cvs.stroke(p, [white])
        if self.rotate:
            cvs = Canvas([Rotate(self.rotate), cvs])
        return cvs

    def _repr_svg_(self):
        cvs = self.render()
        svg = cvs._repr_svg_()
        return svg


class Compound(Box):
    def __init__(self, nleft, nright, boxs=[], **kw):
        Box.__init__(self, nleft, nright, **kw)
        # _assoc ?? no not here, sometimes we don't want _assoc.
        self.boxs = list(boxs)

    def __getitem__(self, idx):
        return self.boxs[idx]

    def __len__(self):
        return len(self.boxs)

    def on_render(self, layout, cvs):
        for i, lay in enumerate(layout.lays):
            lay.render(cvs)


class HBox(Compound):
    st_sep = None # draw a separating line between boxes

    def get_hunits(self):
        return len(self)

    def get_vunits(self):
        return max(box.get_vunits() for box in self)

    def on_constrain(self, layout):
        system = layout.system
        Compound.on_constrain(self, layout)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0 # my origin
        x = x0
        n = len(self)
        lays = []
        add = system.add
        for i in range(n):
            lay = self[i].constrain(layout.system)
            add(lay.x0 == x)
            add(lay.y0 == y0)
            add(lay.height == height)
            x += lay.width
            if lays:
                add(lay.width == lays[-1].width, 1.0)  # bottom-up layout
            lays.append(lay)
        add(x0+width == x)
        for i in range(n-1):
            left, right = self[i], self[i+1]
            assert left.nright == right.nleft, "%s != %s"%(left.nright, right.nleft)
            l, r = lays[i], lays[i+1]
            #print(l.box, r.box)
            for j in range(left.nright):
                add(l.rys[j] == r.lys[j])
                add(l.rdys[j] == -r.ldys[j])
        layout.lays = lays
        # the ports on child Box's must be consistent with the ports on self
        for (l,r) in zip(layout.lys, lays[0].lys):
            add(l == r)
        for (l,r) in zip(layout.ldys, lays[0].ldys):
            add(l == r)
        for (l,r) in zip(layout.rys, lays[-1].rys):
            add(l == r)
        for (l,r) in zip(layout.rdys, lays[-1].rdys):
            add(l == r)
        return layout

    def on_render(self, layout, cvs):
        if self.DEBUG:
            width, height = layout.width, layout.height
            x0, y0 = layout.x0, layout.y0 # my origin
            r = 0.05
            cvs.stroke(path.rect(x0+2*r, y0+r, width-4*r, height-2*r),
                st_THick+[orange.alpha(0.5)])
        Compound.on_render(self, layout, cvs)
        st_sep = self.st_sep
        if st_sep is None:
            return # <------ return
        for i, lay in enumerate(layout.lays[1:]):
            p = path.line(lay.x0, lay.y0, lay.x0, lay.y0+lay.height)
            cvs.stroke(p, st_sep)


class VBox(Compound):
    """
    Arrange box's vertically, up the page.
    """
    def get_hunits(self):
        return max(box.get_hunits() for box in self)

    def get_vunits(self):
        return len(self)

    def on_constrain(self, layout):
        system = layout.system
        Compound.on_constrain(self, layout)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0 # my origin at lower left corner
        y = y0
        n = len(self)
        lays = []
        add = system.add
        for i in range(n):
            lay = self[i].constrain(layout.system)
            add(lay.y0 == y)
            add(lay.x0 == x0)
            add(lay.width == width)
            y += lay.height
            if lays:
                add(lay.height == lays[-1].height, 1.0)  # bottom-up layout
            lays.append(lay)
        add(y0+height == y)

        # the ports on child Box's must match the ports on self
        i = j = 0
        for lay in lays:
            for v,dv in zip(lay.lys,lay.ldys):
                add(v == layout.lys[i])
                add(dv == layout.ldys[i])
                i += 1
            for v,dv in zip(lay.rys,lay.rdys):
                add(v == layout.rys[j])
                add(dv == layout.rdys[j])
                j += 1
        assert i==self.nleft, "i=%d, nleft=%d"%(i, self.nleft)
        assert j==self.nright, "j=%d, nright=%d"%(j, self.nright)
        layout.lays = lays
        return layout

    def on_render(self, layout, cvs):
        if self.DEBUG:
            width, height = layout.width, layout.height
            x0, y0 = layout.x0, layout.y0 # my origin
            r = 0.05
            cvs.stroke(path.rect(x0+r, y0+2*r, width-2*r, height-4*r),
                st_THick+[blue.alpha(0.5)])
        Compound.on_render(self, layout, cvs)



class Spider(Box):
    def __init__(self, nleft=1, nright=1, **kw):
        Box.__init__(self, nleft, nright, **kw)
        self.__dict__.update(kw)

    def on_constrain(self, layout):
        Box.on_constrain(self, layout)
        add = layout.add
        yc = layout.y0 + 0.5*layout.height
        for i in range(self.nleft):
            y, dy = layout.lys[i], layout.ldys[i]
            add(y + dy == conv(y, yc, 0.7), 1.0)
        for i in range(self.nright):
            y, dy = layout.rys[i], layout.rdys[i]
            add(y + dy == conv(y, yc, 0.7), 1.0)

    label = None
    pip_cvs = None
    pip_colour = black
    pip_radius = 0.1
    def on_render(self, layout, cvs):
#        if self.DEBUG:
#            Box.on_render(self, layout, cvs)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        xc = x0 + 0.5*width
        if self.shape == (1,1):
            yc = sum(layout.lys+layout.rys) / (self.nleft + self.nright)
        elif self.nleft == 1:
            yc = layout.lys[0]
        elif self.nright == 1:
            yc = layout.rys[0]
        elif self.nleft+self.nright:
            yc = sum(layout.lys+layout.rys) / (self.nleft + self.nright)
        else:
            yc = y0 + 0.5*height
        for i in range(self.nleft):
            y = layout.lys[i]
            dy = layout.ldys[i]
            p = path.curve(
                x0, y, 
                conv(x0, xc), y+dy, 
                conv(x0, xc), conv(y, yc),
                xc, yc)
            st = self.st_stroke
            if self.st_lstrokes is not None:
                st = st + self.st_lstrokes[i]
            cvs.stroke(p, st)
        x1 = x0+width
        for i in range(self.nright):
            y = layout.rys[i]
            dy = layout.rdys[i]
            p = path.curve(
                x1, y, 
                conv(x1, xc), y+dy, 
                conv(x1, xc), conv(y, yc),
                xc, yc)
            st = self.st_stroke
            if self.st_rstrokes is not None:
                st = st + self.st_rstrokes[i]
            cvs.stroke(p, st)
        if self.pip_cvs is not None:
            cvs.insert(xc, yc, self.pip_cvs)
        elif self.pip_colour is not None:
            p = path.circle(xc, yc, self.pip_radius)
            cvs.fill(p, [self.pip_colour])
            cvs.stroke(p)
        if self.label is not None:
            cvs.text(xc, yc - 1.2*self.pip_radius, self.label, st_north)


class Red(Spider):
    pip_colour = red

class Green(Spider):
    pip_colour = green

class Black(Spider):
    pip_colour = black

class White(Spider):
    pip_colour = white

class Hadamard(Spider):
    pip_radius = 0.1
    p = path.rect(-pip_radius, -pip_radius, 2*pip_radius, 2*pip_radius)
    pip_cvs = Canvas().fill(p, [yellow]).stroke(p)


class Rect(Box):
    def __init__(self, m, n, label=None, **kw):
        Box.__init__(self, m, n, **kw)
        self.label = label

    def on_render(self, layout, cvs):
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        p = path.rect(x0, y0, width, height)
        cvs.fill(p, [white])
        cvs.stroke(p)
        label = self.label
        if label is not None:
            x, y = x0+0.5*width, y0+0.5*height
            cvs.text(x, y, label, st_center)


class Relation(Box):

    rigid = True
    def __init__(self, A, lpip_cvs=None, rpip_cvs=None, st_strokes=None, **kw):
        A = numpy.array(A)
        m, n = A.shape
        Box.__init__(self, m, n, **kw)
        self.A = A
        if st_strokes is not None:
            st_strokes = numpy.array(st_strokes, dtype=object)
            assert st_strokes.shape == A.shape
        self.st_strokes = st_strokes
        self.lpip_cvs = lpip_cvs
        self.rpip_cvs = rpip_cvs

    @property
    def t(self):
        A = self.A.transpose()
        return Relation(A, self.rpip_cvs, self.lpip_cvs)

    def on_constrain(self, layout):
        Box.on_constrain(self, layout)
        if not self.rigid:
            return
        rys = layout.rys
        lys = layout.lys
        for v in layout.ldys + layout.rdys:
            layout.add(v==0)
        if len(rys) == len(lys):
            for (l,r) in zip(lys, rys):
                layout.add(l==r)
                

    def on_render(self, layout, cvs):
#        if self.DEBUG:
#            Box.on_render(self, layout, cvs)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        x1 = x0+width
        lys, rys = layout.lys, layout.rys
        ldys, rdys = layout.ldys, layout.rdys
        A = self.A
        st_strokes = self.st_strokes
        # index goes top down...
        lys = list(reversed(lys))
        rys = list(reversed(rys))
        ldys = list(reversed(ldys))
        rdys = list(reversed(rdys))
        for i in range(self.nleft):
          for j in range(self.nright):
            if not A[i,j]:
                continue
            #p = path.line(
            #    x0, lys[i], 
            #    x1, rys[j])
            p = path.curve(
                x0, lys[i], 
                conv(x0,x1), lys[i]+ldys[i],
                conv(x0,x1), rys[j]+rdys[i],
                x1, rys[j])
            st = self.st_stroke
            if st_strokes is not None:
                st = st + st_strokes[i,j]
            cvs.stroke(p, st)
        lpip_cvs = self.lpip_cvs
        rpip_cvs = self.rpip_cvs
        if lpip_cvs is not None:
            for i in range(self.nleft):
                cvs.insert(x0, lys[i], lpip_cvs)
        if rpip_cvs is not None:
            for i in range(self.nleft):
                cvs.insert(x1, rys[i], rpip_cvs)


class Identity(Relation):
    def __init__(self, **kw):
        A = [[1]]
        Relation.__init__(self, A, **kw)

            
def Permutation(perm, *args, **kw):
    n = len(perm)
    assert set(perm) == set(range(n)), "wup"
    A = numpy.zeros((n, n))
    for i,j in enumerate(perm):
        A[j,i] = 1
    return Relation(A, *args, **kw)


class Element(Relation):
    def __init__(self, n):
        A = numpy.identity(n, dtype=int)
        Relation.__init__(self, A)


class CNOT(Element):
    def __init__(self, n, idx, jdx):
        Element.__init__(self, n)
        self.idx = idx
        self.jdx = jdx

    def on_render(self, layout, cvs):
        Element.on_render(self, layout, cvs)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        lys = list(reversed(layout.lys)) # arghhh
        idx, jdx = self.idx, self.jdx
        x = x0 + 0.5*width
        yi, yj = lys[idx], lys[jdx]
        cvs.stroke(path.line(x, yi, x, yj), self.st_stroke)
        cvs.insert(x, yi, Circuit.gcvs)
        cvs.insert(x, yj, Circuit.rcvs)


class CZ(Element):
    def __init__(self, n, idx, jdx):
        Element.__init__(self, n)
        self.idx = idx
        self.jdx = jdx

    def on_render(self, layout, cvs):
        Element.on_render(self, layout, cvs)
        width, height = layout.width, layout.height
        x0, y0 = layout.x0, layout.y0
        lys = list(reversed(layout.lys)) # arghhh
        idx, jdx = self.idx, self.jdx
        x = x0 + 0.5*width
        yi, yj = lys[idx], lys[jdx]
        if yi > yj:
            yi, yj = yj, yi
        cvs.stroke(path.line(x, yi, x, yj), self.st_stroke)
        cvs.insert(x, yi, Circuit.gcvs)
        cvs.insert(x, yj, Circuit.gcvs)
        if (idx+jdx)%2:
            y = conv(yi, yj)
            cvs.insert(x, y, Circuit.ycvs)
        else:
            y = conv(yi, yj) - 0.5*abs(yj-yi)/abs(idx-jdx)
            cvs.insert(x, y, Circuit.ycvs)


class Circuit(object):

    RED = color.rgb(0.9, 0.2, 0.1)
    GREEN = color.rgb(0.3, 0.7, 0.2)
    YELLOW = color.rgb(0.9, 0.9, 0.0)
    
    # https://zxcalculus.com/accessibility.html
    #RED = color.rgb(232/255, 165/255, 165/255)
    #GREEN = color.rgb(216/255, 248/255, 216/255)
    # YUCK !

    radius = Spider.pip_radius
    p = path.circle(0, 0, radius)
    rcvs = Canvas().fill(p, [RED]).stroke(p)
    gcvs = Canvas().fill(p, [GREEN]).stroke(p)
    bcvs = Canvas().fill(p, [black]).stroke(p)
    wcvs = Canvas().fill(p, [white]).stroke(p)
    p = path.rect(-radius, -radius, 2*radius, 2*radius)
    ycvs = Canvas().fill(p, [YELLOW]).stroke(p)

    @classmethod
    def get_phase(cls, phase, pip_cvs):
        cvs = Canvas()
        cvs.text(0, -1.2*cls.radius, "$%s$"%phase, st_north)
        cvs = Canvas([cvs, pip_cvs])
        return cvs

    def __init__(self, n):
        self.n = n

    def get_P(self, *args):
        assert len(args) == self.n
        return Permutation(args)

    def get_SWAP(self, idx=0, jdx=1, **kw):
        f = list(range(self.n))
        f[idx], f[jdx] = f[jdx], f[idx]
        return Permutation(f, **kw)

    def get_identity(self, **kw):
        f = list(range(self.n))
        return Permutation(f, **kw)

    def get_gate(self, idx, box, **kw):
        n = self.n
        if idx is None:
            boxs = [box]*n
        else:
            boxs = [Identity() if i!=idx else box for i in range(n)]
        boxs = reversed(list(boxs))
        return VBox(n, n, boxs, **kw)

    def get_H(self, idx=None):
        box = Spider(1, 1, pip_cvs=self.ycvs)
        return self.get_gate(idx, box)

    def get_S(self, idx=None):
        cvs = self.get_phase(1, self.gcvs)
        box = Spider(1, 1, pip_cvs=cvs)
        return self.get_gate(idx, box)

    def get_X(self, idx=None):
        cvs = self.get_phase(2, self.rcvs)
        box = Spider(1, 1, pip_cvs=cvs)
        return self.get_gate(idx, box)

    def get_Z(self, idx=None):
        cvs = self.get_phase(2, self.gcvs)
        box = Spider(1, 1, pip_cvs=cvs)
        return self.get_gate(idx, box)

    #def get_CZ(self, idx=0, jdx=1):

    def get_pair(self, idx, jdx, lbox, rbox):
        #if idx > jdx:
        #    return self.get_pair(jdx, idx, rbox, lbox) # recurse WHOOPS, no..
        assert idx < jdx
        assert lbox.shape == (1, 2)
        assert rbox.shape == (2, 1)
        n = self.n
        boxs = [Identity() if i!=idx else lbox for i in range(n)]
        boxs = reversed(list(boxs))
        lhs = VBox(n, n+1, boxs)
        boxs = [Identity() if i!=jdx else rbox for i in range(n)]
        boxs = reversed(list(boxs))
        rhs = VBox(n+1, n, boxs)
        assert idx < jdx
        perm = [None]*(n+1)
        for i in range(n):
            #print()
            #print("i =", i)
            #print("perm =", perm)
            if i < idx:
                #print("set A")
                perm[i] = i
            elif i == idx:
                #print("set B")
                perm[i] = i
                perm[i+1] = jdx
            elif i < jdx:
                #print("set C")
                perm[i+1] = i
            else:
                #print("set D")
                perm[i+1] = i+1
            #print("perm =", perm)
        assert None not in perm
        #print("perm:", perm)
        mid = Permutation(perm, rigid=False)
        mid = Relation(mid.A.transpose())
        #return lhs * mid * rhs
        return HBox(n, n, [lhs, mid, rhs])

    def get_CNOT(self, idx=0, jdx=1):
        return CNOT(self.n, idx, jdx)

    def ugly_get_CNOT(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        if idx < jdx:
            src = Spider(1, 2, pip_cvs=self.gcvs)
            tgt = Spider(2, 1, pip_cvs=self.rcvs)
            box = self.get_pair(idx, jdx, src, tgt)
        else:
            src = Spider(2, 1, pip_cvs=self.gcvs)
            tgt = Spider(1, 2, pip_cvs=self.rcvs)
            box = self.get_pair(jdx, idx, tgt, src)
        return box

    def get_CZ(self, idx=0, jdx=1):
        return CZ(self.n, idx, jdx)

    def ugly_get_CZ(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        if idx > jdx:
            idx, jdx = jdx, idx
        src = Spider(1, 2, pip_cvs=self.gcvs)
        src = src * (Identity() + Hadamard())
        tgt = Spider(2, 1, pip_cvs=self.gcvs)
        box = self.get_pair(idx, jdx, src, tgt)
        return box

    def get_expr(self, expr):
        if expr == ():
            op = self.get_identity()
        elif type(expr) is tuple:
            #print("get_expr", expr)
            op = reduce(operator.mul, [self.get_expr(e) for e in expr]) # recurse
        else:
            expr = "self.get_"+expr
            #print("\tget_expr", expr)
            op = eval(expr, {"self":self})
        return op

    def render_expr(self, expr, *args, **kw):
        print("render_expr", expr)
        op = self.get_expr(expr)
        cvs = op.render(*args, **kw)
        return cvs



def test():
    box = Red(2, 2) * Red(2, 2) + Red(2,2)*Red(2,2)
    box = (Red(2,2)+Red(2,2))*(Red(2,2)+Red(2,2))
    box = ((Red(2, 1) * Red(1, 2)) + Red(2, 2)) * (Red(4, 1) + Red(0,0))
    #box = ((Red(2, 1) * Red(1, 2)) + Red(2, 2)) * Red(4, 1)
    #box = (Red(1, 2) + Red(2, 2)) * Red(4, 1)
    #box = (Red(1, 2) + Red(2, 2)) * (Red(3, 1) + Red(1,1)) * Red(2,1)
    #box = box + Hadamard(1, 1)
    cvs = box.render(size=2)
    cvs.writePDFfile("test.pdf")

    box = Red(1,2) * Relation([[1,0],[1,1]]) * Green(2,1)

    idxs = [4, 2, 3, 1, 0]

    r = Spider.pip_radius
    lp = path.circle(r, 0, r)
    rp = path.circle(-r, 0, r)
    gpip = Canvas().fill(lp, [green]).stroke(lp)
    rpip = Canvas().fill(rp, [red]).stroke(rp)
    box = Red(0, len(idxs))*Permutation(idxs, gpip, rpip)*Green(len(idxs),1)
    #box += Green(1,1, label="$2$")
    I = Identity()
    box += I
    box = box * Relation([[1,0],[0,1]])

    mul = Green(1, 2)
    unit = Green(1, 0)
    #box = mul*(I<<unit)

    s = Circuit(5)
    box = (s.get_H(3) * s.get_S(1) * s.get_CNOT(0, 2) * s.get_CNOT(4, 1)
        * s.get_CZ(0, 1)
        * s.get_CZ(0, 2)
        * s.get_CZ(0, 3)
        * s.get_CZ(1, 3)
    )
    #box = s.get_CNOT(4,0)*s.get_CZ(1, 4)
    #box = s.get_CNOT(2, 3)


    #cvs = box.render(height=2.)
    #cvs = box.render()
    #cvs.writePDFfile("test.pdf")

def test_rect():
    r0 = 0.6
    r1 = 1.7
    c = Circuit(4)
    H = Spider(1, 1, pip_cvs=c.ycvs)
    boxs = [H, H, Identity(), Identity()]
    HH = VBox(4, 4, boxs, min_width=r0)
    SWAP = c.get_SWAP(2, 3, min_width=r0)
    I4 = c.get_identity()
    E = Rect(4, 4, "$[[4,2,2]]$", min_width=r1)
    rhs = I4 * E * SWAP * HH

    H4 = VBox(4, 4, [H]*4, min_width=r0)
    lhs = H4 * E * I4

    lhs = lhs.render(width=3.4)
    rhs = rhs.render(width=3.4)
    lbb = lhs.get_bound_box()
    rbb = rhs.get_bound_box()
    cvs = Canvas()
    x, y = lbb.east
    cvs.insert(-x-0.5, -y, lhs)
    x, y = rbb.west
    cvs.insert(-x+0.5, -y, rhs)
    cvs.text(0, 0, "$=$", st_center)

    cvs.writePDFfile("test.pdf")


if __name__ == "__main__":

    test_rect()
    print("OK\n\n")



