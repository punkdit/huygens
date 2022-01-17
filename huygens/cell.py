#!/usr/bin/env python3

"""
Render 2-morphisms in a bicategory using sheet/string diagrams.

previous version: bicategory.py

"""

import copy
from random import random, randint, shuffle
import operator
from functools import reduce
from math import pi, sin, cos
from time import sleep

from huygens.sat import System, Listener, Variable
from huygens.front import Compound, Deco, Path, Canvas, Transform, Scale
from huygens.front import path, style, canvas, color
from huygens.argv import argv

from huygens import pov
from huygens.pov import View, Mat

EPSILON = 1e-6

def conv(a, b, alpha=0.5):
    return (1.-alpha)*a + alpha*b


"""

The three directions:
    H : horizontal : operator <<  : x coordinate : width property  : left, right
    D : depth      : operator @   : y coordinate : depth property  : front, back
    V : vertical   : operator *   : z coordinate : height property : top, bot

atributes:
              pip_x  pip_y  pip_z
    -ve dir:  left   back   bot
    +ve dir:  right  front  top

"""

black = (0,0,0,1)
grey = (0,0,0,0.2)

PIP = 0.001

class Shape(Listener):
    def __init__(self, name, weight=1.0, no_constrain=False, **kw):
        self.name = name
        self.weight = weight
        self.no_constrain = no_constrain
        self.__dict__.update(kw)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        assert isinstance(other, Shape)
        assert 0, "this is just too weak..."
        return self.name == other.name

    @property
    def key(self):
        return id(self)

    def clone(self, alias=False):
        for v in self.__dict__.values():
            assert not isinstance(v, Variable), "wup"
        if alias:
            return self
        image = copy.deepcopy(self)
        return image
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self][idx]

    def __contains__(self, other):
        if self is other:
            return True

    def index(self, other):
        if self is other:
            return 0
        assert 0, "not found"

    @property
    def depth(self):
        return self.back + self.front

    @property
    def height(self):
        return self.top + self.bot

    @property
    def width(self):
        return self.left + self.right

    @property
    def pip(self):
        return self.pip_x, self.pip_y, self.pip_z

    def listen_var(self, system, attr, weight=0.0, vmin=None):
        stem = self.__class__.__name__ + "." + attr
        v = system.listen_var(self, attr, stem, weight, vmin)
        assert getattr(self, attr, None) is None, str(self.__dict__)
        setattr(self, attr, v)
        return v

    def on_constrain(self, system):
        assert not self.no_constrain
        self.listen_var(system, "pip_x")
        self.listen_var(system, "pip_y")
        self.listen_var(system, "pip_z")

    system = None
    def constrain(self, x=0, y=0, z=0, width=None, depth=None, height=None):
        system = System()
        self.on_constrain(system)
        system.add(self.pip_x == x)
        system.add(self.pip_y == y)
        system.add(self.pip_z == z)
        if width is not None and hasattr(self, "left"):
            system.add(self.width == width)
        elif width is None and hasattr(self, "hunits"):
            system.add(self.width == 0.7*self.hunits)
        if height is not None and hasattr(self, "top"):
            system.add(self.height == height)
        elif height is None and hasattr(self, "vunits"):
            system.add(self.height == 0.5*self.vunits)
        if depth is not None:
            system.add(self.depth == depth)
        elif hasattr(self, "dunits"):
            system.add(self.depth == 0.5*self.dunits)
        return system

    def layout(self, *args, **kw):
        system = self.constrain(*args, **kw)
        system.solve()
        return system

    def longstr(self):
        s = self.__class__.__name__ + "("
        keys = list(self.__dict__.keys())
        keys.sort()
        for key in keys:
            value = self.__dict__[key]
            if type(value) is float:
                value = "%.3f"%value
            elif type(value) is str:
                value = repr(value)
            s += "%s=%s, "%(key, value)
        s = s[:-2]+")"
        return s
    __repr__ = longstr

    def render(self, view):
        pass

    def visit(self, callback, instance=None, **kw):
        if instance is None or self.__class__ == instance:
            callback(self)

    def search(self, **kw):
        cells = []
        def cb(cell):
            cells.append(cell)
        self.visit(cb, **kw)
        return cells

    def format(self, attr):
        if hasattr(self, attr):
            value = getattr(self, attr)
            if type(value) is float:
                value = "%.2f"%value
        else:
            value = "--"
        return "%5s"%value

    def deepstr(self, depth=0):
        s = "%s%s(%s %s %s %s:%s %s:%s %s:%s)"% (
            "  "*depth, 
            self.__class__.__name__,
            self.format("pip_x"),
            self.format("pip_y"),
            self.format("pip_z"),
            self.format("left"), self.format("right"),
            self.format("front"), self.format("back"),
            self.format("top"), self.format("bot"),
        )
        return s


class Compound(object):
    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx]

    def __contains__(self, other):
        for cell in self.cells:
            if cell is other:
                return True
        return False

    def index(self, other):
        for idx, cell in enumerate(self.cells):
            if cell is other:
                return idx
        assert 0, "not found"

    def _associate(self, items):
        cls = self.__class__
        itemss = [(item.cells if isinstance(item, cls) else [item])
            for item in items]
        items = reduce(operator.add, itemss, [])
        return items

    def on_constrain(self, system):
        #Shape.on_constrain(self, system)
        for cell in self.cells:
            cell.on_constrain(system)

    def render(self, view):
        Shape.render(self, view)
        for cell in self.cells:
            cell.render(view)

    def visit(self, callback, **kw):
        for child in self.cells:
            child.visit(callback, **kw)
        Shape.visit(self, callback, **kw)

    def deepstr(self, depth=0):
        lines = [Shape.deepstr(self, depth)]
        lines += [cell.deepstr(depth+1) for cell in self.cells]
        return "\n".join(lines)

    def dbg_render(self, cvs):
        for child in self.cells:
            child.dbg_render(cvs)

def setop(cls, opname, parent):
    def meth(left, right):
        return parent([left, right])
    setattr(cls, opname, meth)


# -------------------------------------------------------

class Cell0(Shape):
    "These are the 0-cells"

    colour = black
    show_pip = False

    def __init__(self, name, **kw):
        Shape.__init__(self, name, **kw)
        if "colour" in kw:
            self.colour = kw["colour"]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self][idx]

    def on_constrain(self, system):
        Shape.on_constrain(self, system)
        back = self.listen_var(system, "back")
        front = self.listen_var(system, "front")

        # put the pip in the middle
        system.add(back == front)



class DCell0(Compound, Cell0):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        name = "@".join(cell.name for cell in cells) or "ii"
        Cell0.__init__(self, name, **kw)
        self.cells = cells

    def on_constrain(self, system):
        Cell0.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        if not len(self):
            return
        add = system.add
        y = self.pip_y - self.front
        w = 1./len(self)
        for cell in self.cells:
            add(self.pip_x == cell.pip_x) # _align
            add(self.pip_z == cell.pip_z) # _align
            add(cell.pip_y == y + cell.front, self.weight) # soft equal
            add(cell.depth == w*self.depth, self.weight) # soft equal
            y += cell.depth
        add(self.pip_y + self.back == y, self.weight) # should be hard equal?


setop(Cell0, "__matmul__", DCell0)

Cell0.ii = DCell0([])



# -------------------------------------------------------


class Cell1(Shape):
    """
        These are the 1-cells.
    """

    colour = black
    show_pip = True
    pip_radius = 0.3

    def __init__(self, tgt, src, name=None, weight=1.0, alias=False, **kw):
        assert isinstance(tgt, Cell0)
        assert isinstance(src, Cell0)
        if name is None:
            name = "(%s<---%s)"%(tgt, src)
        Shape.__init__(self, name, weight, **kw)
        self.tgt = tgt.clone(alias)
        self.src = src.clone(alias)
        self.hom = (self.tgt, self.src)
        #if "colour" in kw:
        #    self.colour = kw["colour"]

    def get_paths(self):
        # a path is a list of [tgt, src]
        tgt, src = self.hom
        if not len(tgt) and not len(src):
            yield [self]
        elif not len(tgt):
            for j in src:
                yield [self, j]
        elif not len(src):
            for i in tgt:
                yield [i, self]
        else:
            for i in tgt:
              for j in src:
                yield [i, self, j]

    def on_constrain(self, system):
        Shape.on_constrain(self, system)
        back = self.listen_var(system, "back")
        front = self.listen_var(system, "front")
        left = self.listen_var(system, "left")
        right = self.listen_var(system, "right")

        # put the pip in the middle
        system.add(back == front, self.weight) # soft equal
        system.add(left == right, self.weight) # soft equal

        if self.__class__ != Cell1: # bit of a hack
            return # < --------- return

        tgt, src = self.tgt, self.src
        tgt.on_constrain(system)
        src.on_constrain(system)
        add = system.add
        add(tgt.pip_x == self.pip_x - self.left)
        add(src.pip_x == self.pip_x + self.right)
        for cell in [tgt, src]:
            add(cell.pip_z == self.pip_z) # hard equal
            add(cell.pip_y - cell.front == self.pip_y - self.front)
            add(cell.pip_y + cell.back == self.pip_y + self.back)
        #add(tgt.pip_z == self.pip_z)
        #add(src.pip_z == self.pip_z)
        #add(tgt.pip_y == self.pip_y)
        #add(src.pip_y == self.pip_y)
        #add(tgt.depth == self.depth)
        #add(src.depth == self.depth)

    def dbg_render(self, bg):
        cvs = Canvas()
        pip_x, pip_y, pip_z = self.pip_x, self.pip_y, self.pip_z
        tx = Transform(
            xx=1.0, yx=0.0, 
            xy=0.6, yy=0.5, 
            x0=0.0, y0=pip_z)
        cvs.append(tx)
        cvs.fill(path.circle(pip_x, pip_y, 0.05))
        for cell in self.tgt:
            cvs.stroke(path.line(pip_x, pip_y, cell.pip_x, cell.pip_y))
        for cell in self.src:
            cvs.stroke(path.line(pip_x, pip_y, cell.pip_x, cell.pip_y))
        cvs.stroke(path.rect(
            pip_x - self.left, pip_y - self.front, self.width, self.depth),
            [color.rgb(1,0,0)])
        bg.append(cvs)

    @classmethod
    def random(cls, tgt, src, depth=0):
        assert depth>=0
        n_tgt, n_src = len(tgt), len(src)
        assert isinstance(tgt, Cell0), repr(tgt)
        assert isinstance(src, Cell0), repr(src)
        if depth == 0:
            cell = Cell1(tgt, src)
        elif n_tgt>1 and n_src>1 and random() < 0.5:
            a = randint(1, n_tgt-1)
            b = randint(1, n_src-1)
            tensor = lambda items : reduce(operator.matmul, items)
            front = Cell1.random(tensor(tgt[:a]), tensor(src[:b]), depth-1)
            back = Cell1.random(tensor(tgt[a:]), tensor(src[b:]), depth-1)
            cell = front @ back
        else:
            a = randint(1, 3)
            mid = DCell0([Cell0("m")]*a)
            left = Cell1.random(tgt, mid, depth-1)
            right = Cell1.random(mid, src, depth-1)
            cell = left << right
        assert cell.tgt.name == tgt.name
        assert cell.src.name == src.name
        return cell

    def extrude(self, show_pip=False, **kw):
        return Cell2(self, self, show_pip=show_pip, **kw)

    def reassoc(self):
        yield [self]
        

class DCell1(Compound, Cell1):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        tgt = DCell0([cell.tgt for cell in cells], alias=True, no_constrain=True) # don't on_constrain this!
        src = DCell0([cell.src for cell in cells], alias=True, no_constrain=True) # don't on_constrain this!
        name = "@".join(cell.name for cell in cells)
        Cell1.__init__(self, tgt, src, name, alias=True, **kw) # already clone'd tgt, src
        self.cells = cells

    def get_paths(self):
        for cell in self.cells:
            for path in cell.get_paths():
                yield path

    def on_constrain(self, system):
        Cell1.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        add = system.add
        y = self.pip_y - self.front
        w = 1./len(self)
        for cell in self.cells:
            add(self.width == cell.width) # fit width
            add(cell.pip_x == self.pip_x) # _align pip_x
            add(cell.pip_z == self.pip_z) # _align pip_z
            add(cell.pip_y - cell.front == y)
            add(cell.depth == w*self.depth, self.weight)
            y += cell.depth
        add(self.pip_y + self.back == y, self.weight)

    def extrude(self, show_pip=False, **kw):
        cells = [cell.extrude(show_pip=show_pip, **kw) for cell in self.cells]
        return DCell2(cells, show_pip=show_pip, **kw)

    #def reassoc(self):

        

class HCell1(Compound, Cell1):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        tgt = cells[0].tgt
        src = cells[-1].src
        i = 0
        while i+1 < len(cells):
            if cells[i].src.name != cells[i+1].tgt.name:
                msg = ("can't compose %s and %s"%(cells[i], cells[i+1]))
                raise TypeError(msg)
            i += 1
        name = "<<".join(cell.name for cell in cells)
        Cell1.__init__(self, tgt, src, name, alias=True, **kw) # already clone'd tgt, src
        self.cells = cells

    def get_paths(self):
        #print("get_paths", self)
        cells = self.cells
        assert len(cells) >= 2, "wup"
        left = cells[0]
        if len(cells) > 2:
            right = HCell1(cells[1:], alias=True)
        else:
            right = cells[1]

        #print("get_paths", left, right)
        lpaths = [[] for _ in left.src]
        for lpath in left.get_paths():
            if isinstance(lpath[-1], Cell1):
                yield lpath
                continue 
            cell0 = lpath[-1]
            assert isinstance(cell0, Cell0)
            assert cell0 in left.src, "%s not in %s"%(cell0.key, left.src.key)
            idx = left.src.index(cell0)
            lpaths[idx].append(lpath)

        for rpath in right.get_paths():
            if isinstance(rpath[0], Cell1):
                yield rpath
                continue 
            cell0 = rpath[0]
            assert isinstance(cell0, Cell0)
            assert cell0 in right.tgt, "%s not in %s"%(cell0.key, right.tgt.key)
            idx = right.tgt.index(cell0)
            for lpath in lpaths[idx]:
                yield lpath + rpath



    def on_constrain(self, system):
        Cell1.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        add = system.add
        x = self.pip_x - self.left
        w = 1./len(self)
        for cell in self.cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            add(cell.pip_z == self.pip_z) # _align pip_z
            add(cell.pip_x - cell.left == x)
            add(cell.width == w*self.width, self.weight)
            x += cell.width
        add(self.pip_x + self.right == x) # hard eq

        i = 0
        while i+1 < len(self):
            lhs, rhs = self.cells[i:i+2]
            n = len(lhs.src)
            assert n == len(rhs.tgt)
            for l, r in zip(lhs.src, rhs.tgt):
                add(l.pip_y == r.pip_y) # hard eq !
            i += 1

    def extrude(self, show_pip=False, **kw):
        cells = [cell.extrude(show_pip=show_pip, **kw) for cell in self.cells]
        return HCell2(cells, show_pip=show_pip, **kw)
        


setop(Cell1, "__matmul__", DCell1)
setop(Cell1, "__lshift__", HCell1)

# -------------------------------------------------------


class Segment(object):
    def __init__(self, v0, v1, v2, v3, colour=(0,0,0,1)):
        self.vs = (v0, v1, v2, v3)
        self.colour = colour

    def __eq__(self, other):
        return self.vs == other.vs

    #def incident(self, other):
    #    return self == other or self.reversed == other

    @classmethod
    def mk_line(cls, v0, v2, **kw):
        #v0, v2 = Mat(v0), Mat(v2)
        v01 = 0.5*(v0+v2)
        v12 = 0.5*(v0+v2)
        return cls(v0, v01, v12, v2, **kw)

    @property
    def reversed(self):
        v0, v1, v2, v3 = self.vs
        return Segment(v3, v2, v1, v0, self.colour)

    def __getitem__(self, idx):
        return self.vs[idx]

    def __len__(self):
        return len(self.vs)


class Surface(object):
    def __init__(self, segments, colour=(0,0,0,1)):
        self.segments = list(segments)
        self.colour = colour

    def __getitem__(self, idx):
        return self.segments[idx]

    def __len__(self):
        return len(self.segments)

    @property
    def reversed(self):
        segments = [seg.reversed for seg in reversed(self.segments)]
        return Surface(segments, self.colour)

    @classmethod
    def mk_triangle(cls, v0, v1, v2, colour):
        segments = [
            Segment.mk_line(v0, v1),
            Segment.mk_line(v1, v2),
            Segment.mk_line(v2, v0),
        ]
        return Surface(segments, colour=colour)

    def incident(self, other):
        if self.colour != other.colour:
            return False
        for s in self.segments:
          s = s.reversed
          for o in other.segments:
            #if s.incident(o):
            if s==o:
                return True
        return False

    def join(self, other):
        left = list(self.segments)
        right = list(other.segments)
        for i, l in enumerate(left):
          for j, r in enumerate(right):
            if l!=r.reversed:
                continue
            segs = left[:i] + right[j+1:] + right[:j] + left[i+1:]
            return Surface(segs, self.colour)
        assert 0, "not incident"

    @staticmethod
    def merge(srfs):
        srfs = list(srfs)
        i = 0
        while i < len(srfs):
            j = i+1
            while j < len(srfs):
                right = srfs[j]
                left = srfs[i]
                rleft = left.reversed
                if left.incident(right):
                    left = left.join(right)
                    srfs[i] = left
                    srfs.pop(j)
                    j = i+1
                elif rleft.incident(right):
                    rleft = rleft.join(right)
                    srfs[i] = rleft
                    srfs.pop(j)
                    j = i+1

                else:
                    j += 1
            i += 1
        return srfs

    def render(self, view):
        view.add_surface(self.segments, fill=self.colour)
        if Cell2.DEBUG:
            for seg in self.segments:
                view.add_curve(*seg, stroke=(0,0,1,0.2), lw=1.0, epsilon=None)


# -------------------------------------------------------



class Cell2(Shape):
    "These are the 2-cells"

    DEBUG = False
    colour = black
    show_pip = True
    pip_radius = 0.5
    cone = 0.6 # closer to 1. is more cone-like

    def __init__(self, tgt, src, name=None, alias=False, **kw):
        assert isinstance(tgt, Cell1)
        assert isinstance(src, Cell1)
        assert tgt.src.name == src.src.name, "%s != %s" % (tgt.src, src.src)
        assert tgt.tgt.name == src.tgt.name, "%s != %s" % (tgt.tgt, tgt.src)
        if name is None:
            name = "(%s<===%s)"%(tgt, src)
        Shape.__init__(self, name, **kw)
        self.tgt = tgt.clone(alias)
        self.src = src.clone(alias)
        self.hom = (self.tgt, self.src)
        #if "colour" in kw:
        #    self.colour = kw["colour"]

    @property
    def center(self):
        return self.pip_x, self.pip_y, self.pip_z

    @property
    def rect(self):
        return (
            self.pip_x-self.left,  self.pip_y-self.front, self.pip_z-self.bot,
            self.pip_x+self.right, self.pip_y+self.back, self.pip_z+self.top,
        )

    @property
    def hunits(self):
        return 1

    @property
    def dunits(self):
        return 1

    @property
    def vunits(self):
        return 1

    def on_constrain(self, system):
        Shape.on_constrain(self, system)
        back = self.listen_var(system, "back")
        front = self.listen_var(system, "front")
        left = self.listen_var(system, "left")
        right = self.listen_var(system, "right")
        top = self.listen_var(system, "top")
        bot = self.listen_var(system, "bot")

        # put the pip in the middle
        system.add(back == front, self.weight) # soft equal
        system.add(left == right, self.weight) # soft equal
        system.add(top == bot, self.weight) # soft equal

        if self.__class__ != Cell2: # bit of a hack
            return # < --------- return

        tgt, src = self.tgt, self.src
        tgt.on_constrain(system)
        src.on_constrain(system)
        add = system.add
        add(tgt.pip_z == self.pip_z + self.top) # hard equal
        add(src.pip_z == self.pip_z - self.bot) # hard equal
        for cell in [tgt, src]:
            add(cell.pip_x - cell.left == self.pip_x - self.left) # hard equal
            add(cell.pip_x + cell.right == self.pip_x + self.right) # hard equal
            add(cell.pip_y - cell.front == self.pip_y - self.front) # hard equal
            add(cell.pip_y + cell.back == self.pip_y + self.back) # hard equal

    def dbg_render(self, cvs):
        self.tgt.dbg_render(cvs)
        self.src.dbg_render(cvs)

    def render(self, view):
        (x0, y0, z0, x1, y1, z1) = self.rect
        x01 = conv(x0, x1)
        y01 = conv(y0, y1)
        z01 = conv(z0, z1)
        add_line = lambda p0, p1, lw=1.0, stroke=(0,0,0,1) : view.add_line(
            Mat(p0), Mat(p1), lw=lw, stroke=stroke)

        tgt, src = self.tgt, self.src
        if self.DEBUG and 0:
            add_line((x0, y0, z0), (x0, y0, z1), stroke=(0,0,1,0.5))
            add_line((x1, y0, z0), (x1, y0, z1), stroke=(0,0,1,0.5))
            add_line((x0, y1, z0), (x0, y1, z1), stroke=(0,0,1,0.5))
            add_line((x1, y1, z0), (x1, y1, z1), stroke=(0,0,1,0.5))
            view.add_circle(Mat([x0,y0,z0]), 2.0, fill=(1,0,0,1))

        surfaces = []
        
        pip2 = Mat(self.pip)

        cone = 1. - self.cone
        cone = max(PIP, cone)

        def p_over(v1, v2):
            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
            return v12

        def seg_over(v1, v2):
            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
            line = Segment( v1, v1 + cone*(v12-v1), v2 + cone*(v12-v2), v2)
            return line

        def callback(self):
            assert self.__class__ == Cell1
            #print("callback", self.__class__.__name__, self)

            colour = self.colour
            pip1 = Mat(self.pip)

            #pip12 = p_over(pip1, pip2) # a point over pip1
            #line = Segment( pip1, pip1 + cone*(pip12-pip1), pip2 + cone*(pip12-pip2), pip2)
            line = seg_over(pip1, pip2)
            if colour is not None:
                view.add_curve(*line, lw=0.2, stroke=colour)
                # show spider (1-cell) pip
                if self.show_pip:
                    view.add_circle(pip1, self.pip_radius, fill=colour)

            tgt, src = self.tgt, self.src
            for cell in tgt:
                colour = cell.colour
                v = Mat(cell.pip)
                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
                leg = Segment(v, conv(v, vpip1), vpip1, pip1) # spider leg
                line2 = seg_over(v, pip2).reversed
                triangle = Surface([
                    #Segment.mk_line(pip2, v),
                    line2,
                    leg, 
                    line,  # pip1 --> pip2
                ], colour)
                surfaces.append(triangle)
                if abs(cell.pip_x - x0) < 2*PIP:
                    l_ports.append( (triangle[0], colour) )
                view.add_curve(*leg, stroke=black)

            for cell in src:
                colour = cell.colour
                v = Mat(cell.pip)
                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
                line2 = seg_over(v, pip2).reversed
                leg = Segment(v, conv(v, vpip1), vpip1, pip1)
                triangle = Surface([
                    #Segment.mk_line(pip2, v),
                    line2,
                    leg, 
                    line,  # pip1 --> pip2
                ], colour)
                surfaces.append(triangle)
                if abs(cell.pip_x - x1) < 2*PIP:
                    r_ports.append( (triangle[0], colour) )
                view.add_curve(*leg, stroke=black)

        # left and right ports
        l_ports = l_tgt = []
        r_ports = r_tgt = []
        z = z1
        tgt.visit(callback, instance=Cell1)

        # left and right ports
        l_ports = l_src = []
        r_ports = r_src = []
        z = z0
        src.visit(callback, instance=Cell1)

        #for cell in tgt.search(instance=Cell1)+src.search(instance=Cell1):
        #    if cell.show_pip:
        #        view.add_circle(Mat(cell.pip), cell.pip_radius, fill=cell.colour)


        if len(l_src) != len(l_tgt) or len(r_src) != len(r_tgt):
            print("Cell2.render: FAIL", id(self), end=" ")
            print( len(l_src) , len(l_tgt) , end=" ")
            print( len(r_src) , len(r_tgt) )
            for surface in surfaces:
                surface.render(view)
            return

        assert len(l_src) == len(l_tgt)
        assert len(r_src) == len(r_tgt)

        for (p_src, p_tgt) in zip(l_src, l_tgt):
            colour = p_src[1]
            seg_src, seg_tgt = p_src[0], p_tgt[0]
            seg = Segment.mk_line(seg_src[-1], seg_tgt[-1])
            surf = Surface([seg_src, seg, seg_tgt.reversed], colour)
            surfaces.append(surf)

        for (p_src, p_tgt) in zip(r_src, r_tgt):
            colour = p_tgt[1]
            seg_src, seg_tgt = p_src[0], p_tgt[0]
            seg = Segment.mk_line(seg_src[-1], seg_tgt[-1])
            surf = Surface([seg_src, seg, seg_tgt.reversed], colour)
            surfaces.append(surf)

        #surfaces = Surface.merge(surfaces)
        for surface in surfaces:
            surface.render(view)

        if self.colour is not None and self.show_pip:
            view.add_circle(Mat(pip2), self.pip_radius, fill=self.colour)

    @classmethod
    def random(cls, tgt0, src0, depth=0):
        tgt = Cell1.random(tgt0, src0, depth)
        src = Cell1.random(tgt0, src0, depth)
        return Cell2(tgt, src)


class DCell2(Compound, Cell2):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        tgt = DCell1([cell.tgt for cell in cells], alias=True, no_constrain=True)
        src = DCell1([cell.src for cell in cells], alias=True, no_constrain=True)
        name = "@".join(cell.name for cell in cells)
        Cell2.__init__(self, tgt, src, name, alias=True, **kw)
        self.cells = cells

    @property
    def hunits(self):
        return max(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return sum(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return max(cell.vunits for cell in self.cells)

    def on_constrain(self, system):
        Cell2.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        add = system.add
        y = self.pip_y - self.front
        w = 1./self.dunits
        for cell in self.cells:
            add(self.width == cell.width) # fit width
            add(self.height == cell.height) # fit width
            add(cell.pip_x == self.pip_x) # _align pip_x
            add(cell.pip_z == self.pip_z) # _align pip_z
            add(cell.pip_y - cell.front == y)
            add(cell.depth == w*cell.dunits*self.depth, self.weight)
            y += cell.depth
        add(self.pip_y + self.back == y, self.weight)

    def render(self, view):
        Shape.render(self, view)
        for cell in reversed(self.cells):
            cell.render(view)


class HCell2(Compound, Cell2):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        tgt = HCell1([cell.tgt for cell in cells], alias=True, no_constrain=True)
        src = HCell1([cell.src for cell in cells], alias=True, no_constrain=True)
        name = "<<".join(cell.name for cell in cells)
        Cell2.__init__(self, tgt, src, name, alias=True, **kw)
        self.cells = cells

    @property
    def hunits(self):
        return sum(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return max(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return max(cell.vunits for cell in self.cells)

    def on_constrain(self, system):
        Cell2.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        add = system.add
        cells = self.cells
        assert cells, "??"
        x = self.pip_x - self.left # start here
        w = 1./self.hunits
        for cell in self.cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            add(cell.pip_y == self.pip_y)

            #add(cell.pip_z+cell.top == self.pip_z+cell.top) 
            #add(cell.pip_z-cell.bot == self.pip_z-cell.bot) 
            add(cell.pip_z == self.pip_z)
            add(self.height == cell.height) # fit height

            add(cell.pip_x == cell.left + x)
            add(cell.width == w*cell.hunits*self.width, self.weight) # soft equal
            x += cell.width

        #add(self.pip_x + self.right == x, self.weight)
        add(self.pip_x + self.right == x) # hard equal
        #add(cell.pip_x + cell.right == x) # hard equal.. redundant ?

        i = 0
        while i+1 < len(self):
            lhs, rhs = self.cells[i:i+2]
            for (lhs, rhs) in [(lhs.tgt, rhs.tgt), (lhs.src, rhs.src)]:
                n = len(lhs.src)
                assert n == len(rhs.tgt)
                for l, r in zip(lhs.src, rhs.tgt):
                    add(l.pip_y == r.pip_y) # hard eq !
            i += 1


class VCell2(Compound, Cell2):
    def __init__(self, cells, alias=False, **kw):
        cells = self._associate(cells)
        cells = [cell.clone(alias) for cell in cells]
        tgt = cells[0].tgt
        src = cells[-1].src
        i = 0
        while i+1 < len(cells):
            if cells[i].src.name != cells[i+1].tgt.name:
                msg = ("can't compose\n%s and\n%s"%(cells[i], cells[i+1]))
                #raise TypeError(msg)
                print("VCell2.__init__: WARNING", msg)
            i += 1
        name = "*".join(cell.name for cell in cells)
        Cell2.__init__(self, tgt, src, name, alias=True, **kw)
        self.cells = cells

    @property
    def hunits(self):
        return max(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return max(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return sum(cell.vunits for cell in self.cells)

    def on_constrain(self, system):
        Cell2.on_constrain(self, system)
        Compound.on_constrain(self, system) # constrain children
        add = system.add
        z = self.pip_z - self.bot
        cells = self.cells # cells go top down
        cells = list(reversed(cells)) # now it's bottom up
        w = 1./self.vunits
        for cell in cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            add(cell.pip_y == self.pip_y)
            add(cell.pip_z == z + cell.bot)
            add(cell.pip_x - cell.left == self.pip_x - self.left)
            add(cell.pip_x + cell.right == self.pip_x + self.right)
            add(cell.height == w*cell.vunits*self.height, self.weight)
            z += cell.height
        add(self.pip_z + self.top == z) # hard equal

        i = 0
        while i+1 < len(cells):
            src, tgt = cells[i:i+2]
            src, tgt = src.tgt, tgt.src
            #print(src.deepstr())
            #print(tgt.deepstr())
            for (s, t) in match(src, tgt):
                add(s.pip_x == t.pip_x) # hard equal
                add(s.pip_y == t.pip_y) # hard equal
            i += 1



def match(src, tgt):
    if src.name == tgt.name:
        src = src.search(instance=Cell1)
        tgt = tgt.search(instance=Cell1)
        assert len(src) == len(tgt)
        for (s,t) in zip(src, tgt):
            yield (s, t)
        return # <---------- return
    assert 0, "todo"
    # hopefully we can use Cell1.get_paths ?!!??


setop(Cell2, "__matmul__", DCell2)
setop(Cell2, "__lshift__", HCell2)
setop(Cell2, "__mul__", VCell2)

# -------------------------------------------------------


#class DCell1(Cell1):


def test():
    l = Cell0("l")
    m = Cell0("m")
    n = Cell0("n")
    o = Cell0("o")
    p = Cell0("p")

    assert str(m@n) == "m@n", str(m@n)
    #assert m != n
    #assert m@m == m@m

    ii = Cell0.ii
    mm = m@m
    mmm = m@m@m
    mmmm = m@m@m@m
    A = Cell1(mm, mm)
    B = Cell1(m, m) @ Cell1(m, m)
    AA = A<<A

    f = Cell2(B, B)

    #cell = Cell2(B, AA) * Cell2(AA, A)
    #cell = cell @ (f*f*f)
    #cell = Cell2(B, A<<B)

    cell = Cell1(mm,mm)<<((Cell1(m,mmm)) @ Cell1(m,m)) << Cell1(mmmm,m)
    cell = Cell2(cell, Cell1(mm,m))

    mm_ = Cell1(mm, ii)
    _mm = Cell1(ii, mm)
    cell = Cell2(_mm, _mm) << Cell2(mm_, mm_)

    cell.layout(depth=1.2, width=2., height=1.4)

    cvs = Canvas()
    cell.dbg_render(cvs)
    #cvs.writePDFfile("debug.pdf")

    A, B = Cell1(m@n, l), Cell1(l, p)
    A1, B1 = Cell1(m@n, o@o), Cell1(o@o, p)
    A2, B2 = Cell1(m@n, p@l), Cell1(p@l, p)

    AB = A << B
    print(A@A)
    print(AB)

    f = Cell2(A, A)
    g = Cell2(B, B)
    print(f)
    print(f@f)
    print(f << g)

    f = Cell2(A<<B, A1<<B1)
    g = Cell2(A1<<B1, A2<<B2)
    print(f*g)


def main(state):

    # https://coolors.co
    scheme = "ff5e5b-d8d8d8-ffffea-00cecb-ffed66"
    scheme = scheme.split("-")
    scheme = [color.rgbhex(rgb).alpha(0.5) for rgb in scheme]

    names = 'lmnop'
    l, m, n, o, p = [Cell0(name, colour=scheme[i%len(scheme)]) for i, name in enumerate('lmnop')]
    i0 = Cell0("i", colour=None)

    ident = lambda top, bot : Cell2(Cell1(top, bot), Cell1(top, bot))

    #bot = Cell2(Cell1(m@m,i0)*Cell1(i0,m@m), Cell1(m,m)@Cell1(m,m))
    #top = Cell2(Cell1(m,m)@Cell1(m,m), Cell1(m@m,i0)*Cell1(i0,m@m))
    #morph = top*bot

    cell = Cell1(m, m@m) << Cell1(m@m, n)
    assert len(list(cell.get_paths())) == 2
    cell = cell @ cell
    assert len(list(cell.get_paths())) == 4

    # i don't think we can handle bubbles... 
    ii = Cell0.ii
    bubble = Cell1(ii, m@m) << Cell1(m@m, ii)

    cell = bubble
    assert len(list(cell.get_paths())) == 2

    cell = bubble<<bubble
    assert len(list(cell.get_paths())) == 4

    cell = bubble@bubble
    assert len(list(cell.get_paths())) == 4

    cell = Cell1(m, m) @ bubble
    assert len(list(cell.get_paths())) == 3

    morphisms = []

    Cell0.colour = (0,0,0,0.1)

    cell = None
    tgt = None
    for i in range(3):
        src = Cell1.random(m@m@m, m@m, depth=randint(1,3))
        if tgt is None:
            tgt = Cell1.random(m@m@m, m@m, depth=randint(1,3))
        if cell is None:
            cell = Cell2(tgt, src)
        else:
            cell = cell * Cell2(tgt, src)
        tgt = src
    #cell.layout(0,0,0,3,3,3)
    #morphisms.append(cell)

    # this one breaks no_constrain above
    #cell = Cell2.random(m@m@m, m@m, depth=2) << Cell2.random(m@m, m@m, depth=2)
    #cell = cell * Cell2(cell.src.clone(), Cell1.random(m@m@m, m@m, depth=2))

    if 0:
        a = Cell1.random(m@m@m, m@m, depth=2)
        b = Cell1.random(m@m@m, m@m, depth=2)
        c = Cell1.random(m@m, m@m, depth=2)
        d = Cell1.random(m@m, m@m, depth=2)
        e = b << c
        f = Cell1.random(m@m@m, m@m, depth=2)
    
        cell = Cell2(a, b) << Cell2(d, c)
        cell = cell * Cell2(e, f)
    
        cell.layout(0,0,0, 4, 4, 2.5)
    
        morphisms.append(cell)

    I_l = Cell1(l, l, show_pip=False, colour=None)
    I_m = Cell1(m, m, show_pip=False, colour=None)
    I_n = Cell1(n, n, show_pip=False, colour=None)
    I_o = Cell1(o, o, show_pip=False, colour=None)
    swap = lambda a,b : Cell1(a@b, b@a)
    S_ml = swap(m,l)
    S_nl = swap(n,l)
    S_nm = swap(n,m)

    # Yang-Baxter
    def yang_baxter(n, m, l, reversed=False, **kw):
        I_l = Cell1(l, l, show_pip=False, colour=None)
        I_m = Cell1(m, m, show_pip=False, colour=None)
        I_n = Cell1(n, n, show_pip=False, colour=None)
        tgt = (I_n @ swap(m, l)) << (swap(n, l) @ I_m) << (I_l @ swap(n, m))
        src = (swap(n, m) @ I_l) << (I_m @ swap(n, l)) << (swap(m, l) @ I_n) 
        if reversed:
            tgt, src = src, tgt
        morph = Cell2(tgt, src, cone=1.0, show_pip=False, **kw)
        return morph

    #tgt = (I_n @ I_m ) << S_nm
    #src = S_nm << (I_m @ I_n)
    #morph = Cell2(tgt, src)

    # a part of the Zamolodchikov Tetrahedron Equation
    lhs = (I_o @ ((I_n@swap(m,l))<<(swap(n,l)@I_m)) ).extrude(show_pip=False)
    rhs = (I_l @ ((swap(o,m)@I_n)<<(I_m@swap(o,n)) )).extrude(show_pip=False)

    tgt = swap(n,m) << (I_m @ I_n)
    src = (I_n @ I_m) << swap(n,m)
    back = Cell2(tgt, src, show_pip=False, cone=1.0)
    tgt = (I_o @ I_l) << swap(o,l)
    src = swap(o,l) << (I_l @ I_o)
    front = Cell2(tgt, src, show_pip=False, cone=1.0)

    morph_0 = lhs << (front @ back) << rhs

    rhs = I_l.extrude(show_pip=False) @ yang_baxter(o, n, m, reversed=True)
    lhs = (I_o @ I_n @ swap(m,l)) << (I_o @ swap(n,l) @ I_m) << (swap(o,l) @ I_n @ I_m)
    lhs = lhs.extrude()
    morph_1 = lhs << rhs

    #morph = morph_0 * morph_1 # Fails

    for morph in [morph_0, morph_1]:
        #morph.layout(0,0,0,4,2,0.5)
        morph.layout()
        morphisms.append(morph)

    for _ in range(1):
        morph = Cell2(
            Cell1(l,n) << Cell1(n,o) << Cell1(o, m),
            Cell1(l, p) << Cell1(p, m)
        )
        morph = Cell2(
            Cell1(l, m),
            Cell1(l,n) << Cell1(n,o) << Cell1(o, m),
        ) * morph
        morph.layout() #0, 0, 0, 2, 4, 3)
        morphisms.append(morph)
        #break
    
        morph = (ident(m,l)@ident(n,l)@ident(o,m)) << (ident(l@l,m)@ident(m,m))
        morph.layout() #0, 0, 0, 2, 4, 1)
        morphisms.append(morph)
    
        Cell1.show_pip = False
        Cell2.show_pip = False
        Cell1.show_pip = True
        Cell2.show_pip = True
    
        A = Cell1(m,n,colour=black)
        At = Cell1(n,m,colour=black)
        I_m = Cell1(m,m,colour=None)
        I_n = Cell1(n,n,colour=None)
    
        F_n = Cell1(n@n, i0, colour=None)  # fold
        G_n = Cell1(i0, n@n, colour=None)  # unfold
        F_m = Cell1(m@m, i0, colour=None)  # fold
        G_m = Cell1(i0, m@m, colour=None)  # unfold
    
        rfold = Cell2(
            (I_m@At) << F_m,
            (A@I_n) << F_n,
        )
        lfold = Cell2(
            G_n << (At @ I_n),
            G_m << (I_m @ A),
        )
        morph = lfold << rfold
        rfold = Cell2(
            (A@I_n) << F_n,
            (I_m@At) << F_m,
        )
        lfold = Cell2(
            G_m << (I_m @ A),
            G_n << (At @ I_n),
        )
        morph = (lfold<<rfold) * morph
        morph.layout() #0, 0, 0, 2, 4, 2)
        morphisms.append(morph)
    
        #Cell2.colour = None
    
        lcap = Cell2( I_m, A << At )
        lcup = Cell2( A << At, I_m )
        rcap = Cell2( I_n, At << A )
        rcup = Cell2( At << A, I_n )
        rcup2 = Cell2( At << A, I_n<<I_n )
        
        #morph = Cell2( I_m << A, A << I_n, colour=None )
    
        top = Cell2(A, A) << rcup2 << Cell2(At, At)
        #comul = comul * Cell2(A<<I_n<<At, A<<At)
        bot = (Cell2(A<<I_n, A) << Cell2(I_n<<At, At))
    
        morph = top * bot
        morph.layout() #0, 0, 0, 2, 4, 2)
        morphisms.append(morph)

        top = Cell2(
            #Cell1(i0, m@m) << Cell1(m@m, i0),
            Cell1(i0, i0, colour=None),
            Cell1(i0, m@m) << Cell1(m@m, i0),
        )
        #morph.layout(0, 0, 0, 2, 4, 1)
    
        lhs = Cell2( Cell1(i0, m@m), Cell1(i0, m@m),)
        rhs = Cell2( Cell1(m@m, i0), Cell1(m@m, i0),)
        morph = top * (lhs << rhs)
        morph.layout() #0, 0, 0, 2, 4, 1)
        morphisms.append(morph)
    
        saddle = Cell2(
            Cell1(m@m, i0) << Cell1(i0, m@m),
            Cell1(m, m, colour=None) @ Cell1(m, m, colour=None),
        )
        lfold = Cell2(Cell1(i0, m@m), Cell1(i0, m@m))
        rfold = Cell2(Cell1(m@m, i0), Cell1(m@m, i0))
        cup = Cell2(
            Cell1(i0, m@m) <<(Cell1(m, m, colour=None) @ Cell1(m, m, colour=None)) << Cell1(m@m, i0),
            Cell1(i0, i0), 
        )
        cup = Cell2(
            Cell1(i0, m@m) << Cell1(m@m, i0),
            Cell1(i0, i0)
        )
    
        morph = lfold << saddle << rfold
        #morph = morph * cup
        #morph = saddle
        morph.layout() #0, 0, 0, 4, 4, 2)
        morphisms.append(morph)

    #morph = ident(p@p,m@m) << ident(m@m, l)
    #morph.layout(0, 0, 0, 2, 4, 1)


    idx = 0
    theta = -0.2*pi

    dy = 0.6
    dz = 0.6

    while 1:

        if state.did_key("n"):
            idx = (idx+1)%len(morphisms)
        morph = morphisms[idx]

        view = View(state.WIDTH, state.HEIGHT, sort_gitems=False)
        #view.perspective(15)
        view.ortho()

        x0, y0, z0 = morph.center

        R = 3.
        x = 2*sin(theta) + x0
        y = -R
        z = 1*cos(theta) + z0 + morph.top + 0.
        theta += 0.005*pi
        pos = [x, y, z]
        view.lookat(pos, [x0, y0, z0], [0, 0, 1]) # eyepos, lookat, up

        # ---------------------------------------------------------

        morph.render(view)

        if Cell2.DEBUG and 0:
            view.add_line(Mat([-0.5,0,0]), Mat([8,0,0]), 1.0, stroke=(1,0,0,0.5))
            view.add_line(Mat([0,-0.5,0]), Mat([0,8,0]), 1.0, stroke=(0,1,0,0.5))
            view.add_line(Mat([0,0,-0.5]), Mat([0,0,8]), 1.0, stroke=(0,0,1,0.5))

        # just does not work well enough...
        # we have to sort: GCurve, GSurface, GCircle
        #shuffle(view.gitems)
        to_sort = [pov.GSurface, pov.GCurve, pov.GCircle]
        def less_than(lhs, rhs):
            # lhs < rhs means draw lhs before rhs, lhs is *behind* rhs
            depth = view.get_depth(lhs) < view.get_depth(rhs)
            ltp, rtp = type(lhs), type(rhs)
            if ltp == rtp:
                return depth
            elif lhs.incident(rhs, 0.05):
                #print("*", end=" ")
                idx, jdx = to_sort.index(ltp), to_sort.index(rtp)
                return idx < jdx
            return depth

        # bubble sort still does not fix
#        gitems = view.gitems
#        n = len(gitems)
#        for i in range(n-1):
#          for j in range(n-i-1):
#            if less_than(gitems[j+1], gitems[j]):
#                gitems[j], gitems[j+1] = gitems[j+1], gitems[j]

        cvs = Canvas()
        view.render(cvs=cvs, less_than=less_than)
        #for gitem in view.gitems:
        #    print(gitem.__class__.__name__, end=" ")
        #print()

        yield Canvas([Scale(2, 2, 0.5*state.width, 0.5*state.height), cvs])
        sleep(0.1)




if __name__ == "__main__":
    print("\n")
    test()

    print("OK")




