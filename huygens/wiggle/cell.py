#!/usr/bin/env python3

"""
render 2-morphisms in a monoidal bicategory using sheet/string diagrams.

"""

from random import random, randint, shuffle
import operator
from functools import reduce
from math import pi, sin, cos
from time import sleep
import warnings

from huygens.front import color
from huygens.sat import System, Listener, Variable
from huygens.back import Compound, Deco, Path, Transform, Scale
from huygens.front import path, style, canvas, color, Canvas
from huygens.back import Visitor
from huygens.namespace import st_normal, st_thin, st_thick, st_Thick, st_THick, st_THIck
from huygens.argv import argv

if argv.noerr:
    import os
    #fd = os.open("/dev/null", os.O_WRONLY)
    #os.dup2(2, fd)
    os.close(2)


from huygens import pov
from huygens.pov import View, Mat, GSurface, GCurve, GCircle, GCvs


EPSILON = 1e-6
PIP = 0.001

black = (0,0,0,1)
grey = (0,0,0,0.2)

def conv(a, b, alpha=0.5):
    return (1.-alpha)*a + alpha*b


"""

The default coordinate system we use _looks like this:

   z
   ^
   |     y
   |    ^
   |   /
   |  /
   | /
   |/
   +-------------> x

The three directions:
    H : horizontal : operator <<  : x coord : .width property  : .left, .right
    D : depth-wise : operator @   : y coord : .depth property  : .front, .back
    V : vertical   : operator *   : z coord : .height property : .top, .bot

The classes that the user interacts with:
    Cell0, DCell0
    Cell1, DCell1, HCell1
    Cell2, DCell2, HCell2, VCell2

These all have "shadow" classes that actually do the layout & rendering
    _Cell0, _DCell0
    _Cell1, _DCell1, _HCell1
    _Cell2, _DCell2, _HCell2, _VCell2

We need this shadow hierarchy because the user can reuse (alias) object's
when building a term (a compound 2-cell),
but when rendering we need a unique cell object for each occurance of a cell in a 
compound cell. This transition is accomplished by the .deepclone method.
Think of this like how a copiler translates from a high level syntax
to a lower level syntax.

Important atributes:
                   .pip_x  .pip_y  .pip_z
    negative dir:  .left   .front  .bot
    positive dir:  .right  .back   .top

These six attributes: front back left right top bot, define
a box that us used as a scaffolding for placing the pip's.
For rendering all that matters is where the pip's are for
each Cell0, Cell1 and Cell2.

Here is a list of the classes below:
class Atom(object):
class Compound(object):
class Render(Listener):
class _Compound(object):
class Cell0(Atom):
class _Cell0(Cell0, Render):
class DCell0(Compound, Cell0):
class _DCell0(DCell0, _Compound, _Cell0):
class Cell1(Atom):
class _Cell1(Cell1, Render):
class DCell1(Compound, Cell1):
class _DCell1(DCell1, _Compound, _Cell1):
class HCell1(Compound, Cell1):
class _HCell1(HCell1, _Compound, _Cell1):
class Segment(object):
class Surface(object):
class Cell2(Atom):
class _Cell2(Cell2, Render):
class DCell2(Compound, Cell2):
class _DCell2(DCell2, _Compound, _Cell2):
class HCell2(Compound, Cell2):
class _HCell2(HCell2, _Compound, _Cell2):
class VCell2(Compound, Cell2):
class _VCell2(VCell2, _Compound, _Cell2):

Yes, this hierarchy is bonkers, and maybe once the code stabilizes
more it can be cleaned up / renamed.

"""


# -------------------------------------------------------

# These are helper classes for _Cell1._render and _Cell2.render below

class Segment(object):
    "a 3d bezier curve"
    def __init__(self, v0, v1, v2, v3, color=(0,0,0,1)):
        self.vs = (v0, v1, v2, v3)
        self.color = color

    def __eq__(self, other):
        return self.vs == other.vs

    #def incident(self, other):
    #    return self == other or self.reversed == other

    def midpoint(self):
        v0, v1, v2, v3 = self.vs
        v = (1./4) * (v0 + v1 + v2 + v3) # um... just hack it for now
        return v

    @classmethod
    def mk_line(cls, v0, v2, **kw):
        #v0, v2 = Mat(v0), Mat(v2)
        v01 = 0.5*(v0+v2)
        v12 = 0.5*(v0+v2)
        return cls(v0, v01, v12, v2, **kw)

    @property
    def reversed(self):
        v0, v1, v2, v3 = self.vs
        return Segment(v3, v2, v1, v0, self.color)

    def grow(self, epsilon=0.03):
        v0, v1, v2, v3 = self.vs
        v0 = v0 - epsilon*(v1 - v0)
        v3 = v3 - epsilon*(v2 - v3)
        return Segment(v0, v1, v2, v3, self.color)

    def __getitem__(self, idx):
        return self.vs[idx]

    def __len__(self):
        return len(self.vs)


class Surface(object):

    def __init__(self, segments, color=(0,0,0,1), pip_cvs=None):
        self.segments = list(segments)
        self.color = color
        self.pip_cvs = pip_cvs

    def __getitem__(self, idx):
        return self.segments[idx]

    def __len__(self):
        return len(self.segments)

    @property
    def reversed(self):
        segments = [seg.reversed for seg in reversed(self.segments)]
        return Surface(segments, self.color)

    @classmethod
    def mk_triangle(cls, v0, v1, v2, color):
        segments = [
            Segment.mk_line(v0, v1),
            Segment.mk_line(v1, v2),
            Segment.mk_line(v2, v0),
        ]
        return Surface(segments, color=color)

    def midpoint(self):
        vs = [segment.midpoint() for segment in self.segments]
        v = vs[0]
        for u in vs[1:]:
            v = v + u
        v = (1/len(vs))*v
        return v

    def incident(self, other):
        if self.color != other.color:
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
            return Surface(segs, self.color)
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
        view.add_surface(self.segments, fill=self.color)
        if self.pip_cvs is not None:
            view.add_cvs(self.midpoint(), self.pip_cvs)
        if Cell2.DEBUG or 0:
            for seg in self.segments:
                view.add_curve(*seg, stroke=(0,0,1,0.2), epsilon=None)


# -------------------------------------------------------

# These are the abstract base classes for all the cell classes to come

class Atom(object):

    weight = 1.0
    assoc = True

    def __init__(self, name, **kw):
        self.name = name
        #self.__dict__.update(kw)
        for (k,v) in kw.items():
            assert hasattr(self, k), "attribute %r not found on %s" % (
                k, self.__class__.__name__)
            setattr(self, k, v)
        #print("%s: Atom.__init__"%self.__class__.__name__, list(kw.keys()))

    def __str__(self):
        return self.name

    def visit(self, callback, cls=None, is_tgt=True, is_src=True, **kw):
        if cls is None or self.__class__ == cls:
            callback(self, is_tgt=is_tgt, is_src=is_src, **kw)

    def search(self, **kw):
        cells = []
        def cb(cell, **kw):
            cells.append(cell)
        self.visit(cb, **kw)
        return cells

    def index(self, other):
        if self is other:
            return 0
        assert 0, "not found"

    def traverse(self, cb, depth=0, full=True):
        cb(self, depth)

    def __call__(self, **kw):
        self = self.deepclone() # **kw for deepclone ?
        name = self.name
        if "name" in kw:
            name = kw["name"]
            del kw["name"]
        Atom.__init__(self, name, **kw)
        return self

    def all_atoms(self):
        yield self

    def h_rev(self):
        "horizontal reverse"
        return self.deepclone(h_rev=True)
    h_op = h_rev

    def v_rev(self):
        "vertical reverse"
        return self.deepclone(v_rev=True)
    v_op = v_rev

    def d_rev(self):
        "depth-wise reverse"
        return self.deepclone(d_rev=True)
    d_op = d_rev

    def is_flat(self):
        return True

    def _repr_svg_(self):
        item = self.layout()
        flat = item.is_flat()
        pos = "center" if flat else "northeast"
        item = item.render_cvs(weld=flat, pos=pos)
        svg = item._repr_svg_()
        return svg

    def render_cvs(self, pos="center", eyepos=None, lookat=None, up=None, weld=None, ortho=True):
        item = self.layout()
        flat = item.is_flat()
        if pos is None:
            pos = "center" if flat else "northeast"
        weld = flat if weld is None else weld
        cvs = item.render_cvs(pos, eyepos, lookat, up, weld, ortho)
        return cvs


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

    def all_atoms(self):
        for cell in self.cells:
            for atom in cell.all_atoms():
                yield atom

    def index(self, other):
        for idx, cell in enumerate(self.cells):
            if cell is other:
                return idx
        assert 0, "not found"

    def _associate(self, items):
        for item in items:
            if not item.assoc:
                return items
        cls = self.__class__
        itemss = [(item.cells 
            if isinstance(item, cls) # and item.assoc
            else [item])
            for item in items]
        items = reduce(operator.add, itemss, [])
        return items

    def visit(self, callback, is_tgt=True, is_src=True, **kw):
        for child in self.cells:
            child.visit(callback, is_tgt=is_tgt, is_src=is_src, **kw)
        Atom.visit(self, callback, is_tgt=is_tgt, is_src=is_src, **kw)

    def traverse(self, cb, depth=0, full=True):
        Atom.traverse(self, cb, depth, full)
        for cell in self.cells:
            cell.traverse(cb, depth+1, full)

    def str(self, depth=0):
        lines = [Atom.str(self, depth)]
        lines += [cell.str(depth+1) for cell in self.cells]
        return "\n".join(lines)


def setop(cls, opname, parent):
    def meth(left, right):
        return parent([left, right])
    setattr(cls, opname, meth)


def dbg_constrain(self, depth):
    print("  "*depth, "%s.pre_constrain"%(self.__class__.__name__), self.name)



class Render(Listener): # rename as _Render, RenderAtom, or _Atom ?

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

    def pstr(self, depth=0):
        pip_x = getattr(self, "pip_x", None)
        pip_y = getattr(self, "pip_y", None)
        pip_z = getattr(self, "pip_z", None)
        if pip_x is None:
            return "  "*depth + self.__class__.__name__
        #pip_x = "%.2f"%pip_x if pip_x is not None else "--"
        #pip_y = "%.2f"%pip_y if pip_y is not None else "--"
        #pip_z = "%.2f"%pip_z if pip_z is not None else "--"
        x0, x1 = pip_x - self.left, pip_x + self.right
        y0, y1 = pip_y - self.front, pip_y + self.back
        z0, z1 = pip_z - self.bot, pip_z + self.top
        return "%s([%.2f:%.2f],[%.2f:%.2f],[%.2f:%.2f])"%(
            "  "*depth + self.__class__.__name__, 
            x0, x1, y0, y1, z0, z1)

    def dump(self, full=True):
        self.lno = 0
        found = {}
        def callback(cell, depth):
            s = Render.pstr(cell, depth)
            s = "%4d:"%self.lno + s
            if cell in found:
                s += " " + str(found[cell])
            else:
                found[cell] = self.lno
            print(s)
            self.lno += 1
        self.traverse(callback, full=full)

    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self.listen_var(system, "pip_x")
        self.listen_var(system, "pip_y")
        self.listen_var(system, "pip_z")

    def post_constrain(self, system):
        if self.on_constrain is not None:
            self.on_constrain(self, system)

    system = None
    def constrain(self, 
            x=0, y=0, z=0, 
            width=None, depth=None, height=None, 
            size=1.0, verbose=False):
        # top-level call 
        system = System()
        self.pre_constrain(system, 0, verbose=verbose)
        system.add(self.pip_x == x)
        system.add(self.pip_y == y)
        system.add(self.pip_z == z)
        if width is not None and hasattr(self, "left"):
            system.add(self.width == size*width)
        elif width is None and hasattr(self, "w_units"):
            system.add(self.width == 0.7*size*self.w_units)
        if height is not None and hasattr(self, "top"):
            system.add(self.height == size*height)
        elif height is None and hasattr(self, "h_units"):
            system.add(self.height == 0.5*size*self.h_units)
        if depth is not None:
            system.add(self.depth == size*depth)
        elif hasattr(self, "d_units"):
            system.add(self.depth == 0.5*size*self.d_units)
        def constrain(cell, **kw):
            #print("constrain", cell.__class__, cell, cell.on_constrain)
            cell.post_constrain(system)
        self.visit(constrain)
        return system

    did_layout = None
    def layout(self, *args, callback=None, verbose=False, simplify=True, **kw):
        # top-level call 
        #print("Render.layout")
        if self.did_layout:
            #print("already did_layout")
            return self.did_layout
        system = self.constrain(*args, verbose=verbose, **kw)
        if callback is not None:
            callback(self, system)
        system.solve(simplify=simplify)
        self.did_layout = system
        #self.dump(full=False)
        #return system

    def save_dbg(self, name):
        #self.layout()
        cvs = Canvas()
        self.dbg_render(cvs)
        cvs.writePDFfile(name)



class _Compound(object):
    def pre_constrain(self, system, depth, verbose=False):
        #print("%s.pre_constrain(%s)"%(self.__class__.__name__, self.name))
        if verbose:
            dbg_constrain(self, depth)
        #Render.pre_constrain(self, system)
        for cell in self.cells:
            cell.pre_constrain(system, depth+1, verbose)

    def render(self, view):
        #Shape.render(self, view)
        for cell in self.cells:
            cell.render(view)



# -------------------------------------------------------

def check_renderable(cell):
    fail = []
    def callback(cell, depth):
        if not isinstance(cell, Render):
            fail.append(cell)
    cell.traverse(callback)
    return not fail

    
def dump(cell):
    def callback(cell, depth):
        print("  "*depth + cell.__class__.__name__)
    cell.traverse(callback)
    return cell
    

class Cell0(Atom):
    "These are the 0-cells, or object's."

    fill = None
    stroke = black 
    st_stroke = st_normal
    pip_cvs = None 
    skip = False
    on_constrain = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self][idx]

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = dict(self.__dict__)
        kw["assoc"] = self.assoc
        kw["name"] = self.name
        kw["weight"] = self.weight
        kw["fill"] = self.fill
        kw["stroke"] = self.stroke
        kw["st_stroke"] = self.st_stroke
        kw["pip_cvs"] = self.pip_cvs
        kw["skip"] = self.skip
        kw["on_constrain"] = self.on_constrain
        assert self.on_constrain is None, "on_constrain not implemented for Cell0's"
        cell = _Cell0(**kw)
        check_renderable(cell)
        return cell

    def extrude(self, pip_color=None, **kw):
        cell = Cell1(self, self, stroke=None, pip_color=None)
        return cell

    def extrude2(self, **kw):
        # here we stick the pip_cvs on the Cell2
        pip_cvs = self.pip_cvs
        self = self(pip_cvs=None)
        cell = self.extrude() # extrude to Cell1
        cell = cell.extrude(pip_cvs=pip_cvs) # extrude to Cell2
        return cell

    def is_flat(self):
        return True

    def layout(self, *args, **kw):
        cell = self.extrude2()
        cell = cell.layout(*args, **kw)
        return cell


class _Cell0(Cell0, Render):

    def eq_constrain(self, other, system):
        add = system.add
        for attr in "pip_x pip_y pip_z front back".split():
            lhs = getattr(self, attr)
            rhs = getattr(other, attr)
            add(lhs == rhs)

    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.pre_constrain(self, system, depth, verbose)
        back = self.listen_var(system, "back")
        front = self.listen_var(system, "front")

        # put the pip in the middle
        system.add(back == front)




class DCell0(Compound, Cell0):
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        name = "@".join(cell.name for cell in cells) or "ii"
        Cell0.__init__(self, name, **kw)
        self.cells = cells

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = dict(self.__dict__)
        del kw["name"]
        del kw["cells"]
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["assoc"] = self.assoc
        kw["on_constrain"] = self.on_constrain
        kw["skip"] = self.skip
        assert self.on_constrain is None, "on_constrain not implemented for Cell0's"
        cells = list(reversed(self.cells)) if d_rev else self.cells
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _DCell0(cells, **kw)
        check_renderable(cell)
        return cell

    def extrude(self, pip_color=None, **kw):
        #cells = [Cell1(cell, cell, stroke=None, pip_color=None) for cell in self.cells]
        cells = [cell.extrude(**kw) for cell in self.cells]
        cell = DCell1(cells)
        return cell

    def extrude2(self, **kw):
        cells = [cell.extrude2(**kw) for cell in self.cells]
        cell = DCell2(cells)
        return cell

    def is_flat(self):
        if len(self) > 1:
            return False
        return len(self)==0 or self[0].is_flat()


class _DCell0(DCell0, _Compound, _Cell0):
    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell0.pre_constrain(self, system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
        if not len(self):
            return
        add = system.add
        y = self.pip_y - self.front
        w = 1./len(self)
        # Evenly space the pip's along y coord, 
        # _aligned to self pip_x & pip_z
        for cell in self.cells:
            add(self.pip_x == cell.pip_x, self.weight) # _align
            add(self.pip_z == cell.pip_z) # _align
            add(cell.pip_y - cell.front == y, self.weight) # soft equal
            add(cell.depth == w*self.depth, self.weight) # soft equal
            y += cell.depth
        add(self.pip_y + self.back == y) # hard equal



setop(Cell0, "__matmul__", DCell0)

Cell0.ii = DCell0([])



# -------------------------------------------------------


class Cell1(Atom):
    """
        These are the 1-cells.
    """

    stroke = black
    st_stroke = st_Thick
    pip_color = black
    pip_radius = 0.06
    pip_cvs = None
    _width = None # hmmmmm......
    skip = False
    on_constrain = None

    def __init__(self, tgt, src, name=None, **kw):
        assert isinstance(tgt, Cell0)
        assert isinstance(src, Cell0)
        if name is None:
            name = "(%s<---%s)"%(tgt, src)
        assert "color" not in kw, "use stroke"
        Atom.__init__(self, name, **kw) # will update kw's on __dict__
        self.tgt = tgt
        self.src = src
        self.hom = (self.tgt, self.src)

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        tgt, src = self.tgt, self.src
        if h_rev:
            tgt, src = src, tgt
        tgt = tgt.deepclone(h_rev, v_rev, d_rev)
        src = src.deepclone(h_rev, v_rev, d_rev)
        kw = {}
        kw["assoc"] = self.assoc
        kw["weight"] = self.weight
        kw["stroke"] = self.stroke
        kw["st_stroke"] = self.st_stroke
        kw["pip_color"] = self.pip_color
        kw["pip_radius"] = self.pip_radius
        kw["pip_cvs"] = self.pip_cvs
        kw["_width"] = self._width
        kw["skip"] = self.skip
        kw["on_constrain"] = self.on_constrain
        assert self.on_constrain is None, "on_constrain not implemented for Cell0's"
        cell = _Cell1(tgt, src, self.name, **kw)
        check_renderable(cell)
        return cell

    def extrude(self, pip_color=None, rigid=False, **kw):
        def on_constrain(cell, system):
            system.add(cell.pip_x == cell.tgt.pip_x)
            system.add(cell.pip_x == cell.src.pip_x)
            system.add(cell.pip_y == cell.tgt.pip_y)
            system.add(cell.pip_y == cell.src.pip_y)
        if not rigid:
            on_constrain = None
        def on_constrain(cell, system):
            pip_x, pip_y = cell.pip_x, cell.pip_y
            system.add(pip_x == (1/2)*(cell.tgt.pip_x + cell.src.pip_x))
            system.add(pip_y == (1/2)*(cell.tgt.pip_y + cell.src.pip_y))
        cell = Cell2(self, self, pip_color=pip_color, cone=1., on_constrain=on_constrain, **kw)
        return cell

    def reassoc(self):
        yield [self]

    def npaths(self):
        return len(list(self.get_paths()))

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)

    def is_flat(self):
        return self.src.is_flat() and self.tgt.is_flat()

    def layout(self, *args, **kw):
        cell = self.extrude()
        cell = cell.layout(*args, **kw)
        return cell


class _Cell1(Cell1, Render):
    def get_paths(self):
        # a path is a sequence of sub-lists of [tgt, self, src]
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

    def fail_match(tgt, src, message=""):
        print("fail_match:", message)
        print("\t", tgt)
        print("\t", src)
        assert 0
    
    def match(tgt, src):
        assert isinstance(src, Cell1)
        if tgt.name == src.name:
            tgt = tgt.search(cls=_Cell1)
            src = src.search(cls=_Cell1)
            assert len(src) == len(tgt)
            for (t,s) in zip(tgt, src):
                yield (t,s)
            return # <---------- return
        lhs = [[cell for cell in path if not cell.skip] for path in tgt.get_paths()]
        rhs = [[cell for cell in path if not cell.skip] for path in src.get_paths()]
        if len(lhs) != len(rhs):
            tgt.fail_match(src, "%s tgt paths != %s src paths"%(len(lhs), len(rhs)))
    
        send = {}
        found = set() # XXX just use send
        for (left,right) in zip(lhs, rhs):
          if len(left)!=len(right):
              for l in left:
                  print("\tleft:", l)
              for r in right:
                  print("\tright:", r)
              tgt.fail_match(src, 
                  "tgt path len %d != src path len %d"%(len(left), len(right)))
          li = ri = 0
          while li<len(left) and ri<len(right):
            l, r = left[li], right[ri]
            #print("match", l, r)
            if l in send:
                if send[l] != r:
                    tgt.fail_match(src)
            send[l] = r
            if type(l) != type(r): # type is _Cell0 or _Cell1
                tgt.fail_match(src)
            if l.name != r.name:
                tgt.fail_match(src)
            if isinstance(l, Cell1):
                if (l,r) not in found:
                    yield (l,r)
                    found.add((l,r))
            li += 1
            ri += 1

    def eq_constrain(self, other, system):
        add = system.add
        for attr in "pip_x pip_y pip_z front back left right".split():
            lhs = getattr(self, attr)
            rhs = getattr(other, attr)
            add(lhs == rhs)
        self.tgt.eq_constrain(other.tgt, system)
        self.src.eq_constrain(other.src, system)

    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.pre_constrain(self, system, depth, verbose)
        back = self.listen_var(system, "back")
        front = self.listen_var(system, "front")
        left = self.listen_var(system, "left")
        right = self.listen_var(system, "right")

        # put the pip in the middle
        system.add(back == front, self.weight) # soft equal
        system.add(left == right, self.weight) # soft equal

        # keep the pip inside !
        system.add(self.front >= 0)
        system.add(self.back >= 0)
        system.add(self.left >= 0)
        system.add(self.right >= 0)

        if self.__class__ != _Cell1: # bit of a hack
            return # < --------- return

        # now we constrain the _Cell0 src & tgt
        tgt, src = self.tgt, self.src
        tgt.pre_constrain(system, depth+1, verbose)
        src.pre_constrain(system, depth+1, verbose)
        add = system.add
        add(tgt.pip_x == self.pip_x - self.left, self.weight) # tgt to the left
        add(src.pip_x == self.pip_x + self.right, self.weight) # src to the right
        for cell in [tgt, src]:
            add(cell.pip_z == self.pip_z) # hard equal
            add(cell.pip_y - cell.front == self.pip_y - self.front)
            add(cell.pip_y + cell.back == self.pip_y + self.back)
        if self._width is not None:
            add(self.width == self._width)

    def all_src(self):
        for cell in self.src.all_atoms():
            yield cell

    def all_tgt(self):
        for cell in self.tgt.all_atoms():
            yield cell

    def dbg_render(self, bg):
        from huygens import namespace as ns
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
            [color.rgba(1,0,0,0.5)]+ns.st_THick+ns.st_round)
        bg.append(cvs)

    def _render(self, view, parent, is_tgt=True, is_src=True):
        # parent is the Cell2 that self lives inside of
        cone = max(PIP, 1-parent.cone)
        def seg_over(v1, v2):
            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
            line = Segment(v1, v1 + cone*(v12-v1), v2 + cone*(v12-v2), v2)
            return line
        if cone < 0.1:
            # just use a straight line
            seg_over = Segment.mk_line

        pip2 = Mat(parent.pip)
        pip1 = Mat(self.pip)

        line = seg_over(pip1, pip2)
        if self.stroke is not None:
            view.add_curve(*line, stroke=self.stroke, st_stroke=self.st_stroke)

        # show spider (1-cell) pip
        if self.pip_cvs is not None:
            view.add_cvs(Mat(pip1), self.pip_cvs)
        elif self.pip_color is not None:
            view.add_circle(pip1, self.pip_radius, fill=self.pip_color)

        tgt, src = self.tgt, self.src
        for cell in tgt:
            assert isinstance(cell, _Cell0)
            color = cell.fill
            pip0 = Mat(cell.pip)
            vpip1 = Mat([conv(pip0[0], pip1[0]), pip0[1], pip0[2]])
            leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1) # spider leg
            line2 = seg_over(pip0, pip2).reversed
            triangle = Surface([
                line2, # pip2 --> pip0
                leg,   # pip0 --> pip1
                line,  # pip1 --> pip2
            ], color)
            if is_tgt:
                parent.l_ports.append((leg, line, cell))
            elif not cell.skip:
                parent.surfaces.append(triangle)
            if cell.stroke is not None:
                leg = leg.grow() # hack to remove gaps 
                try:
                    view.add_curve(*leg, stroke=cell.stroke, st_stroke=cell.st_stroke)
                except:
                    warnings.warn("_Cell1._render: view.add_curve Exception!")

        for cell in src:
            assert isinstance(cell, _Cell0)
            color = cell.fill
            pip0 = Mat(cell.pip)
            vpip1 = Mat([conv(pip0[0], pip1[0]), pip0[1], pip0[2]])
            line2 = seg_over(pip0, pip2).reversed
            leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1)
            triangle = Surface([
                line2, # pip2 --> pip0
                leg,   # pip0 --> pip1
                line,  # pip1 --> pip2
            ], color)
            if is_src:
                parent.r_ports.append((leg, line, cell))
            elif not cell.skip:
                parent.surfaces.append(triangle)
            if cell.stroke is not None:
                leg = leg.grow() # hack to remove gaps 
                try:
                    view.add_curve(*leg, stroke=cell.stroke, st_stroke=cell.st_stroke)
                except:
                    warnings.warn("_Cell1._render: view.add_curve Exception!")


class DCell1(Compound, Cell1):
    bdy = DCell0
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "@".join(cell.name for cell in cells) + ")"
        Cell1.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["skip"] = self.skip
        kw["assoc"] = self.assoc
        kw["on_constrain"] = self.on_constrain
        assert self.on_constrain is None, "on_constrain not implemented for Cell0's"
        cells = self.cells
        if d_rev:
            cells = list(reversed(cells))
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _DCell1(cells, **kw)
        if not check_renderable(cell):
            dump(cell)
            assert 0, "found non Render's"
        return cell

    def extrude(self, pip_color=None, rigid=False, **kw):
        cells = [cell.extrude(pip_color=pip_color, rigid=rigid, **kw) for cell in self.cells]
        return DCell2(cells, **kw)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)

    def is_flat(self):
        if len(self) > 1:
            return False
        return len(self)==0 or self[0].is_flat()



class _DCell1(DCell1, _Compound, _Cell1):

    def get_paths(self):
        for cell in self.cells:
            for path in cell.get_paths():
                yield path

    bdy = _DCell0
    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell1.pre_constrain(self, system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        y = self.pip_y - self.front
        w = 1./len(self)
        for cell in self.cells:

            #add(self.width == cell.width) # fit width
            #add(cell.pip_x == self.pip_x) # _align pip_x
            # this seems to have the same effect as previous two lines:
            add(self.pip_x - self.left == cell.pip_x - cell.left)
            add(self.pip_x + self.right == cell.pip_x + cell.right)

            add(cell.pip_z == self.pip_z) # _align pip_z
            add(cell.pip_y - cell.front == y)
            add(cell.depth == w*self.depth, self.weight)
            y += cell.depth
        add(self.pip_y + self.back == y) # hard constrain, yes!

    def all_src(self):
        for cell in self.cells:
          for src in cell.all_src():
            yield src

    def all_tgt(self):
        for cell in self.cells:
          for tgt in cell.all_tgt():
            yield tgt

    def dbg_render(self, bg):
        from huygens import namespace as ns
        cvs = Canvas()
        if 0 and hasattr(self, "pip_x"):
            pip_x, pip_y, pip_z = self.pip_x, self.pip_y, self.pip_z
            tx = Transform(
                xx=1.0, yx=0.0,
                xy=0.6, yy=0.5,
                x0=0.0, y0=pip_z)
            cvs.append(tx)
            #cvs.fill(path.circle(pip_x, pip_y, 0.05))
            #for cell in self.tgt:
            #    cvs.stroke(path.line(pip_x, pip_y, cell.pip_x, cell.pip_y))
            #for cell in self.src:
            #    cvs.stroke(path.line(pip_x, pip_y, cell.pip_x, cell.pip_y))
            #print("DCell1.dbg_render")
            cvs.stroke(path.rect(
                pip_x - self.left, pip_y - self.front, self.width, self.depth),
                [color.rgba(0,0,0,1.0)]+ns.st_THIck+ns.st_round)
            bg.append(cvs)
        #else:
        #    assert self.no_constrain == True

        for child in self.cells:
            child.dbg_render(bg)




class HCell1(Compound, Cell1):
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = cells[0].tgt
        src = cells[-1].src
        i = 0
        while i+1 < len(cells):
            if cells[i].src.name != cells[i+1].tgt.name:
                msg = ("can't compose %s and %s"%(cells[i], cells[i+1]))
                raise TypeError(msg)
            i += 1
        name = "(" + "<<".join(cell.name for cell in cells) + ")"
        Cell1.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["skip"] = self.skip
        kw["assoc"] = self.assoc
        kw["on_constrain"] = self.on_constrain
        assert self.on_constrain is None, "on_constrain not implemented for Cell0's"
        cells = list(reversed(self.cells)) if h_rev else self.cells
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _HCell1(cells, **kw)
        check_renderable(cell)
        return cell

    def extrude(self, pip_color=None, rigid=False, **kw):
        cells = [cell.extrude(pip_color=pip_color, rigid=rigid, **kw) for cell in self.cells]
        return HCell2(cells, pip_color=pip_color, **kw)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)

    def is_flat(self):
        for cell in self.cells:
            if not cell.is_flat():
                return False
        return True


class _HCell1(HCell1, _Compound, _Cell1):
    def visit(self, callback, is_tgt=True, is_src=True, **kw):
        n = len(self.cells)
        for i, child in enumerate(self.cells):
            child.visit(callback, 
                is_tgt=(is_tgt and i==0), 
                is_src=(is_src and i==n-1), **kw)
        Atom.visit(self, callback, is_tgt=is_tgt, is_src=is_src, **kw)

    def get_paths(self):
        #print("HCell1.get_paths", self)
        cells = self.cells
        assert len(cells) >= 2, "wup"
        left = cells[0]
        if len(cells) > 2:
            right = _HCell1(cells[1:])
        else:
            right = cells[1]

        #print("_HCell1.get_paths", left, right)
        lpaths = [[] for _ in left.src]
        for lpath in left.get_paths():
            if isinstance(lpath[-1], _Cell1):
                yield lpath
                continue
            cell0 = lpath[-1]
            assert isinstance(cell0, _Cell0)
            assert cell0 in left.src, "%s not in %s"%(cell0.key, left.src.key)
            idx = left.src.index(cell0)
            #print("lpaths: idx = ", idx)
            lpaths[idx].append(lpath)

        for rpath in right.get_paths():
            if isinstance(rpath[0], _Cell1):
                yield rpath
                continue
            cell0 = rpath[0]
            assert isinstance(cell0, _Cell0)
            assert cell0 in right.tgt, "%s not in %s"%(cell0.key, right.tgt.key)
            idx = right.tgt.index(cell0)
            #print("idx = ", idx)
            for lpath in lpaths[idx]:
                yield lpath + rpath

    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell1.pre_constrain(self, system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
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

    def all_src(self):
        assert self.cells
        for src in self.cells[-1].all_src():
            yield src

    def all_tgt(self):
        assert self.cells
        for tgt in self.cells[0].all_tgt():
            yield tgt

    def dbg_render(self, cvs):
        for child in self.cells:
            child.dbg_render(cvs)



setop(Cell1, "__matmul__", DCell1)
setop(Cell1, "__lshift__", HCell1)



# -------------------------------------------------------



class Cell2(Atom):
    "These are the 2-cells"

    DEBUG = False
    pip_color = black
    pip_radius = 0.08
    pip_cvs = None
    cone = 0.6 # closer to 1. is more cone-like
    on_constrain = None

    def __init__(self, tgt, src, name=None, **kw):
        assert isinstance(tgt, Cell1), tgt.__class__.__name__
        assert isinstance(src, Cell1), src.__class__.__name__
        assert tgt.src.name == src.src.name, "%s != %s" % (tgt.src, src.src)
        assert tgt.tgt.name == src.tgt.name, "%s != %s" % (tgt.tgt, tgt.src)
        assert "color" not in kw, "this is called pip_color now"
        assert "show_pip" not in kw, "just set pip_color to None"
        if name is None:
            name = "(%s<===%s)"%(tgt, src)
        Atom.__init__(self, name, **kw) # update's kw's on __dict__
        self.tgt = tgt
        self.src = src
        self.hom = (self.tgt, self.src)

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        kw["assoc"] = self.assoc
        kw["DEBUG"] = self.DEBUG
        kw["weight"] = self.weight
        kw["pip_color"] = self.pip_color
        kw["pip_radius"] = self.pip_radius
        kw["pip_cvs"] = self.pip_cvs
        kw["cone"] = self.cone
        kw["on_constrain"] = self.on_constrain
        src, tgt = self.src, self.tgt
        if v_rev:
            src, tgt = tgt, src
        tgt = tgt.deepclone(h_rev, v_rev, d_rev)
        src = src.deepclone(h_rev, v_rev, d_rev)
        cell = _Cell2(tgt, src, **kw)
        check_renderable(cell)
        return cell

    @property
    def w_units(self): # width units
        return 1

    @property
    def d_units(self): # depth units
        return 1

    @property
    def h_units(self): # height units
        return 1

    def is_flat(self):
        return self.src.is_flat() and self.tgt.is_flat()

    def vflip(self): # XXX pass all attr's along
        tgt, src = self.src, self.tgt
        return Cell2(tgt, src)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)

    did_layout = False
    def layout(self, *args, **kw):
        if self.did_layout:
            return self
        cell = self.deepclone()
        Render.layout(cell, *args, **kw)
        return cell


class _Cell2(Cell2, Render):

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
    def x0(self):
        return self.pip_x - self.left

    @property
    def x1(self):
        return self.pip_x + self.right

    def _pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.pre_constrain(self, system, depth, verbose)
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

    def pre_constrain(self, system, depth, verbose=False):
        self._pre_constrain(system, depth, verbose)
        tgt, src = self.tgt, self.src
        tgt.pre_constrain(system, depth+1, verbose)
        src.pre_constrain(system, depth+1, verbose)
        add = system.add
        add(tgt.pip_z == self.pip_z + self.top) # hard equal
        add(src.pip_z == self.pip_z - self.bot) # hard equal

        for cell in [tgt, src]:
            add(cell.pip_x - cell.left == self.pip_x - self.left, self.weight)
            add(cell.pip_x + cell.right == self.pip_x + self.right, self.weight)
            add(cell.pip_y - cell.front == self.pip_y - self.front, self.weight)
            add(cell.pip_y + cell.back == self.pip_y + self.back, self.weight)
            #add(cell.pip_x == self.pip_x, self.weight)
            #add(cell.pip_y == self.pip_y, self.weight)

        # put the _Cell2 pip at the average x/y coord
        pips = []
        def callback(cell, **kw):
            pips.append((cell.pip_x, cell.pip_y))
        tgt.visit(callback, cls=_Cell1)
        src.visit(callback, cls=_Cell1)
        assert pips
        n = len(pips)
        pip_x = (1/n)*reduce(operator.add, [pip[0] for pip in pips], 0)
        pip_y = (1/n)*reduce(operator.add, [pip[1] for pip in pips], 0)
        add(self.pip_x == pip_x, self.weight)
        add(self.pip_y == pip_y, self.weight)

    def all_chains_src(self):
        a_items = list(self.src.all_src())
        b_items = list(self.tgt.all_src())
        assert len(a_items) == len(b_items)
        return [(a,b) for (a,b) in zip(a_items, b_items)]

    def all_chains_tgt(self):
        a_items = list(self.src.all_tgt())
        b_items = list(self.tgt.all_tgt())
        assert len(a_items) == len(b_items)
        return [(a,b) for (a,b) in zip(a_items, b_items)]

    def render(self, view):

        # We hang a bunch of attr's off self for the visit callback below.
        self.surfaces = surfaces = []
        
        # left and right ports
        self.l_ports = l_tgt = []
        self.r_ports = r_tgt = []
        self.tgt.visit(_Cell1._render, cls=_Cell1, view=view, parent=self)

        # left and right ports
        self.l_ports = l_src = []
        self.r_ports = r_src = []
        self.src.visit(_Cell1._render, cls=_Cell1, view=view, parent=self)

        if len(l_src) != len(l_tgt) or len(r_src) != len(r_tgt):
            warnings.warn("_Cell2.render: FAIL", id(self), end=" ")
            warnings.warn( len(l_src) , len(l_tgt) , end=" ")
            warnings.warn( len(r_src) , len(r_tgt) )
            for surface in surfaces:
                surface.render(view)

        assert len(l_src) == len(l_tgt)
        assert len(r_src) == len(r_tgt)

        for (p_src, p_tgt) in zip(l_src, l_tgt):
            # these are five-sided Surface's
            segs = [p_src[0], p_src[1], p_tgt[1].reversed, p_tgt[0].reversed]
            cell = p_src[2]
            line = Segment.mk_line(segs[-1][3], segs[0][0])
            segs.append(line)
            for i in range(len(segs)):
                l, r = segs[i], segs[(i+1)%len(segs)]
                error = l[3] - r[0]
                assert error.norm() < EPSILON
            surf = Surface(segs, cell.fill, cell.pip_cvs)
            surfaces.append(surf)

        for (p_src, p_tgt) in zip(r_src, r_tgt):
            # these are five-sided Surface's
            segs = [p_src[0], p_src[1], p_tgt[1].reversed, p_tgt[0].reversed]
            cell = p_src[2]
            line = Segment.mk_line(segs[-1][3], segs[0][0])
            segs.append(line)
            for i in range(len(segs)):
                l, r = segs[i], segs[(i+1)%len(segs)]
                error = l[3] - r[0]
                assert error.norm() < EPSILON
            surf = Surface(segs, cell.fill, cell.pip_cvs)
            surfaces.append(surf)

        #surfaces = Surface.merge(surfaces) # does not work very well...
        for surface in surfaces:
            try:
                surface.render(view)
            except:
                warnings.warn("_Cell2.render: surface.render Exception")
        for surface in surfaces:
            if surface.pip_cvs is None:
                continue
            #view.add_cvs(surface.midpoint(), surface.pip_cvs)

        if self.pip_cvs is not None:
            view.add_cvs(Mat(self.pip), self.pip_cvs)
        elif self.pip_color is not None:
            view.add_circle(Mat(self.pip), self.pip_radius, fill=self.pip_color)

#    # XXX can't we do this by setting attr's and calling .render() ?
#    def render_boundary(self, view, src=True, tgt=True):
#        (x0, y0, z0, x1, y1, z1) = self.rect
#        x01 = conv(x0, x1)
#        y01 = conv(y0, y1)
#        z01 = conv(z0, z1)
#
#        pip2 = Mat(self.pip)
#        cone = 1. - self.cone
#        cone = max(PIP, cone)
#
#        def seg_over(v1, v2):
#            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
#            line = Segment(v1, v1 + cone*(v12-v1), v2 + cone*(v12-v2), v2)
#            return line
#
#        def callback(self):
#            assert self.__class__ == _Cell1
#            color = self.color
#            pip1 = Mat(self.pip)
#
#            line = seg_over(pip1, pip2)
#            if color is not None:
#                #view.add_curve(*line, stroke=color)
#                # show spider (1-cell) pip
#                if self.pip_color:
#                    view.add_circle(pip1, self.pip_radius, fill=color)
#
#            tgt, src = self.tgt, self.src
#            for cell in tgt:
#                v = Mat(cell.pip)
#                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
#                leg = Segment(v, conv(v, vpip1), vpip1, pip1) # spider leg
#                view.add_curve(*leg, stroke=cell.stroke)
#
#            for cell in src:
#                v = Mat(cell.pip)
#                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
#                leg = Segment(v, conv(v, vpip1), vpip1, pip1)
#                view.add_curve(*leg, stroke=cell.stroke)
#
#        if tgt:
#            z = z1
#            self.tgt.visit(callback, cls=_Cell1)
#
#        if src:
#            z = z0
#            self.src.visit(callback, cls=_Cell1)
#
#    def render_src(self, view):
#        self.render_boundary(view, src=True, tgt=False)
#
#    def render_tgt(self, view):
#        self.render_boundary(view, src=False, tgt=True)

    def dbg_render(self, cvs):
        self.tgt.dbg_render(cvs)
        self.src.dbg_render(cvs)

    def get_view(self,  pos="center", eyepos=None, lookat=None, up=None):
        #                .pip_x  .pip_y  .pip_z
        # negative dir:  .left   .front  .bot
        # positive dir:  .right  .back   .top
        view = View(200, 200)
        x1, y1, z1 = self.center
        if pos[0].isupper():
            R = 5.
        else:
            R = 2.
        x0, x2 = x1-0.8*R*self.left, x1+0.8*R*self.right
        z0, z2 = z1-R*self.bot, z1+R*self.top
        R = 3.
        x, y, z = 0., -R, 0.
        pos = pos.lower()
        if pos == "center":
            x, z = 0., 0.
        elif pos == "north":
            z = z2
        elif pos == "northeast":
            x = x2
            z = z2
        elif pos == "northwest":
            z = z2
            x = x0
        elif pos == "south":
            z = z0
        elif pos == "southeast":
            x = x2
            z = z0
        elif pos == "southwest":
            z = z0
            x = x0
        elif pos == "east":
            x = x2
        elif pos == "west":
            x = x0
        else:
            assert 0, "pos %r not understood"%(pos,)
        eyepos = [x, y, z] if eyepos is None else eyepos
        lookat = [x1, y1, z1] if lookat is None else lookat
        up = [0, 0, 1] if up is None else up
        #eyepos, lookat, up = Mat(eyepos), Mat(lookat), Mat(up)
        view.perspective()
        view.lookat(eyepos, lookat, up)
        return view

    def get_view_ortho(self,  pos="center", eyepos=None, lookat=None, up=None):
        view = View(100, 100)
        view.ortho()
        pos = pos.lower()
        if pos=="center":
            view.rotate(-90, 1, 0, 0)
        elif pos=="east":
            view.rotate(-90, 1, 0, 0)
            view.rotate(-10, 0, 0, 1)
        elif pos=="west":
            view.rotate(-90, 1, 0, 0)
            view.rotate(+10, 0, 0, 1)
        elif pos=="north":
            view.rotate(-70, 1, 0, 0)
        elif pos=="northeast":
            view.rotate(-70, 1, 0, 0)
            view.rotate(-10, 0, 0, 1)
        elif pos=="northwest":
            view.rotate(-70, 1, 0, 0)
            view.rotate(+10, 0, 0, 1)
        elif pos=="south":
            view.rotate(-100, 1, 0, 0)
        elif pos=="southeast":
            view.rotate(-100, 1, 0, 0)
            view.rotate(-10, 0, 0, 1)
        elif pos=="southwest":
            view.rotate(-100, 1, 0, 0)
            view.rotate(+10, 0, 0, 1)
        else:
            assert 0, "what is this: %r"%(pos,)
        return view

    def render_cvs(self, pos="center", eyepos=None, lookat=None, up=None, weld=False, ortho=True):

        if ortho:
            view = self.get_view_ortho(pos, eyepos, lookat, up)
        else:
            view = self.get_view(pos, eyepos, lookat, up)

        self.render(view)
        #print("proj:")
        #print(view.proj)
        #print("model:")
        #print(view.model)

        if weld:
            # this only works for simple diagrams...
            view.weld_surfaces()

        # sort gitem's back to front for rendering onto Canvas
        make_poset(view)

        #shuffle(view.gitems)
        cvs = Canvas()
        #view.render(cvs=cvs, less_than=lambda lhs,rhs:self.view_less_than(view,lhs,rhs))
        view.render(cvs=cvs)
        return cvs

    # just does not work well enough...
    # we have to sort: GCurve, GSurface, GCircle, GCvs
    @staticmethod
    def view_less_than(view, lhs, rhs):
        to_sort = [pov.GSurface, pov.GCurve, pov.GCircle, pov.GCvs]
        # lhs < rhs means draw lhs before rhs, lhs is *behind* rhs
        deeper = view.get_depth(lhs) < view.get_depth(rhs)
        ltp, rtp = type(lhs), type(rhs)
        if ltp == rtp:
            return deeper
        lname, rname = (ltp.__name__, rtp.__name__)
        incident = lhs.incident(rhs, 0.05)
        debug = "GCircle" in (lname, rname) and 'GCurve' in (lname,rname)
        if debug:
            print(lname, getattr(lhs, "fill", ""), getattr(lhs, "stroke", ""))
            print(rname, getattr(rhs, "fill", ""), getattr(rhs, "stroke", ""))
            print('\t', incident)
            print()
        if incident:
            idx, jdx = to_sort.index(ltp), to_sort.index(rtp)
            return idx < jdx
        #else:
        #    print("/", end="\n")
        return deeper


class Poset(object):
    def __init__(self, items=[]):
        import string
        letters = list(string.ascii_letters)
        i = 0
        while len(letters) < len(items):
            letters += ["%s_%d"%(c, i) for c in string.ascii_letters]
            i += 1
        self.names = dict((a,b) for (a,b) in zip(items, letters)) # only used for debug
        self.items = set(items)
        self.down = dict((a, []) for a in items)
        self.up = dict((a, []) for a in items)
        self.pairs = set()

    def compare(self, a, b):
        down = self.down
        bdy = [a]
        found = set([a])
        while bdy:
            _bdy = set()
            for c in bdy:
                els = down[c]
                for d in els:
                    if b is d:
                        return True
                    elif d not in found:
                        _bdy.add(d)
                        found.add(d)
            bdy = _bdy
        return False

    def add(self, a, b):
        assert a in self.items
        assert b in self.items
        assert a is not b
        if (a,b) in self.pairs:
            return
        assert not self.compare(b, a)
        self.pairs.add((a, b))
        self.down[a].append(b)
        self.up[b].append(a)
        assert self.compare(a, b)

    def todot(self, name):
        items = list(self.items)
        down = self.down
        pairs = []
        for (a, b) in self.pairs:
            for c in down[a]:
                if b in down[c]:
                    break
            else:
                pairs.append((a,b))
        names = self.names
        f = open(name, 'w')
        print("digraph\n{", file=f)
        for (a,b) in pairs:
            print('  "%s" -> "%s"' % (names[a], names[b]), file=f)
        print("}", file=f)
        f.close()

    def get_linear(self):
        linear = []
        items = self.items
        pairs = self.pairs
        remain = set(items)
        found = set()
        while remain:
            size = len(remain)
            ready = set(remain)
            for (a,b) in pairs:
                # a before b
                if a not in found and b in ready:
                    ready.remove(b)
            for a in ready:
                remain.remove(a)
                linear.append(a)
                found.add(a)
            assert len(remain)<size, "loop found in Poset"
        return linear

    def get_deps(self, item):
        up = self.up
        deps = set([item])
        bdy = set(deps) # unsatisfied
        while bdy:
            _bdy = set()
            for item in bdy:
                for dep in up[item]:
                    if dep not in deps:
                        _bdy.add(dep)
                        deps.add(dep)
            bdy = _bdy
        return deps

#    def get_greedy(self):
#        # add some more relations to force rendering of later
#        # GItem's if possible: pip's and curves.
#        items = self.items
#        down = self.down
#        bots = [a for a in items if not down[a]]
#        assert bots
#        names = self.names
#        #print("get_greedy")
#        rank = {}
#        for bot in bots:
#            deps = self.get_deps(bot)
#            rank[bot] = deps
#        bots.sort( key = lambda item : len(rank[item]) )
#        prev = bots[0]
#        found = set(rank[prev])
#        for bot in bots[1:]:
#            for other in rank[bot]:
#                if other not in found:
#                    self.add(prev, other)
#                    found.add(other)
#            prev = bot
    

INCIDENT = 0.1 

def make_poset(view):
    "sort items back to front for rendering into 2d Canvas"

    gitems = view.gitems
    #print("make_poset: gitems", len(gitems))
    rank = [GSurface, GCurve, GCircle, GCvs]
    #gitems.sort(key = lambda item : rank.index(item.__class__))
    poset = Poset(gitems)
    ranked = dict((r,[]) for r in rank)
    for item in gitems:
        ranked[item.__class__].append(item)

    # Argh, this is fragile code... 

    for l in ranked[GSurface]:
      for r in ranked[GSurface]:
        if id(l)>=id(r):
            continue
        verts = l.incident(r, INCIDENT)
        if len(verts)<=1:
            continue
        deeper = view.get_depth(l) < view.get_depth(r)
        if deeper:
            poset.add(l, r)
        else:
            poset.add(r, l)

      for r in ranked[GCurve]:
        verts = l.incident(r, INCIDENT)
        if not verts:
            continue
        if len(verts)>1:
            poset.add(l, r) # l before r
            continue

      for r in ranked[GCircle] + ranked[GCvs]:
        verts = l.incident(r, INCIDENT)
        if verts:
            poset.add(l, r) # l before r

    for l in ranked[GCurve]:
      for r in ranked[GCircle] + ranked[GCvs]:
        verts = l.incident(r, INCIDENT)
        if verts:
            poset.add(l, r) # l before r
    for l in ranked[GSurface]:
      v0 = l.center
      for r in ranked[GCvs]:
        dist = (r.center - l.center).norm()
        if dist < INCIDENT: # this is not exact because of camera view..?
            poset.add(l, r)
    for l in ranked[GSurface]:
      for r in ranked[GSurface]:
        if id(l)>=id(r):
            continue
        if poset.compare(l, r) or poset.compare(r, l):
            continue
        deeper = view.get_depth(l) < view.get_depth(r)
        if deeper:
            poset.add(l, r)
        else:
            poset.add(r, l)

    for l in ranked[GSurface]:
      for r in ranked[GCurve]:
        if poset.compare(l, r) or poset.compare(r, l):
            continue
        deeper = view.get_depth(l) < view.get_depth(r)
        if deeper:
            poset.add(l, r)
        else:
            poset.add(r, l)

    gitems = poset.get_linear()

    # remove duplicate GCurve's
    i = 0
    while i < len(gitems):
        if not isinstance(gitems[i], GCurve):
            i += 1
            continue # <---- continue
        j = i+1
        while j < len(gitems):
            if isinstance(gitems[j], GCurve) and gitems[i].eq(gitems[j]):
                gitems.pop(j)
                #print("pop", j)
            else:
                j += 1
        i += 1

    view.gitems[:] = gitems




class DCell2(Compound, Cell2):
    bdy = DCell1
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "@".join(cell.name for cell in cells) + ")"
        Cell2.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        kw["assoc"] = self.assoc
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["on_constrain"] = self.on_constrain
        cells = list(reversed(self.cells)) if d_rev else self.cells
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _DCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def w_units(self):
        return max(cell.w_units for cell in self.cells)

    @property
    def d_units(self):
        return sum(cell.d_units for cell in self.cells)

    def is_flat(self):
        if len(self) > 1:
            return False
        return len(self)==0 or self[0].is_flat()

    @property
    def h_units(self):
        return max(cell.h_units for cell in self.cells)

    def vflip(self):
        cells = [cell.vflip() for cell in self.cells]
        return DCell2(cells)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)


class _DCell2(DCell2, _Compound, _Cell2):
    bdy = _DCell1
    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._pre_constrain(system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        y = self.pip_y - self.front
        w = 1./self.d_units
        for cell in self.cells:
            # we don't care if the pip's are aligned, only the left/right bdy
            #add(self.width == cell.width) # fit width
            #add(cell.pip_x == self.pip_x) # _align pip_x
            add(cell.pip_x - cell.left == self.pip_x - self.left)
            add(cell.pip_x + cell.right == self.pip_x + self.right)

            # we don't care if the pip's are aligned, only the top/bot bdy
            #add(self.height == cell.height) # fit width
            #add(cell.pip_z == self.pip_z) # _align pip_z
            add(cell.pip_z - cell.bot == self.pip_z - self.bot)
            add(cell.pip_z + cell.top == self.pip_z + self.top)

            add(cell.pip_y - cell.front == y)
            add(cell.depth == w*cell.d_units*self.depth, self.weight) # soft
            y += cell.depth
        #add(self.pip_y + self.back == y, self.weight)
        add(self.pip_y + self.back == y) # don't be a softy!

    def all_chains_src(self):
        chains = []
        for cell in self.cells:
            chains += cell.all_chains_src()
        return chains

    def all_chains_tgt(self):
        chains = []
        for cell in self.cells:
            chains += cell.all_chains_tgt()
        return chains




class HCell2(Compound, Cell2):
    debug = False
    bdy = HCell1
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "<<".join(cell.name for cell in cells) + ")"
        Cell2.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        kw["assoc"] = self.assoc
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["on_constrain"] = self.on_constrain
        cells = list(reversed(self.cells)) if h_rev else self.cells
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _HCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def w_units(self):
        return sum(cell.w_units for cell in self.cells)

    @property
    def d_units(self):
        return max(cell.d_units for cell in self.cells)

    @property
    def h_units(self):
        return max(cell.h_units for cell in self.cells)

    def is_flat(self):
        for cell in self.cells:
            if not cell.is_flat():
                return False
        return True

    def vflip(self):
        cells = [cell.vflip() for cell in self.cells]
        return HCell2(cells)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)


class _HCell2(HCell2, _Compound, _Cell2):
    bdy = _HCell1
    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._pre_constrain(system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        cells = self.cells
        assert cells, "??"
        x = self.pip_x - self.left # start here
        w = 1./self.w_units
        for cell in self.cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            #add(cell.pip_y == self.pip_y)

            add(cell.pip_z+cell.top == self.pip_z+self.top) 
            add(cell.pip_z-cell.bot == self.pip_z-self.bot) 
            #add(cell.pip_z == self.pip_z)
            #add(self.height == cell.height) # fit height

            add(cell.pip_x == cell.left + x)
            add(cell.width == w*cell.w_units*self.width, self.weight) # soft equal
            x += cell.width
        add(self.pip_x + self.right == x) # hard equal

        # Now weld the Cell0 pip's together!
        i = 0
        while i+1 < len(self):
            lhs, rhs = self.cells[i:i+2]
            lhs, rhs = self.cells[i:i+2]
            lchains = lhs.all_chains_src()
            rchains = rhs.all_chains_tgt()
            assert len(lchains) == len(rchains)
            for lchain, rchain in zip(lchains, rchains):
                for (l,r) in zip(lchain, rchain):
                    if self.debug:
                        l.stroke = color.rgb(0.7,0,0,0.5) # for debug
                        r.stroke = color.rgb(0,0.5,0.5,0.5) # for debug
                    add(l.pip_x == r.pip_x) # hard eq
                    add(l.pip_y == r.pip_y) # hard eq
                    add(l.pip_z == r.pip_z) # hard eq
            i += 1

    def all_chains_src(self):
        return self.cells[-1].all_chains_src()

    def all_chains_tgt(self):
        return self.cells[0].all_chains_tgt()


class VCell2(Compound, Cell2):
    def __init__(self, cells, **kw):
        cells = self._associate(cells) # cell's go top down
        tgt = cells[0].tgt
        src = cells[-1].src
        name = "(" + "*".join(cell.name for cell in cells) + ")"
        Cell2.__init__(self, tgt, src, name, **kw)
        self.cells = cells
        i = 0
        while i+1 < len(cells):
            l, r = cells[i].src, cells[i+1].tgt
            #list(l.match(r))
            #if cells[i].src.name != cells[i+1].tgt.name:
            #    msg = ("can't compose\n%s and\n%s"%(cells[i], cells[i+1]))
            #    raise TypeError(msg)
            #    #print("VCell2.__init__: WARNING", msg)
            i += 1

    def deepclone(self, h_rev=False, v_rev=False, d_rev=False):
        kw = {}
        kw["assoc"] = self.assoc
        #kw["color"] = self.color
        #kw["stroke"] = self.stroke
        kw["on_constrain"] = self.on_constrain
        cells = list(reversed(self.cells)) if v_rev else self.cells
        cells = [cell.deepclone(h_rev, v_rev, d_rev) for cell in cells]
        cell = _VCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def w_units(self):
        return max(cell.w_units for cell in self.cells)

    @property
    def d_units(self):
        return max(cell.d_units for cell in self.cells)

    @property
    def h_units(self):
        return sum(cell.h_units for cell in self.cells)

    def is_flat(self):
        for cell in self.cells:
            if not cell.is_flat():
                return False
        return True

    def vflip(self):
        cells = [cell.vflip() for cell in reversed(self.cells)]
        return VCell2(cells)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)


class _VCell2(VCell2, _Compound, _Cell2):
    def pre_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._pre_constrain(system, depth, verbose)
        _Compound.pre_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        z = self.pip_z - self.bot
        cells = self.cells # cells go top down
        cells = list(reversed(cells)) # now it's bottom up
        w = 1./self.h_units
        for cell in cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            add(cell.pip_y == self.pip_y)
            add(cell.pip_z == z + cell.bot)
            add(cell.pip_x - cell.left == self.pip_x - self.left)
            add(cell.pip_x + cell.right == self.pip_x + self.right)
            add(cell.height == w*cell.h_units*self.height, self.weight)
            z += cell.height
        add(self.pip_z + self.top == z) # hard equal

        i = 0
        while i+1 < len(cells):
            src, tgt = cells[i:i+2] # Cell2's
            src, tgt = src.tgt, tgt.src # Cell1's
            for (t, s) in tgt.match(src):
                # t, s are Cell1's
                s.eq_constrain(t, system)
            i += 1

    def all_chains_src(self):
        chainss = [cell.all_chains_src() for cell in self.cells]
        #print("chainss:", chainss)
        n = len(chainss[0])
        for chains in chainss:
            assert len(chains) == n
        chains = []
        for _chains in zip(*chainss):
            #print('\t', _chains)
            chain = reduce(operator.add, _chains, ())
            chains.append(chain)
        for chain in chains:
            for cell in chain:
                assert isinstance(cell, Cell0)
        return chains

    def all_chains_tgt(self):
        chainss = [cell.all_chains_tgt() for cell in self.cells]
        n = len(chainss[0])
        for chains in chainss:
            assert len(chains) == n
        chains = []
        for _chains in zip(*chainss):
            chains.append( reduce(operator.add, _chains, ()) )
        for chain in chains:
            for cell in chain:
                assert isinstance(cell, Cell0)
        return chains

    def render(self, view):
        #Shape.render(self, view)
        n = len(self.cells)
        for i, cell in enumerate(self.cells):
            tgt = (i==0)
            src = True
            cell.render(view)


setop(Cell2, "__matmul__", DCell2)
setop(Cell2, "__lshift__", HCell2)
setop(Cell2, "__mul__", VCell2)

# -------------------------------------------------------



class BraidDeco(Deco):
    def __init__(self, alpha):
        Deco.__init__(self)
        self.alpha = alpha

    def on_decorate(self, pre, path, post):
        curve_to = path[1]
        x = conv(curve_to.x1, curve_to.x2, self.alpha)
        y = conv(curve_to.y1, curve_to.y2, self.alpha)
        curve_to.x2 = x
        curve_to.y2 = y

st_braid = [BraidDeco(0.8)]


# -------------------------------------------------------



def show_uniq(cell):
    found = set()
    def dump(cell, depth):
        x = id(cell)
        print("  "*depth + cell.__class__.__name__, "*" if x in found else "")
        found.add(x)
    cell.traverse(dump)


def test():
    l = Cell0("l")
    m = Cell0("m")
    n = Cell0("n")
    o = Cell0("o")
    p = Cell0("p")

    l = l.deepclone()

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

    assert str(mm.deepclone()) == "m@m"

    #cell = Cell2(B, AA) * Cell2(AA, A)
    #cell = cell @ (f*f*f)
    #cell = Cell2(B, A<<B)

    cell = Cell1(mm,mm) << Cell1(mm,mm)
    cell = Cell1(mm,mm)<<((Cell1(m,mmm)) @ Cell1(m,m)) << Cell1(mmmm,m)

    c = cell.deepclone()
    assert str(cell) == str(c)


    cell = Cell2(cell, Cell1(mm,m))


    mm_ = Cell1(mm, ii)
    m_m = Cell1(m, m)
    mm_m = Cell1(mm, m)
    _mm = Cell1(ii, mm)
    _m = Cell1(ii, m)
    cell = Cell2(_mm, _mm) << Cell2(mm_, mm_)
    cell = Cell2(m_m, (m_m @ _m) << mm_m)

    c = cell.deepclone()
    #show_uniq(cell)
    #show_uniq(c)

    A, B = Cell1(m@n, l), Cell1(l, p)
    A1, B1 = Cell1(m@n, o@o), Cell1(o@o, p)
    A2, B2 = Cell1(m@n, p@l), Cell1(p@l, p)

    AB = A << B
    str(A@A)
    str(AB)

    f = Cell2(A, A)
    g = Cell2(B, B)
    str(f)
    str(f@f)
    str(f << g)

    f = Cell2(A<<B, A1<<B1)
    g = Cell2(A1<<B1, A2<<B2)
    str(f*g)


def test_match():

    ii = Cell0.ii
    n = Cell0("n", fill=grey, stroke=(0,0,0,1))
    k = Cell0("k", fill=grey, stroke=(0,0,0,1))
    _n = Cell1(ii, n)
    n_ = Cell1(n, ii)
    nn_n = Cell1(n@n, n)
    n_nn = Cell1(n, n@n)
    n_n = Cell1(n, n, pip_color=None, stroke=None)
    pair = Cell1(n, n, pip_color=None, stroke=black)
    swap = Cell1(n@n, n@n, pip_color=None)

# broken
#    assert n_nn.npaths() == 2
#    assert (n_nn @ nn_n).npaths() == 4
#    count = (n_nn << nn_n).npaths()
#    cell = n_nn << nn_n
#    print(cell)
#    assert count == 2, count
#    assert (n_nn << (n_n @ n_nn)).npaths() == 3

    # we use II to get match to work
    I = Cell2(n_n, n_n, pip_color=None)
    II = I<<I
    assoc = Cell2(
        n_nn<<(n_n @ n_nn),
        n_nn<<(n_nn @ n_n))
    mid = assoc << (n_n @ n_nn @ n_n).extrude()
    top = n_nn.extrude() << (II @ assoc)
    #cell = top * mid
    bot = n_nn.extrude() << (assoc @ II)
    #cell = mid * bot
    #cell = mid * mid.src.extrude()

    #print(cell)


def more_test():

    scheme = "ff5e5b-d8d8d8-ffffea-00cecb-ffed66"
    scheme = scheme.split("-")
    scheme = [color.rgbhex(rgb).alpha(0.5) for rgb in scheme]

    names = 'lmnop'
    l, m, n, o, p = [
        Cell0(name, fill=scheme[i%len(scheme)])
        for i, name in enumerate('lmnop')]
    i0 = Cell0("i", fill=None)

    I_l = Cell1(l, l, pip_color=None, stroke=None)
    I_m = Cell1(m, m, pip_color=None, stroke=None)
    I_n = Cell1(n, n, pip_color=None, stroke=None)
    I_o = Cell1(o, o, pip_color=None, stroke=None)

    cell = Cell1(m, m@m) << Cell1(m@m, n)
    cell = cell @ cell

    # i don't think we can handle bubbles... 
    ii = Cell0.ii
    bubble = Cell1(ii, m@m) << Cell1(m@m, ii)

    cell = bubble

    cell = bubble<<bubble

    cell = bubble@bubble

    cell = Cell1(m, m) @ bubble


    l_mn = Cell1(l, m@n)
    mn_l = Cell1(m@n, l)
    o_mn = Cell1(o, m@n)
    mn_o = Cell1(m@n, o)
    l_l = Cell1(l, l)
    o_o = Cell1(o, o)
    left = (o_mn << mn_o << o_o) @ (l_mn << mn_l << l_l)
    right = (o_mn @ l_mn) << (mn_o @ mn_l) << (o_o @ l_l)

    top = left.extrude()
    bot = right.extrude()
    cell = top * bot
    #cell = top
    cvs = cell.layout().render_cvs(pos="north")
    #cvs.writePDFfile("test.pdf")
    

    o_mn = Cell1(o, m@n)
    cell = Cell2(I_o<<o_mn, o_mn<<(I_m@I_n), cone=1.)

    swap = lambda a,b : Cell1(a@b, b@a)
    # Yang-Baxter
    def yang_baxter(n, m, l, reversed=False, **kw):
        I_l = Cell1(l, l, pip_color=None, stroke=None)
        I_m = Cell1(m, m, pip_color=None, stroke=None)
        I_n = Cell1(n, n, pip_color=None, stroke=None)
        tgt = (I_n @ swap(m, l)) << (swap(n, l) @ I_m) << (I_l @ swap(n, m))
        src = (swap(n, m) @ I_l) << (I_m @ swap(n, l)) << (swap(m, l) @ I_n)
        if reversed:
            tgt, src = src, tgt
        morph = Cell2(tgt, src, cone=1.0, pip_color=None, **kw)
        return morph
    
    # a part of the Zamolodchikov Tetrahedron Equation
    lhs = (I_o @ ((I_n@swap(m,l))<<(swap(n,l)@I_m)) ).extrude(pip_color=None)
    rhs = (I_l @ ((swap(o,m)@I_n)<<(I_m@swap(o,n)) )).extrude(pip_color=None)
    
    tgt = swap(n,m) << (I_m @ I_n)
    src = (I_n @ I_m) << swap(n,m)
    back = Cell2(tgt, src, pip_color=None, cone=1.0)
    tgt = (I_o @ I_l) << swap(o,l)
    src = swap(o,l) << (I_l @ I_o)
    front = Cell2(tgt, src, pip_color=None, cone=1.0)
    
    morph_0 = lhs << (front @ back) << rhs
    
    rhs = I_l.extrude(pip_color=None) @ yang_baxter(o, n, m, reversed=True)
    lhs = (I_o @ I_n @ swap(m,l)) << (I_o @ swap(n,l) @ I_m) << (swap(o,l) @ I_n @ I_m)
    lhs = lhs.extrude()
    morph_1 = lhs << rhs
    morph_1 = morph_1.vflip()
    
    cell = morph_1
    cvs = cell.layout().render_cvs(pos="north")
    #cvs.writePDFfile("test.pdf")
    
            
def test_render():

    from huygens import config
    config(text="pdflatex", latex_header=r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{extarrows}
    """)
    from huygens.namespace import st_center, st_dashed, st_dotted, st_thin

    scheme = "ff5e5b-d8d8d8-ffffea-00cecb-ffed66"
    scheme = scheme.split("-")
    scheme = [color.rgbhex(rgb).alpha(0.5) for rgb in scheme]

    names = 'lmnop'
    l, m, n, o, p = [
        Cell0(name, fill=scheme[i%len(scheme)])
        for i, name in enumerate('lmnop')]

    cl = (1,1,0,1)
    shade = color.rgb(0.8)
    ll_l = Cell1(l@l, l, pip_color=cl, stroke=shade)
    l_ll = Cell1(l, l@l, pip_color=cl, stroke=shade)
    l_lll = Cell1(l, l@l@l, pip_color=cl, stroke=shade)
    l_l = Cell1(l, l, pip_color=cl, stroke=shade)
    ll_ll = Cell1(l@l, l@l, pip_color=cl, stroke=shade)
    tgt = l_lll << (l_l @ ll_l) << ll_ll
    cell = Cell2(tgt, l_ll, pip_color=color.rgb(0.5))
    #cell = cell << ll_l.extrude()
    #cell = l_l.extrude() @ cell
    cell = cell @ l_l.extrude()

    f = cell.layout(width=1.5, height=1, depth=2.)
    cvs = f.render_cvs(pos="northwest")
    cvs.writePDFfile("test_render.pdf")

    return




if __name__ == "__main__":
    print("\n")
    #test()
    #test_match()
    #more_test()
    test_render()

    print("OK\n")

