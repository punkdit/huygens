#!/usr/bin/env python3

"""

render 2-morphisms in a bicategory using sheet/string diagrams.

WARNING: this code is bonkers, but does work sometimes.



"""

from random import random, randint, shuffle
import operator
from functools import reduce
from math import pi, sin, cos
from time import sleep

from huygens.front import color
from huygens.sat import System, Listener, Variable
from huygens.back import Compound, Deco, Path, Transform, Scale
from huygens.front import path, style, canvas, color, Canvas
from huygens.back import Visitor
from huygens.argv import argv

from huygens import pov
from huygens.pov import View, Mat


EPSILON = 1e-6
PIP = 0.001

black = (0,0,0,1)
grey = (0,0,0,0.2)

def conv(a, b, alpha=0.5):
    return (1.-alpha)*a + alpha*b


"""

The three directions:
    H : horizontal : operator <<  : x coord : .width property  : .left, .right
    D : depth      : operator @   : y coord : .depth property  : .front, .back
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
but when rendering we need a uniqe cell object for each occurance of a cell in a 
compound cell. This transition is accomplished by the .deepclone method.

Important atributes:
                   .pip_x  .pip_y  .pip_z
    negative dir:  .left   .front  .bot
    positive dir:  .right  .back   .top

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

"""


class Atom(object):
    def __init__(self, name, weight=1., **kw):
        self.name = name
        self.weight = weight
        self.__dict__.update(kw)

    def __str__(self):
        return self.name

    def visit(self, callback, instance=None, **kw):
        if instance is None or self.__class__ == instance:
            callback(self)

    def search(self, **kw):
        cells = []
        def cb(cell):
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
        self.__dict__.update(kw)
        return self


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
        itemss = [(item.cells 
            if isinstance(item, cls) and item.assoc
            else [item])
            for item in items]
        items = reduce(operator.add, itemss, [])
        return items

    def visit(self, callback, **kw):
        for child in self.cells:
            child.visit(callback, **kw)
        Atom.visit(self, callback, **kw)

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
    print("  "*depth, "%s.on_constrain"%(self.__class__.__name__))



class Render(Listener): # rename as _Render ?
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

    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self.listen_var(system, "pip_x")
        self.listen_var(system, "pip_y")
        self.listen_var(system, "pip_z")

    system = None
    def constrain(self, 
            x=0, y=0, z=0, 
            width=None, depth=None, height=None, 
            size=1.0, verbose=False):
        # top-level call 
        system = System()
        self.on_constrain(system, 0, verbose)
        system.add(self.pip_x == x)
        system.add(self.pip_y == y)
        system.add(self.pip_z == z)
        if width is not None and hasattr(self, "left"):
            system.add(self.width == size*width)
        elif width is None and hasattr(self, "hunits"):
            system.add(self.width == 0.7*size*self.hunits)
        if height is not None and hasattr(self, "top"):
            system.add(self.height == size*height)
        elif height is None and hasattr(self, "vunits"):
            system.add(self.height == 0.5*size*self.vunits)
        if depth is not None:
            system.add(self.depth == size*depth)
        elif hasattr(self, "dunits"):
            system.add(self.depth == 0.5*size*self.dunits)
        return system

    did_layout = None
    def layout(self, *args, callback=None, verbose=False, **kw):
        # top-level call 
        #print("Render.layout")
        if self.did_layout:
            #print("already did_layout")
            return self.did_layout
        system = self.constrain(*args, verbose=verbose, **kw)
        if callback is not None:
            callback(self, system)
        system.solve()
        self.did_layout = system
        #self.dump(full=False)
        #return system

    def save_dbg(self, name):
        #self.layout()
        cvs = Canvas()
        self.dbg_render(cvs)
        cvs.writePDFfile(name)



class _Compound(object):
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        #Render.on_constrain(self, system)
        for cell in self.cells:
            cell.on_constrain(system, depth+1, verbose)

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

    color = None # fill
    stroke = black # outline
    st_stroke = []
    show_pip = False
    pip_cvs = None 
    assoc = True # does not work...
    space = 0.

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self][idx]

    def deepclone(self):
        kw = dict(self.__dict__)
        kw["name"] = self.name
        kw["weight"] = self.weight
        kw["color"] = self.color
        kw["stroke"] = self.stroke
        kw["st_stroke"] = self.st_stroke
        kw["show_pip"] = self.show_pip
        kw["pip_cvs"] = self.pip_cvs
        kw["assoc"] = self.assoc
        kw["space"] = self.space
        cell = _Cell0(**kw)
        check_renderable(cell)
        return cell


class _Cell0(Cell0, Render):

    def eq_constrain(self, other, system):
        add = system.add
        for attr in "pip_x pip_y pip_z front back".split():
            lhs = getattr(self, attr)
            rhs = getattr(other, attr)
            add(lhs == rhs)

    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.on_constrain(self, system, depth, verbose)
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

    def deepclone(self):
        kw = dict(self.__dict__)
        del kw["name"]
        del kw["cells"]
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _DCell0(cells, **kw)
        check_renderable(cell)
        return cell


class _DCell0(DCell0, _Compound, _Cell0):
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell0.on_constrain(self, system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
        if not len(self):
            return
        add = system.add
        y = self.pip_y - self.front
        w = 1./len(self)
        for cell in self.cells:
            add(self.pip_x == cell.pip_x) # _align
            add(self.pip_z == cell.pip_z) # _align
            add(cell.pip_y - cell.front == y, self.weight) # soft equal
            add(cell.depth == w*self.depth, self.weight) # soft equal
            y += cell.depth
        #add(self.pip_y + self.back == y, self.weight) # should be hard equal?
        add(self.pip_y + self.back == y) # should be hard equal?



setop(Cell0, "__matmul__", DCell0)

Cell0.ii = DCell0([])



# -------------------------------------------------------


class Cell1(Atom):
    """
        These are the 1-cells.
    """

    color = black
    st_stroke = []
    show_pip = True
    pip_radius = 0.3
    pip_cvs = None
    assoc = True # does not work...

    def __init__(self, tgt, src, name=None, weight=1.0, **kw):
        assert isinstance(tgt, Cell0)
        assert isinstance(src, Cell0)
        if name is None:
            name = "(%s<---%s)"%(tgt, src)
        Atom.__init__(self, name, weight, **kw) # will update kw's on __dict__
        self.tgt = tgt
        self.src = src
        self.hom = (self.tgt, self.src)

    def deepclone(self):
        tgt = self.tgt.deepclone()
        src = self.src.deepclone()
        kw = {}
        kw["color"] = self.color
        kw["st_stroke"] = self.st_stroke
        kw["show_pip"] = self.show_pip
        kw["pip_radius"] = self.pip_radius
        kw["pip_cvs"] = self.pip_cvs
        kw["assoc"] = self.assoc
        cell = _Cell1(tgt, src, self.name, self.weight, **kw)
        check_renderable(cell)
        return cell

    def extrude(self, show_pip=False, **kw):
        return Cell2(self, self, show_pip=show_pip, **kw)

    def reassoc(self):
        yield [self]

    def npaths(self):
        return len(list(self.get_paths()))

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)


class _Cell1(Cell1, Render):
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

    def fail_match(tgt, src, message=""):
        print("fail_match:", message)
        print("\t", tgt)
        print("\t", src)
        assert 0
    
    def match(tgt, src):
        assert isinstance(src, Cell1)
        if tgt.name == src.name:
            tgt = tgt.search(instance=Cell1)
            src = src.search(instance=Cell1)
            assert len(src) == len(tgt)
            for (t,s) in zip(tgt, src):
                yield (t,s)
            return # <---------- return
        send = {}
        lhs = list(tgt.get_paths())
        rhs = list(src.get_paths())
        if len(lhs) != len(rhs):
            tgt.fail_match(src, "%s tgt paths != %s src paths"%(len(lhs), len(rhs)))
    
        for (left,right) in zip(lhs, rhs):
            if len(left)!=len(right):
                tgt.fail_match(src, 
                    "tgt path len %d != src path len %d"%(len(left), len(right)))
            for (l,r) in zip(left, right):
                if l in send:
                    if send[l] != r:
                        tgt.fail_match(src)
                send[l] = r
                if type(l) != type(r):
                    tgt.fail_match(src)
                if l.name != r.name:
                    tgt.fail_match(src)
                if isinstance(l, Cell1):
                    yield (l,r)

    def eq_constrain(self, other, system):
        add = system.add
        for attr in "pip_x pip_y pip_z front back left right".split():
            lhs = getattr(self, attr)
            rhs = getattr(other, attr)
            add(lhs == rhs)
        self.tgt.eq_constrain(other.tgt, system)
        self.src.eq_constrain(other.src, system)

    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.on_constrain(self, system, depth, verbose)
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
        #system.add(self.left >= 0) # doesn't seem to be needed
        #system.add(self.right >= 0) # doesn't seem to be needed

        if self.__class__ != _Cell1: # bit of a hack
            return # < --------- return

        # now we constrain the _Cell0 src & tgt
        tgt, src = self.tgt, self.src
        tgt.on_constrain(system, depth, verbose)
        src.on_constrain(system, depth, verbose)
        add = system.add
        add(tgt.pip_x == self.pip_x - self.left) # tgt to the left
        add(src.pip_x == self.pip_x + self.right) # src to the right
        for cell in [tgt, src]:
            add(cell.pip_z == self.pip_z) # hard equal
            add(cell.pip_y - cell.front == self.pip_y - self.front)
            add(cell.pip_y + cell.back == self.pip_y + self.back)

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




class DCell1(Compound, Cell1):
    bdy = DCell0
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "@".join(cell.name for cell in cells) + ")"
        Cell1.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self):
        kw = {}
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        #kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _DCell1(cells, **kw)
        if not check_renderable(cell):
            dump(cell)
            assert 0, "found non Render's"
        return cell

    def extrude(self, show_pip=False, **kw):
        cells = [cell.extrude(show_pip=show_pip, **kw) for cell in self.cells]
        return DCell2(cells, show_pip=show_pip, **kw)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)


class _DCell1(DCell1, _Compound, _Cell1):

    def get_paths(self):
        for cell in self.cells:
            for path in cell.get_paths():
                yield path

    bdy = _DCell0
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell1.on_constrain(self, system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
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
        #add(self.pip_y + self.back == y, self.weight) # soft constrain, nah
        add(self.pip_y + self.back == y) # hard constrain, yes!

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

    def deepclone(self):
        kw = {}
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        #kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _HCell1(cells, **kw)
        check_renderable(cell)
        return cell

    def extrude(self, show_pip=False, **kw):
        cells = [cell.extrude(show_pip=show_pip, **kw) for cell in self.cells]
        return HCell2(cells, show_pip=show_pip, **kw)

    def traverse(self, callback, depth=0, full=True):
        Atom.traverse(self, callback, depth, full)
        if full:
            self.tgt.traverse(callback, depth+1, full)
            self.src.traverse(callback, depth+1, full)
        for cell in self.cells:
            cell.traverse(callback, depth+1, full)


class _HCell1(HCell1, _Compound, _Cell1):
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

    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        _Cell1.on_constrain(self, system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
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

    def dbg_render(self, cvs):
        for child in self.cells:
            child.dbg_render(cvs)



setop(Cell1, "__matmul__", DCell1)
setop(Cell1, "__lshift__", HCell1)



# -------------------------------------------------------

# These are helper classes for _Cell2.render below

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

    def __getitem__(self, idx):
        return self.vs[idx]

    def __len__(self):
        return len(self.vs)


class Surface(object):
    pip_cvs = None

    def __init__(self, segments, color=(0,0,0,1), address=None):
        self.segments = list(segments)
        self.color = color
        self.address = address

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
        view.add_surface(self.segments, fill=self.color, address=self.address)
        if Cell2.DEBUG or 0:
            for seg in self.segments:
                view.add_curve(*seg, stroke=(0,0,1,0.2), lw=0.2, epsilon=None)


# -------------------------------------------------------



class Cell2(Atom):
    "These are the 2-cells"

    DEBUG = False
    color = black
    show_pip = True
    pip_radius = 0.5
    cone = 0.6 # closer to 1. is more cone-like
    pip_cvs = None
    assoc = True # does not work...

    def __init__(self, tgt, src, name=None, **kw):
        assert isinstance(tgt, Cell1), tgt.__class__.__name__
        assert isinstance(src, Cell1), src.__class__.__name__
        assert tgt.src.name == src.src.name, "%s != %s" % (tgt.src, src.src)
        assert tgt.tgt.name == src.tgt.name, "%s != %s" % (tgt.tgt, tgt.src)
        if name is None:
            name = "(%s<===%s)"%(tgt, src)
        Atom.__init__(self, name, **kw) # update's kw's on __dict__
        self.tgt = tgt
        self.src = src
        self.hom = (self.tgt, self.src)

    def deepclone(self):
        kw = {}
        kw["DEBUG"] = self.DEBUG
        kw["color"] = self.color
        kw["show_pip"] = self.show_pip
        kw["pip_radius"] = self.pip_radius
        kw["cone"] = self.cone
        kw["pip_cvs"] = self.pip_cvs
        kw["assoc"] = self.assoc
        tgt = self.tgt.deepclone()
        src = self.src.deepclone()
        cell = _Cell2(tgt, src, **kw)
        check_renderable(cell)
        return cell

    @property
    def hunits(self):
        return 1

    @property
    def dunits(self):
        return 1

    @property
    def vunits(self):
        return 1

    def vflip(self):
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

    def _on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        Render.on_constrain(self, system, depth, verbose)
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

    def on_constrain(self, system, depth, verbose=False):
        self._on_constrain(system, depth, verbose)
        tgt, src = self.tgt, self.src
        tgt.on_constrain(system, depth, verbose)
        src.on_constrain(system, depth, verbose)
        add = system.add
        add(tgt.pip_z == self.pip_z + self.top) # hard equal
        add(src.pip_z == self.pip_z - self.bot) # hard equal
        for cell in [tgt, src]:
            add(cell.pip_x - cell.left == self.pip_x - self.left) # hard equal
            add(cell.pip_x + cell.right == self.pip_x + self.right) # hard equal
            add(cell.pip_y - cell.front == self.pip_y - self.front) # hard equal
            add(cell.pip_y + cell.back == self.pip_y + self.back) # hard equal

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

        def seg_over(v1, v2):
            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
            line = Segment(v1, v1 + cone*(v12-v1), v2 + cone*(v12-v2), v2)
            return line

        def callback(self): # XXX make this a method with args
            assert self.__class__ == _Cell1
            #print("callback", self.__class__.__name__, self)

            color = self.color
            pip1 = Mat(self.pip)

            line = seg_over(pip1, pip2)
            if color is not None:
                view.add_curve(*line, lw=0.2, stroke=color, st_stroke=self.st_stroke)
                # show spider (1-cell) pip
                if self.pip_cvs is not None:
                    view.add_cvs(Mat(pip1), self.pip_cvs)
                elif self.show_pip:
                    view.add_circle(pip1, self.pip_radius, fill=color)

            tgt, src = self.tgt, self.src
            for cell in tgt:
                assert isinstance(cell, _Cell0)
                color = cell.color
                pip0 = Mat(cell.pip)
                vpip1 = Mat([conv(pip0[0], pip1[0]), pip0[1], pip0[2]])
                leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1) # spider leg
                line2 = seg_over(pip0, pip2).reversed
                triangle = Surface([
                    line2,
                    leg, 
                    line,  # pip1 --> pip2
                ], color, address=None)
                surfaces.append(triangle)
                if abs(cell.pip_x - x0) < 2*PIP:
                    l_ports.append( (triangle[0], cell) )
                if cell.space>0.:
                    space = max(0., min(1., cell.space))
                    leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1+space*(vpip1-pip1)) # spider leg
                if cell.stroke is not None:
                    view.add_curve(*leg, stroke=cell.stroke, st_stroke=cell.st_stroke)

            for cell in src:
                assert isinstance(cell, _Cell0)
                color = cell.color
                pip0 = Mat(cell.pip)
                vpip1 = Mat([conv(pip0[0], pip1[0]), pip0[1], pip0[2]])
                line2 = seg_over(pip0, pip2).reversed
                leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1)
                triangle = Surface([
                    line2,
                    leg, 
                    line,  # pip1 --> pip2
                ], color, address=None)
                surfaces.append(triangle)
                if abs(cell.pip_x - x1) < 2*PIP:
                    r_ports.append( (triangle[0], cell) )
                if cell.space>0.:
                    space = max(0., min(1., cell.space))
                    leg = Segment(pip0, conv(pip0, vpip1), vpip1, pip1+space*(vpip1-pip1)) # spider leg
                if cell.stroke is not None:
                    view.add_curve(*leg, stroke=cell.stroke, st_stroke=cell.st_stroke)

        # left and right ports
        l_ports = l_tgt = []
        r_ports = r_tgt = []
        z = z1
        tgt.visit(callback, instance=_Cell1)

        # left and right ports
        l_ports = l_src = []
        r_ports = r_src = []
        z = z0
        src.visit(callback, instance=_Cell1)

        #for cell in tgt.search(instance=_Cell1)+src.search(instance=_Cell1):
        #    if cell.show_pip:
        #        view.add_circle(Mat(cell.pip), cell.pip_radius, fill=cell.color)


        if len(l_src) != len(l_tgt) or len(r_src) != len(r_tgt):
            print("_Cell2.render: FAIL", id(self), end=" ")
            print( len(l_src) , len(l_tgt) , end=" ")
            print( len(r_src) , len(r_tgt) )
            for surface in surfaces:
                surface.render(view)
            return

        assert len(l_src) == len(l_tgt)
        assert len(r_src) == len(r_tgt)

        # now we join the top and bot triangles on the left
        for (p_src, p_tgt) in zip(l_src, l_tgt):
            cell = p_src[1]
            seg_src, seg_tgt = p_src[0], p_tgt[0]
            seg = Segment.mk_line(seg_src[-1], seg_tgt[-1])
            surf = Surface([seg_src, seg, seg_tgt.reversed], cell.color, address=cell)
            surf.pip_cvs = cell.pip_cvs # grab this attr for below
            surfaces.append(surf)

        # now we join the top and bot triangles on the right
        for (p_src, p_tgt) in zip(r_src, r_tgt):
            cell = p_tgt[1]
            seg_src, seg_tgt = p_src[0], p_tgt[0]
            seg = Segment.mk_line(seg_src[-1], seg_tgt[-1])
            surf = Surface([seg_src, seg, seg_tgt.reversed], cell.color, address=cell)
            surf.pip_cvs = cell.pip_cvs # grab this attr for below
            surfaces.append(surf)

        #surfaces = Surface.merge(surfaces)
        for surface in surfaces:
            surface.render(view)
        for surface in surfaces:
            if surface.pip_cvs is None:
                continue
            view.add_cvs(surface.midpoint(), surface.pip_cvs)

        if self.pip_cvs is not None:
            view.add_cvs(Mat(pip2), self.pip_cvs)
        elif self.color is not None and self.show_pip:
            view.add_circle(Mat(pip2), self.pip_radius, fill=self.color, address=self)

    def render_boundary(self, view, src=True, tgt=True):
        (x0, y0, z0, x1, y1, z1) = self.rect
        x01 = conv(x0, x1)
        y01 = conv(y0, y1)
        z01 = conv(z0, z1)

        pip2 = Mat(self.pip)
        cone = 1. - self.cone
        cone = max(PIP, cone)

        def seg_over(v1, v2):
            v12 = Mat([v1[0], v1[1], v2[2]]) # a point over v1
            line = Segment(v1, v1 + cone*(v12-v1), v2 + cone*(v12-v2), v2)
            return line

        def callback(self):
            assert self.__class__ == _Cell1
            color = self.color
            pip1 = Mat(self.pip)

            line = seg_over(pip1, pip2)
            if color is not None:
                #view.add_curve(*line, lw=0.2, stroke=color)
                # show spider (1-cell) pip
                if self.show_pip:
                    view.add_circle(pip1, self.pip_radius, fill=color)

            tgt, src = self.tgt, self.src
            for cell in tgt:
                v = Mat(cell.pip)
                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
                leg = Segment(v, conv(v, vpip1), vpip1, pip1) # spider leg
                view.add_curve(*leg, stroke=cell.stroke)

            for cell in src:
                v = Mat(cell.pip)
                vpip1 = Mat([conv(v[0], pip1[0]), v[1], v[2]])
                leg = Segment(v, conv(v, vpip1), vpip1, pip1)
                view.add_curve(*leg, stroke=cell.stroke)

        if tgt:
            z = z1
            self.tgt.visit(callback, instance=_Cell1)

        if src:
            z = z0
            self.src.visit(callback, instance=_Cell1)

    def render_src(self, view):
        self.render_boundary(view, src=True, tgt=False)

    def render_tgt(self, view):
        self.render_boundary(view, src=False, tgt=True)

    def dbg_render(self, cvs):
        self.tgt.dbg_render(cvs)
        self.src.dbg_render(cvs)

    def render_cvs(self, pos="center"):
        view = View(400, 400, sort_gitems=False)
        view.ortho()
        #                .pip_x  .pip_y  .pip_z
        # negative dir:  .left   .front  .bot
        # positive dir:  .right  .back   .top
        x1, y1, z1 = self.center
        R = 3.
        x0, x2 = x1-R*self.left, x1+R*self.right
        z0, z2 = z1-R*self.bot, z1+R*self.top
        R = 3.
        x, y, z = 0., -R, 0.
        if pos == "center":
            x, z = 0., 0.
        elif pos == "north":
            z = z2
        elif pos == "northeast":
            x = x0
            z = z2
        elif pos == "northwest":
            z = z2
            x = x1
        elif pos == "south":
            z = z0
        elif pos == "southeast":
            x = x0
            z = z0
        elif pos == "southwest":
            z = z0
            x = x1
        elif pos == "east":
            x = x0
        elif pos == "west":
            x = x1
        else:
            assert 0, "pos %r not understood"%(pos,)
        view.lookat([x, y, z], [x1, y1, z1], [0, 0, 1]) # eyepos, lookat, up

        self.render(view)

        # just does not work well enough...
        # we have to sort: GCurve, GSurface, GCircle
        def less_than(lhs, rhs):
            to_sort = [pov.GSurface, pov.GCurve, pov.GCircle, pov.GCvs]
            # lhs < rhs means draw lhs before rhs, lhs is *behind* rhs
            depth = view.get_depth(lhs) < view.get_depth(rhs)
            ltp, rtp = type(lhs), type(rhs)
            if ltp == rtp:
                return depth
            elif lhs.incident(rhs, 0.1):
                #print("*", end=" ")
                idx, jdx = to_sort.index(ltp), to_sort.index(rtp)
                return idx < jdx
            return depth

        #shuffle(view.gitems)
        cvs = Canvas()
        view.render(cvs=cvs, less_than=less_than)
        return cvs





class DCell2(Compound, Cell2):
    bdy = DCell1
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "@".join(cell.name for cell in cells) + ")"
        Cell2.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self):
        kw = {}
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        #kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _DCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def hunits(self):
        return max(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return sum(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return max(cell.vunits for cell in self.cells)

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
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._on_constrain(system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        y = self.pip_y - self.front
        w = 1./self.dunits
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
            add(cell.depth == w*cell.dunits*self.depth, self.weight) # soft
            y += cell.depth
        #add(self.pip_y + self.back == y, self.weight)
        add(self.pip_y + self.back == y) # don't be a softy!

    def render(self, view):
        #Shape.render(self, view)
        for cell in reversed(self.cells):
            cell.render(view)




class HCell2(Compound, Cell2):
    bdy = HCell1
    def __init__(self, cells, **kw):
        cells = self._associate(cells)
        tgt = self.bdy([cell.tgt for cell in cells])
        src = self.bdy([cell.src for cell in cells])
        name = "(" + "<<".join(cell.name for cell in cells) + ")"
        Cell2.__init__(self, tgt, src, name, **kw)
        self.cells = cells

    def deepclone(self):
        kw = {}
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        #kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _HCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def hunits(self):
        return sum(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return max(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return max(cell.vunits for cell in self.cells)

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
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._on_constrain(system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
        add = system.add
        cells = self.cells
        assert cells, "??"
        x = self.pip_x - self.left # start here
        w = 1./self.hunits
        for cell in self.cells:
            add(cell.pip_y-cell.front == self.pip_y-self.front)
            add(cell.pip_y+cell.back == self.pip_y+self.back)
            #add(cell.pip_y == self.pip_y)

            add(cell.pip_z+cell.top == self.pip_z+self.top) 
            add(cell.pip_z-cell.bot == self.pip_z-self.bot) 
            #add(cell.pip_z == self.pip_z)
            #add(self.height == cell.height) # fit height

            add(cell.pip_x == cell.left + x)
            add(cell.width == w*cell.hunits*self.width, self.weight) # soft equal
            x += cell.width

        #add(self.pip_x + self.right == x, self.weight)
        add(self.pip_x + self.right == x) # hard equal

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

    def deepclone(self):
        kw = {}
        kw["show_pip"] = self.show_pip
        kw["color"] = self.color
        #kw["stroke"] = self.stroke
        cells = [cell.deepclone() for cell in self.cells]
        cell = _VCell2(cells, **kw)
        check_renderable(cell)
        return cell

    @property
    def hunits(self):
        return max(cell.hunits for cell in self.cells)

    @property
    def dunits(self):
        return max(cell.dunits for cell in self.cells)

    @property
    def vunits(self):
        return sum(cell.vunits for cell in self.cells)

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
    def on_constrain(self, system, depth, verbose=False):
        if verbose:
            dbg_constrain(self, depth)
        self._on_constrain(system, depth, verbose)
        _Compound.on_constrain(self, system, depth, verbose) # constrain children
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
            src, tgt = cells[i:i+2] # Cell2's
            src, tgt = src.tgt, tgt.src # Cell1's
            for (t, s) in tgt.match(src):
                # t, s are Cell1's
                #add(s.pip_x == t.pip_x) # hard equal
                #add(s.pip_y == t.pip_y) # hard equal
                s.eq_constrain(t, system)
            i += 1




setop(Cell2, "__matmul__", DCell2)
setop(Cell2, "__lshift__", HCell2)
setop(Cell2, "__mul__", VCell2)

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
    n = Cell0("n", color=grey, stroke=(0,0,0,1))
    k = Cell0("k", color=grey, stroke=(0,0,0,1))
    _n = Cell1(ii, n)
    n_ = Cell1(n, ii)
    nn_n = Cell1(n@n, n)
    n_nn = Cell1(n, n@n)
    n_n = Cell1(n, n, show_pip=False, color=None)
    pair = Cell1(n, n, show_pip=False, color=black)
    #swap = Cell1(n@n, n@n, show_pip=True, color=black)
    swap = Cell1(n@n, n@n, show_pip=False)

# broken
#    assert n_nn.npaths() == 2
#    assert (n_nn @ nn_n).npaths() == 4
#    count = (n_nn << nn_n).npaths()
#    cell = n_nn << nn_n
#    print(cell)
#    assert count == 2, count
#    assert (n_nn << (n_n @ n_nn)).npaths() == 3

    # we use II to get match to work
    I = Cell2(n_n, n_n, show_pip=False)
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
        Cell0(name, color=scheme[i%len(scheme)], address=name) 
        for i, name in enumerate('lmnop')]
    i0 = Cell0("i", color=None)

    I_l = Cell1(l, l, show_pip=False, color=None)
    I_m = Cell1(m, m, show_pip=False, color=None)
    I_n = Cell1(n, n, show_pip=False, color=None)
    I_o = Cell1(o, o, show_pip=False, color=None)

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
        I_l = Cell1(l, l, show_pip=False, color=None)
        I_m = Cell1(m, m, show_pip=False, color=None)
        I_n = Cell1(n, n, show_pip=False, color=None)
        tgt = (I_n @ swap(m, l)) << (swap(n, l) @ I_m) << (I_l @ swap(n, m))
        src = (swap(n, m) @ I_l) << (I_m @ swap(n, l)) << (swap(m, l) @ I_n)
        if reversed:
            tgt, src = src, tgt
        morph = Cell2(tgt, src, cone=1.0, show_pip=False, **kw)
        return morph
    
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
        Cell0(name, color=scheme[i%len(scheme)], address=name) 
        for i, name in enumerate('lmnop')]

    pip_cvs = Canvas().text(0, 0, "$n$", st_center)
    i = Cell1(l@l, l, show_pip=False, color=None)
    f = i.extrude()(pip_cvs = pip_cvs)
    f = f.layout()
    cvs = f.render_cvs("north")
    #cvs.writePDFfile("test_render.pdf")

    i = Cell1(l@l(space=0.2), l(space=0.2)@l, show_pip=False, st_stroke=st_dotted)
    f = Cell2(i, i, show_pip=False)
    f = f.layout(width=1, height=1)
    cvs = f.render_cvs("north")
    cvs.writePDFfile("test_render.pdf")



if __name__ == "__main__":
    print("\n")
    #test()
    #test_match()
    #more_test()
    test_render()

    print("OK")

