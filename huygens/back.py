#!/usr/bin/env python3

"""
Expose a _drawing api with multiple (hopefully) backends.

The api uses the first quadrant coordinate system:

        y 
        ^
        |
        |
        | positive x and y
        |    
        |
      --+-------------------> x
        |
    
"""

from math import pi, sqrt, sin, cos, sqrt, floor
from copy import deepcopy

from huygens.base import SCALE_CM_TO_POINT, Base, Matrix
from huygens.text import make_text

EPSILON = 1e-8

# ----------------------------------------------------------------------------
# 
#

def n_min(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def n_max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


class Bound(Base):
    def __init__(self, llx=None, lly=None, urx=None, ury=None):
        assert urx is None or llx is None or urx-llx >= 0., (llx, lly, urx, ury)
        assert ury is None or lly is None or ury-lly >= 0., (llx, lly, urx, ury)
        self.llx = llx
        self.lly = lly
        self.urx = urx
        self.ury = ury

    def union(self, other):
        "union"
        llx = n_min(self.llx, other.llx)
        lly = n_min(self.lly, other.lly)
        urx = n_max(self.urx, other.urx)
        ury = n_max(self.ury, other.ury)
        return Bound(llx, lly, urx, ury)
    __add__ = union

    def update(self, other):
        "union update"
        self.llx = n_min(self.llx, other.llx)
        self.lly = n_min(self.lly, other.lly)
        self.urx = n_max(self.urx, other.urx)
        self.ury = n_max(self.ury, other.ury)

    def scale_point_to_cm(self):
        llx = self.llx / SCALE_CM_TO_POINT
        lly = self.lly / SCALE_CM_TO_POINT
        urx = self.urx / SCALE_CM_TO_POINT
        ury = self.ury / SCALE_CM_TO_POINT
        return Bound(llx, lly, urx, ury)

    def nonempty(self):
        return (self.llx is not None or self.lly is not None or 
            self.urx is not None or self.ury is not None)

    def is_empty(self):
        return (self.llx is None and self.lly is None and 
            self.urx is None and self.ury is None)

    def __getitem__(self, idx):
        return (self.llx, self.lly, self.urx, self.ury)[idx]

    @property
    def center(self):
        return 0.5*(self.llx + self.urx), 0.5*(self.lly + self.ury)

    @property
    def width(self):
        return self.urx - self.llx

    @property
    def height(self):
        return self.ury - self.lly

    def contains(self, x, y, epsilon=1e-6):
        return (self.llx-epsilon <= x and self.lly-epsilon <= y and
            self.urx+epsilon >= x and self.ury+epsilon >= y)




# ----------------------------------------------------------------------------
# 
#


class Visitor(object):
    def on_visit(self, item):
        return item

class DumpVisitor(Visitor):
    def on_visit(self, item):
        print('\t%s'%item)
        return item


# XXX this probably should be a full fledged Context XXX
class BoundVisitor(Visitor):
    def __init__(self):
        self.pos = None
        #self.lw = _defaultlinewidth*SCALE_CM_TO_POINT
        self.lw = _defaultlinewidth
        self.bound = Bound()

    def on_visit(self, item):
        if isinstance(item, (Scale, Rotate, Translate)):
            assert 0, "trafo %s not implemented"%item
        elif isinstance(item, Compound):
            assert 0, "%s: save restore not implemented"%item
        elif isinstance(item, MoveTo):
            self.pos = item.x, item.y
        elif isinstance(item, (Stroke, Fill)):
            self.pos = None
        elif isinstance(item, LineWidth):
            self.lw = item.lw
        elif isinstance(item, (LineTo, CurveTo)):
            b = item.get_bound()
            if b.is_empty():
                assert 0
            # add a generous linewidth... does not work very well... XXX
            r = 0.5*self.lw
            b.llx -= r
            b.lly -= r
            b.urx += r
            b.ury += r
            self.bound.update(b)
            if self.pos is not None:
                x, y = self.pos
                self.bound.update(Bound(x, y, x, y))
        else:
            pass
        #print("BoundVisitor.on_visit", self.bound)
        return item


class Replacer(Visitor):
    def __init__(self, src, tgt):
        assert isinstance(src, Item)
        assert isinstance(tgt, Item)
        self.src = src
        self.tgt = tgt

    def on_visit(self, item):
        #if isinstance(item, RGBA):
        #    print(item, self.src, item==self.src)
        if item == self.src:
            #print("Replacer", self.src, self.tgt)
            item = self.tgt
        return item


# ----------------------------------------------------------------------------
# 
#


class Item(Base):

    DEBUG = False

    def pstr(self, indent=0):
        return "  "*indent + str(self)

    def get_bound(self):
        return Bound()

    def get_bound_cairo(self):
        # Not as tight as it could be.... ?
        import cairo
        surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        cxt = cairo.Context(surface)
        self.process_cairo(cxt)
        extents = surface.ink_extents()
        (ulx, uly, width, height) = extents
        llx, lly = ulx, -uly-height
        urx, ury = llx+width, lly+height
        return Bound(llx, lly, urx, ury)

    def get_bound_box(self): # cache this ?
        bb = self.get_bound_cairo()
        bb = bb.scale_point_to_cm()
        return bb

    def process_cairo(self, cxt):
        pass

    def visit(self, visitor, leaves_only=True):
        return visitor.on_visit(self)

    def rewrite(self, visitor):
        return visitor.on_visit(self)

    def dump(self):
        visitor = DumpVisitor()
        self.visit(visitor)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # XXX this is not in agreement with __eq__/__ne__
        return id(self)


class Empty(Item):
    pass


class PathItem(Item):
    "belongs in a Path"
    def get_length(self, curpos, startpos):
        assert 0, self.__class__

    def get_at(self, curpos, startpos, t):
        assert 0, self.__class__

    def diff_at(self, curpos, startpos, t):
        assert 0, self.__class__


class ClosePath(PathItem):
    def get_length(self, curpos, startpos):
        assert curpos is not None
        assert startpos is not None
        x0, y0 = curpos
        x1, y1 = startpos
        r = sqrt((x1-x0)**2 + (y1-y0)**2)
        return startpos, r

    def get_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert startpos is not None
        assert 0.<=t<=1.
        #print("LineTo.get_at", t, curpos, self)
        x0, y0 = curpos
        x1, y1 = startpos
        return (1.-t)*x0 + t*x1, (1.-t)*y0 + t*y1

    def diff_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        #print("LineTo.get_at", t, curpos, self)
        x0, y0 = curpos
        x1, y1 = startpos
        return (x1-x0), (y1-y0)

    def process_cairo(self, cxt):
        cxt.close_path()


class MoveTo(PathItem):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def get_length(self, curpos, startpos):
        return (self.x, self.y), 0.

    def process_cairo(self, cxt):
        x = SCALE_CM_TO_POINT*self.x
        y = SCALE_CM_TO_POINT*self.y
        if self.DEBUG:
            print("ctx.move_to", x, y)
        cxt.move_to(x, -y)

class LineTo(PathItem):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_length(self, curpos, startpos):
        assert curpos is not None, "no current point"
        x0, y0 = curpos
        x1, y1 = self.x, self.y
        r = sqrt((x1-x0)**2 + (y1-y0)**2)
        return (x1, y1), r

    def get_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        #print("LineTo.get_at", t, curpos, self)
        x0, y0 = curpos
        x1, y1 = self.x, self.y
        return (1.-t)*x0 + t*x1, (1.-t)*y0 + t*y1

    def diff_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        #print("LineTo.get_at", t, curpos, self)
        x0, y0 = curpos
        x1, y1 = self.x, self.y
        return (x1-x0), (y1-y0)

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        x = SCALE_CM_TO_POINT*self.x
        y = SCALE_CM_TO_POINT*self.y
        if self.DEBUG:
            print("ctx.line_to", x, y)
        cxt.line_to(x, -y)


class CurveTo(PathItem):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_length(self, curpos, startpos):
        assert curpos is not None, "no current point"
        # Totally fake the length... shameless!
        x0, y0 = curpos
        x1, y1 = self.x2, self.y2
        r = sqrt((x1-x0)**2 + (y1-y0)**2)
        return (x1, y1), r

    def get_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        x0, y0 = curpos
        x1, y1 = self.x0, self.y0
        x2, y2 = self.x1, self.y1
        x3, y3 = self.x2, self.y2
        x = ((-x0 + 3*x1 - 3*x2 + x3)*(t**3) + 
             (3*x0-6*x1+3*x2)*(t**2) +
             (-3*x0+3*x1)*t + x0)
        y = ((-y0 + 3*y1 - 3*y2 + y3)*(t**3) + 
             (3*y0-6*y1+3*y2)*(t**2) +
             (-3*y0+3*y1)*t + y0)
        return x, y

    def diff_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        x0, y0 = curpos
        x1, y1 = self.x0, self.y0
        x2, y2 = self.x1, self.y1
        x3, y3 = self.x2, self.y2
        dx = (3*(-x0 + 3*x1 - 3*x2 + x3)*(t**2) + 
              2*(3*x0-6*x1+3*x2)*t +
                (-3*x0+3*x1))
        dy = (3*(-y0 + 3*y1 - 3*y2 + y3)*(t**2) + 
              2*(3*y0-6*y1+3*y2)*t +
                (-3*y0+3*y1))
        return dx, dy

    def get_bound(self):
        x0 = min([self.x0, self.x1, self.x2])
        x1 = max([self.x0, self.x1, self.x2])
        y0 = min([self.y0, self.y1, self.y2])
        y1 = max([self.y0, self.y1, self.y2])
        return Bound(x0, y0, x1, y1)

    def process_cairo(self, cxt):
        x0 = SCALE_CM_TO_POINT*self.x0
        y0 = SCALE_CM_TO_POINT*self.y0
        x1 = SCALE_CM_TO_POINT*self.x1
        y1 = SCALE_CM_TO_POINT*self.y1
        x2 = SCALE_CM_TO_POINT*self.x2
        y2 = SCALE_CM_TO_POINT*self.y2
        if self.DEBUG:
            print("ctx.curve_to", x0, y0, x1, y1, x2, y2)
        cxt.curve_to(x0, -y0, x1, -y1, x2, -y2)



class Arc(PathItem):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in radians"
        self.x = x
        self.y = y
        self.r = r
        self.angle1 = angle1
        self.angle2 = angle2

    def get_length(self, curpos, startpos):
        r, x, y = self.r, self.x, self.y
        angle1, angle2 = self.angle1, self.angle2
        dangle = abs(angle1 - angle2)
        #x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
        x1, y1 = x+r*cos(angle2), y+r*sin(angle2)
        return (x1, y1), r*dangle

    def get_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        r = self.r
        x, y = self.x, self.y
        angle1, angle2 = self.angle1, self.angle2
        while angle1 > angle2:
            angle2 += 2*pi
        angle = (1.-t)*angle1 + t*angle2
        #x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
        #x1, y1 = x+r*cos(angle2), y+r*sin(angle2)
        #cx, cy = curpos
        px, py = x+r*cos(angle), y+r*sin(angle)
        return (px, py)

    def diff_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        r = self.r
        x, y = self.x, self.y
        angle1, angle2 = self.angle1, self.angle2
        while angle1 > angle2:
            angle2 += 2*pi
        angle = (1.-t)*angle1 + t*angle2
        #x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
        #x1, y1 = x+r*cos(angle2), y+r*sin(angle2)
        #cx, cy = curpos
        dx, dy = -r*sin(angle), r*cos(angle)
        return (dx, dy)

    def get_bound(self):
        r = self.r
        return Bound(self.x-r, self.y-r, self.x+r, self.y+r) # XXX TODO XXX

    def process_cairo(self, cxt):
        x = SCALE_CM_TO_POINT*self.x
        y = SCALE_CM_TO_POINT*self.y
        r = SCALE_CM_TO_POINT*self.r
        #print("cxt.arc_negative", self.__class__)
        cxt.arc_negative(x, -y, r, 2*pi-self.angle1, 2*pi-self.angle2)


class _ArcnMixin(object):
    def get_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        r = self.r
        x, y = self.x, self.y
        angle1, angle2 = self.angle1, self.angle2
        while angle1 < angle2:
            angle2 -= 2*pi
        angle = (1.-t)*angle1 + t*angle2
        #x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
        #x1, y1 = x+r*cos(angle2), y+r*sin(angle2)
        #cx, cy = curpos
        px, py = x+r*cos(angle), y+r*sin(angle)
        return (px, py)

    def diff_at(self, curpos, startpos, t):
        assert curpos is not None, "no current point"
        assert 0.<=t<=1.
        r = self.r
        x, y = self.x, self.y
        angle1, angle2 = self.angle1, self.angle2
        while angle1 < angle2:
            angle2 -= 2*pi
        angle = (1.-t)*angle1 + t*angle2
        #x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
        #x1, y1 = x+r*cos(angle2), y+r*sin(angle2)
        #cx, cy = curpos
        dx, dy = r*sin(angle), -r*cos(angle)
        return (dx, dy)

    def process_cairo(self, cxt):
        #print("cxt.arc", self.__class__)
        x = SCALE_CM_TO_POINT*self.x
        y = SCALE_CM_TO_POINT*self.y
        r = SCALE_CM_TO_POINT*self.r
        cxt.arc(x, -y, r, 2*pi-self.angle1, 2*pi-self.angle2)


class Arcn(_ArcnMixin, Arc):
    pass


class Arcn(_ArcnMixin, Arc):
    pass


# ----------------------------------------------------------------------------
#
#

class Compound(Item):
    def __init__(self, *args):
        items = []
        for arg in args:
            if type(arg) is list:
                items += arg
            else:
                items.append(arg)
        self.items = items

    def append(self, item):
        assert isinstance(item, Item), "expected type Item, got type %s"%(type(item),)
        items = self.items
        items.append(item)
        return self # yes...

    def extend(self, items):
        for item in items:
            self.append(item)
        return self # yes...

    def copy(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __setitem__(self, idx, item):
        assert isinstance(item, Item)
        self.items[idx] = item

    def __add__(self, other):
        assert isinstance(other, self.__class__), repr(other)
        return self.__class__(self.items + other.items)

    def __iadd__(self, other):
        assert isinstance(other, Compound), repr(other)
        self.items += other.items
        return self

    def visit(self, visitor, leaves_only=True):
        # depth first visit
        for item in self.items:
            item.visit(visitor, leaves_only) # recurse
        if not leaves_only:
            Item.visit(self, visitor, leaves_only)

    def pstr(self, indent=0):
        lines = []
        lines.append("  "*indent + self.__class__.__name__+"(")
        for item in self.items:
            lines.append(item.pstr(indent+1)+",")
        s = '\n'.join(lines) + ")"
        s = s.replace(",)", ")")
        return s

    def rewrite(self, visitor):
        items = self.items
        idx = 0
        while idx < len(items):
            item = items[idx].rewrite(visitor)
            if item is None:
                items.pop(idx)
            else:
                items[idx] = item
                idx += 1
        return Item.rewrite(self, visitor)

    def get_bound(self):
        visitor = BoundVisitor()
        self.visit(visitor)
        return visitor.bound

    def process_cairo(self, cxt):
        cxt.save()
        for item in self.items:
            item.process_cairo(cxt)
        cxt.restore()

    def outline(self, lw=0.01):
        # Brittle code, & only works on text (from latex etc.)
        self.rewrite(OutlineRewrite(lw))

    def get_path(self):
        visitor = PathVisitor()
        self.visit(visitor)
        return Path(*visitor.items)


class PathVisitor(Visitor):
    def __init__(self):
        self.items = []

    def on_visit(self, item):
        if isinstance(item, PathItem):
            self.items.append(item)


class OutlineRewrite(Visitor):
    def __init__(self, lw=0.01):
        self.lw = lw

    def on_visit(self, item):
        if isinstance(item, FillPreserve):
            return None # delete this Item
        elif isinstance(item, RGBA) and item.cl==(0,0,0,0):
            return None # delete this Item
        elif isinstance(item, LineWidth):
            item.lw = self.lw # mutate Item
        return item



class Path(Compound):

    # XXX should reqire PathItem elements ?

    def __add__(self, other):
        assert isinstance(other, Path), repr(other)
        return Path(self.items + other.items)

    def __iadd__(self, other):
        assert isinstance(other, Path), repr(other)
        self.items += other.items
        return self

    def reversed(self, pos=None): # used in flatten.py
        if len(self)==0:
            return self
        #print("Path.reversed")
        #for item in self:
        #    print('\t', item)
        n = len(self)
        assert n%2 == 0, "not implemented"
        items = []
        idx = n-2
        while idx>=0:
            item = self[idx]
            assert isinstance(item, LineTo), "not implemented"
            x, y = item.x, item.y
            item = self[idx+1]
            assert isinstance(item, CurveTo), "not implemented"
            x0, y0 = item.x0, item.y0
            x1, y1 = item.x1, item.y1
            x2, y2 = item.x2, item.y2
            items.append(LineTo(x2, y2))
            items.append(CurveTo(x1, y1, x0, y0, x, y))
            idx -= 2
        p = Path(items)
        return p

    def get_length(self):
        curpos = None
        startpos = None
        length = 0.
        for item in self:
            assert isinstance(item, PathItem)
            curpos, _length = item.get_length(curpos, startpos)
            startpos = startpos or curpos
            length += _length
        return length

    def tangent(self, t):
        "find point and differential on curve at time t, with 0<=t<=1."
        assert 0.<=t<=1., repr(t)
        curpos = None
        startpos = None
        length = 0.
        total = self.get_length()
        assert total > EPSILON, "empty path"
        closed = False
        for item in self:
            assert isinstance(item, PathItem)
            assert not closed, "only contiguous paths supported"
            closed = isinstance(item, ClosePath)
            _curpos, _length = item.get_length(curpos, startpos)
            startpos = startpos or curpos
            if _length < EPSILON:
                curpos = _curpos
                continue # <------- continue
            t0 = length / total
            t1 = (length + _length) / total
            if t0 <= t <= t1+EPSILON:
                t = (t-t0) / (t1-t0)
                x, y = item.get_at(curpos, startpos, t)
                #x = x/SCALE_CM_TO_POINT
                #y = y/SCALE_CM_TO_POINT
                dx, dy = item.diff_at(curpos, startpos, t)
                #dx = dx/SCALE_CM_TO_POINT
                #dy = dy/SCALE_CM_TO_POINT
                return x, y, dx, dy # <------------ return
            length += _length
            curpos = _curpos
        print(t0, t, t1)
        assert 0, "ran out of path: length=%s, total=%s"%(length, total)

    def getat(self, t):
        x, y, dx, dy = self.tangent(t)
        return x, y

    def normtangent(self, t):
        x, y, dx, dy = self.tangent(t)
        r = sqrt(dx**2 + dy**2)
        assert r > 1e-8, "wup"
        dx, dy = dx/r, dy/r # normalize
        return x, y, dx, dy

    def subpath(self, t0=0., t1=1., N=10):
        # bit of a hack but works...
        t0 = max(t0, 0.)
        t1 = min(t1, 1.)
        assert 0<=t0<=t1<=1., "t0=%f, t1=%f"%(t0, t1)
        if t0==0. and t1==1.:
            return self
        ps = []
        for i in range(N+1):
            t = (t1-t0)*i/N + t0
            x, y, dx, dy = self.tangent(t)
            r = sqrt(dx**2 + dy**2)
            assert r > 1e-8, "wup"
            dx, dy = dx/r, dy/r # normalize
            if i==0:
                ps.append(MoveTo(x, y))
            else:
                r = 0.5*sqrt((x-x0)**2 + (y-y0)**2)
                #ps.append(path.lineto(x, y))
                ps.append(CurveTo(x0+r*dx0, y0+r*dy0, x-r*dx, y-r*dy, x, y))
            x0, y0, dx0, dy0 = x, y, dx, dy
    
        return Path(ps)




class Line(Path):
    def __init__(self, x0, y0, x1, y1):
        Path.__init__(self, [
            MoveTo(x0, y0), 
            LineTo(x1, y1)])


class Curve(Path):
    def __init__(self, x0, y0, x1, y1, x2, y2, x3, y3):
        Path.__init__(self, [
            MoveTo(x0, y0), 
            CurveTo(x1, y1, x2, y2, x3, y3)])
    

class Rect(Path):
    def __init__(self, x, y, width, height):
        Path.__init__(self, [
            MoveTo(x, y), 
            LineTo(x+width, y),
            LineTo(x+width, y+height),
            LineTo(x, y+height),
            ClosePath()])


class Circle(Path):
    def __init__(self, x, y, r):
        Path.__init__(self, [
            MoveTo(x+r, y),
            Arc(x, y, r, 0, 2*pi),
            ClosePath()])
        

# XXX Arc is PathItem XXX



# ----------------------------------------------------------------------------
# Deco
#


class Deco(Item):
    def on_decorate(self, pre, item, post):
        pre.append(self)

    def __call__(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        #print(d)
        deco = self.__class__(**d)
        return deco


class CompoundDeco(Deco):
    def __init__(self, decos):
        self.decos = list(decos)

    def process_cairo(self, cxt):
        for deco in self.decos:
            deco.process_cairo(cxt)


class Stroke(Deco):
    def on_decorate(self, pre, item, post):
        post.append(self)

    def process_cairo(self, cxt):
        cxt.stroke()


class Fill(Deco):
    def on_decorate(self, pre, item, post):
        post.append(self)

    def process_cairo(self, cxt):
        import cairo
        cxt.set_fill_rule(cairo.FillRule.EVEN_ODD) # XXX where does this belong ?
        cxt.fill()


#class FillEvenOdd(Deco):


class FillPreserve(Deco):
    def on_decorate(self, pre, item, post):
        post.append(self)

    def process_cairo(self, cxt):
        cxt.fill_preserve()


class Clip(Deco):
    def on_decorate(self, pre, item, post):
        post.append(self)

    def process_cairo(self, cxt):
        cxt.clip()


class RGBA(Deco):
    def __init__(self, r, g=None, b=None, a=1.0):
        r = float(r)
        g = r if g is None else float(g)
        b = r if b is None else float(b)
        a = float(a)
        self.cl = (r, g, b, a)

    def __str__(self):
        return "RGBA(%.4f, %.4f, %.4f, %.4f)"%self.cl

    def __getitem__(self, idx):
        return self.cl[idx]

    def alpha(self, a):
        (r, g, b, _) = self.cl
        return RGBA(r, g, b, a)

    def __mul__(self, x):
        (r, g, b, a) = self.cl
        return RGBA(x*r, x*g, x*b, a)
    __rmul__ = __mul__

    def __add__(self, x):
        (r, g, b, a) = self.cl
        if isinstance(x, RGBA):
            rgb = RGBA(x[0]+r, x[1]+g, x[2]+b, a)
        else:
            rgb = RGBA(x+r, x+g, x+b, a)
        return rgb
    __radd__ = __add__

    def process_cairo(self, cxt):
        cxt.set_source_rgba(*self.cl)


class Paint(Item):
    def process_cairo(self, cxt):
        cxt.paint()


_defaultlinewidth = 0.02 # cm


class LineWidth(Deco):
    def __init__(self, lw):
        self.lw = lw

    def __rmul__(self, value):
        return LineWidth(float(value)*self.lw)
    __mul__ = __rmul__

    def process_cairo(self, cxt):
        lw = self.lw*SCALE_CM_TO_POINT
        cxt.set_line_width(lw)


# cairo constants:
#    cairo.LINE_CAP_BUTT           
#    cairo.LINE_CAP_ROUND         
#    cairo.LINE_CAP_SQUARE       
#    cairo.LINE_JOIN_BEVEL      
#    cairo.LINE_JOIN_MITER     
#    cairo.LINE_JOIN_ROUND    


class LineCap(Deco):
    def __init__(self, desc):
        assert type(desc) is str
        assert desc.lower() in "butt round square".split()
        self.desc = desc.upper()

    def process_cairo(self, cxt):
        import cairo
        cap = getattr(cairo, "LINE_CAP_%s"%(self.desc,))
        cxt.set_line_cap(cap)


class LineJoin(Deco):
    def __init__(self, desc):
        assert type(desc) is str
        assert desc.lower() in "bevel miter round".split()
        self.desc = desc.upper()

    def process_cairo(self, cxt):
        import cairo
        cap = getattr(cairo, "LINE_JOIN_%s"%(self.desc,))
        cxt.set_line_join(cap)


class LineDash(Deco):
    def __init__(self, dashes, offset=0):
        self.dashes = dashes
        self.offset = offset

    def process_cairo(self, cxt):
        import cairo
        cxt.set_dash(self.dashes, self.offset)


#class LineDash(LineDash):
#    def __init__(self, dashes, offset=0):
#        dashes = [sz*SCALE_CM_TO_POINT for sz in dashes]
#        LineDash.__init__(self, dashes, offset)


class LineDash(Deco):
    def __init__(self, dashes, offset=0):
        self.dashes = dashes
        self.offset = offset

    def process_cairo(self, cxt):
        import cairo
        lw = cxt.get_line_width()
        scale = lw / _defaultlinewidth / SCALE_CM_TO_POINT
        dashes = [sz * scale for sz in self.dashes]
        cxt.set_dash(dashes, self.offset)


class TextSize(Deco):
    def __init__(self, size_idx, latex_desc=None):
        scale = (2**(0.5*size_idx))
        self.size = 10. * scale
        self.scale = scale
        self.latex_desc = latex_desc # TODO

    def process_cairo(self, cxt):
        if the_text_cls == CairoText: # bit of a hack..
            cxt.set_font_size(self.size)
        #else:
        #    cxt.scale(self.scale, self.scale)


class TextAlign(Deco):
    def __init__(self, desc):
        self.desc = desc

    def on_decorate(self, pre, item, post):
        assert isinstance(item, Text)
        _, ury, width, height = item.text_extents()
        #print("TextAlign", ury, width, height)
        rise = ury
        drop = ury-height
        desc = self.desc
        dx, dy = 0., 0.
        if desc == "boxcenter":
            dx = 0.5*width
        elif desc == "boxright":
            dx = 1.0*width
        elif desc == "boxleft":
            dx = 0.0*width
        elif desc == "top":
            dy = 1.0*rise
        elif desc == "middle":
            dy = 0.5*(rise + drop)
        elif desc == "bottom":
            dy = drop
        else:
            assert 0, "TextAlign: unknown desc %r"%desc
        pre.append(Translate(-dx, -dy))

TextHAlign = TextAlign
TextVAlign = TextAlign


# ------ transform's ----------


class Transform(Deco):
    def __init__(self, xx=1.0, yx=0.0, xy=0.0, yy=1.0, x0=0.0, y0=0.0):
        self.xx = float(xx)
        self.yx = float(yx)
        self.xy = float(xy)
        self.yy = float(yy)
        self.x0 = float(x0)
        self.y0 = float(y0)

    def __getitem__(self, idx):
        return [self.xx, self.yx, self.xy, self.yy, self.x0, self.y0][idx]

    def __str__(self):
        return "Transform%s"%(tuple(self.get_matrix()),)

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        lhs = tuple(self.get_matrix())
        rhs = tuple(other.get_matrix())
        d = sum(abs(lhs[i] - rhs[i]) for i in range(6))
        #print("d =", d)
        return d < EPSILON

    def __ne__(self, other):
        return not self.__eq__(other)

    def transform_point(self, x, y):
        "translate a point, using huygens coordinates"
        m = self.get_matrix()
        return m.transform_point(x, y)

    def get_matrix(self):
        return Matrix(self.xx, self.yx, self.xy, self.yy, self.x0, self.y0)

    def inv(self):
        m = self.get_matrix()
        mi = m.inv()
        return Transform(*mi)

    def __mul__(self, other):
        left = self.get_matrix()
        right = other.get_matrix()
        m = left * right
        return Transform(*m)

    def __pow__(self, n):
        assert n>=0, "not implemented.."
        if n==0:
            return Transform()
        op = self
        while n>1:
            op = self*op
            n -= 1
        return op

    def interpolate(self, other, alpha=0.):
        assert other.__class__ is Transform
        conv = lambda a, b, alpha : (1-alpha)*a+alpha*b
        xx = conv(self.xx, other.xx, alpha)
        yx = conv(self.yx, other.yx, alpha)
        xy = conv(self.xy, other.xy, alpha)
        yy = conv(self.yy, other.yy, alpha)
        x0 = conv(self.x0, other.x0, alpha)
        y0 = conv(self.y0, other.y0, alpha)
        return Transform(xx, yx, xy, yy, x0, y0)

    def process_cairo(self, cxt):
        import cairo
        x0 = self.x0*SCALE_CM_TO_POINT
        y0 = self.y0*SCALE_CM_TO_POINT
        m = cairo.Matrix(self.xx, -self.yx, -self.xy, self.yy, x0, -y0)
        cxt.transform(m)


class Translate(Transform):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def transform_point(self, x, y):
        "translate a point, using huygens coordinates"
        dx = self.dx
        dy = self.dy
        return (x+dx, y+dy)

    def get_matrix(self):
        return Matrix.translate(self.dx, self.dy)

    def process_cairo(self, cxt):
        dx = self.dx*SCALE_CM_TO_POINT
        dy = self.dy*SCALE_CM_TO_POINT
        cxt.translate(dx, -dy)


class Scale(Transform):
    def __init__(self, sx, sy=None, x=0., y=0.):
        if sy is None:
            sy = sx
        assert abs(sx) > EPSILON
        assert abs(sy) > EPSILON
        assert sx < 1e5, "scale %s too big, probably ?"%sx
        assert sy < 1e5, "scale %s too big, probably ?"%sy
        self.sx = float(sx)
        self.sy = float(sy)
        self.x = float(x)
        self.y = float(y)

    def transform_point(self, x, y):
        "translate a point, using huygens coordinates"
        sx, sy = self.sx, self.sy
        x0, y0 = self.x, self.y
        dx, dy = (1.-sx)*x0, (1.-sy)*y0
        x += dx
        y += dy
        x *= sx
        y *= sy
        return (x, y)

    def get_matrix(self):
        sx, sy = self.sx, self.sy
        x, y = self.x, self.y
        dx, dy = (1.-sx)*x, (1.-sy)*y # <--- tricky !
        m = Matrix.translate(dx, -dy)
        m = Matrix.scale(sx, sy) * m
        return m

    def process_cairo(self, cxt):
        sx, sy = self.sx, self.sy
        x = self.x*SCALE_CM_TO_POINT
        y = self.y*SCALE_CM_TO_POINT
        dx, dy = (1.-sx)*x, (1.-sy)*y # <--- tricky !
        cxt.translate(dx, -dy)
        try:
            cxt.scale(sx, sy)
        except: # cairo.Error:
            print("Scale.process_cairo", sx, sy)
            raise


class Rotate(Transform):
    def __init__(self, angle, x=0., y=0.):
        "rotate by angle in radians around point at x,y"
        self.angle = angle
        self.x = float(x)
        self.y = float(y)

    def transform_point(self, x, y):
        "translate a point, using huygens coordinates"
        x0, y0 = self.x, self.y
        x, y = x-x0, y-y0
        s, c = sin(self.angle), cos(self.angle)
        x, y = (c*x+s*y, -s*x+c*y)
        x, y = x+x0, y+y0
        return (x, y)

    def get_matrix(self):
        m = Matrix.rotate(self.angle, self.x, self.y)
        return m

    def process_cairo(self, cxt):
        x = self.x*SCALE_CM_TO_POINT
        y = self.y*SCALE_CM_TO_POINT
        cxt.translate(x, -y)
        cxt.rotate(self.angle)
        cxt.translate(-x, y)


# ----------------------------------------------------------------------------
# 
#

class Image(Item): # abstract base class
    def __init__(self, x=0, y=0, alpha=1.):
        "x, y: bottom left coordinates of image"
        self.x = float(x)
        self.y = float(y)
        self.alpha = float(alpha)

    def get_surf_cairo(self):
        return None

    def process_cairo(self, cxt):
        surf = self.get_surf_cairo()
        x = self.x*SCALE_CM_TO_POINT
        y = self.y*SCALE_CM_TO_POINT
        height = surf.get_height()
        cxt.save()
        cxt.set_source_surface(surf, x, -y-height)
        cxt.paint_with_alpha(self.alpha)
        cxt.restore()



class PNGImage(Image):
    def __init__(self, name, x=0, y=0, alpha=1.):
        "x, y: bottom left coordinates of image"
        Image.__init__(self, x, y, alpha)
        self.name = name

    surf = None
    def get_surf_cairo(self):
        import cairo
        if self.surf is not None:
            return self.surf
        try:
            surf = cairo.ImageSurface.create_from_png(self.name)
        except cairo.Error as e:
            print("ciaro.Error: %s, self.name=%r"%(e, self.name))
            raise
        self.surf = surf
        return self.surf




class NumpyImage(Image):
    def __init__(self, source, x=0, y=0, alpha=1.):
        "x, y: bottom left coordinates of image"
        import numpy
        assert isinstance(source, numpy.ndarray)
        assert len(source.shape) == 3 # (height, width, 4)
        assert source.shape[2] == 4 # (blue, green, red, alpha)
        Image.__init__(self, x, y, alpha)
        self.source = source.copy()

    surf = None
    def get_surf_cairo(self):
        import cairo
        if self.surf is not None:
            return self.surf
        import cairo
        source = self.source
        height, width = source.shape[:2]
        surf = cairo.ImageSurface.create_for_data(
            source, cairo.FORMAT_ARGB32, width, height)
        self.surf = surf
        return self.surf



# ----------------------------------------------------------------------------
# 
#


def text_extents_cairo(text):
    import cairo
    #surface = cairo.PDFSurface("/dev/null", 0, 0)
    # only in cairo 1.11.0:
    surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    cxt = cairo.Context(surface)
    #cxt.move_to(0., 0.)
    #cxt.show_text(text)
    #extents = surface.ink_extents()
    #(ulx, uly, width, height) = extents
    #print("get_bound", extents) # same as text_extents ...
    ex = cxt.text_extents(text)
    return ex


class Text(object):
    def __new__(cls, *args, **kw):
        ob = object.__new__(the_text_cls) # pay attention here !
        return ob

    def text_extents(self):
        bound = self.get_bound()
        llx, lly, urx, ury = bound
        return (0., ury, urx-llx, ury-lly) # 0., ury, width, height



class CairoText(Item, Text):
    def __init__(self, x, y, text, color=None, size=None, **kw):
        self.x = x
        self.y = y
        self.text = text
        self.color = color

    def get_bound(self):
        extents = text_extents_cairo(self.text)
        (dx, dy, width, height, x1, y1) = extents
        #print("CairoText.get_bound", repr(self.text), dx, dy, width, height, x1, y1)
        #assert dx>=0, dx
        x, y = self.x*SCALE_CM_TO_POINT, self.y*SCALE_CM_TO_POINT
        llx, lly = x, y-dy-height
        urx, ury = llx+dx+width, lly+height
        b = Bound(llx/SCALE_CM_TO_POINT, lly/SCALE_CM_TO_POINT, urx/SCALE_CM_TO_POINT, ury/SCALE_CM_TO_POINT)
        #print("CairoText.get_bound", b)
        return b

    def process_cairo(self, cxt):
        cxt.save()
        if self.color:
            cxt.set_source_rgba(*self.color)
        x = self.x*SCALE_CM_TO_POINT
        y = self.y*SCALE_CM_TO_POINT
        cxt.move_to(x, -y)
        cxt.show_text(self.text)
        #cxt.set_font_size(10.)
        cxt.restore()


class CleanupMkText(Visitor):

    def __init__(self):
        self.flag = False

    def on_visit(self, item):
        cls = item.__class__
        #print(cls.__name__, end="")
        if cls is ClosePath:
            self.flag = True
        elif cls is Compound:
            if self.flag:
                self.flag = False
            else:
                item = None
        #print("*" if item is not None else "")
        return item


class MkText(Compound, Text):

    tex_engine = "pdftex"

    _baseline = None
    @classmethod
    def _get_baseline(cls):
        if cls._baseline is None:
            item = make_text("x", cls.tex_engine) # measure this font using "x"
            bound = item.get_bound()
            #print("_get_baseline", bound)
            cls._baseline = bound.lly
        return cls._baseline

    def __init__(self, x, y, text, color=None, size=None, **kw):
        assert text
        self.x = x # = SCALE_CM_TO_POINT*x
        self.y = y # = SCALE_CM_TO_POINT*y
        self.text = text
        item = make_text(text, self.tex_engine)
        if color is not None:
            black = RGBA(0., 0., 0., 1.)
            item.rewrite(Replacer(black, color))
        bound = item.get_bound()
        assert not bound.is_empty(), bound
        llx, lly = bound.llx, bound.lly
        #print("Text.__init__", x, y, bound)
        y0 = self._get_baseline()
        self.bot = y0 - bound.lly
        trafo = Translate(x-llx, y-lly-self.bot)
        #print("Text.__init__ trafo:", trafo)
        self.bound = Bound(x, y-self.bot, x+bound.width, y+bound.height-self.bot)
        items = list(item.items)
        items = [trafo] + items
        Compound.__init__(self, items)

    def get_bound(self):
        return self.bound

    def cleanup(self):
        self.rewrite(CleanupMkText())


the_text_cls = CairoText

#def Text(*args, **kw):
#    return the_text_cls(*args, **kw)



# ----------------------------------------------------------------------------
# Some good code copied from PyX
#


def arc_to_curve(x, y, r, angle1, angle2):
    dangle = angle2-angle1

    if dangle==0:
        return None

    x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
    x3, y3 = x+r*cos(angle2), y+r*sin(angle2)

    l = r*4*(1-cos(dangle/2))/(3*sin(dangle/2))

    x1, y1 = x0-l*sin(angle1), y0+l*cos(angle1)
    x2, y2 = x3+l*sin(angle2), y3-l*cos(angle2)

    items = [
        MoveTo(x0, y0),  # changed from LineTo on 2022/02/02 ... good?
        CurveTo(x1, y1, x2, y2, x3, y3)]
    return items


def _arc_to_bezier(x, y, r, angle1, angle2, relative=True):
    # same as arc_to_curve, but with LineTo instead of MoveTo 
    dangle = angle2-angle1

    if dangle==0:
        return None

    x0, y0 = x+r*cos(angle1), y+r*sin(angle1)
    x3, y3 = x+r*cos(angle2), y+r*sin(angle2)

    l = r*4*(1-cos(dangle/2))/(3*sin(dangle/2))

    x1, y1 = x0-l*sin(angle1), y0+l*cos(angle1)
    x2, y2 = x3+l*sin(angle2), y3-l*cos(angle2)

    items = [LineTo(x0, y0)] if relative else [MoveTo(x0, y0)]
    items.append(CurveTo(x1, y1, x2, y2, x3, y3))
    return items


def arc_to_bezier(x, y, r, angle1, angle2, danglemax=0.5*pi, relative=True):
    if angle2<angle1:
        angle2 = angle2 + (floor((angle1-angle2)/(2*pi))+1)*2*pi
    elif angle2>angle1+2*pi:
        angle2 = angle2 - (floor((angle2-angle1)/(2*pi))-1)*2*pi

    if r == 0 or angle1-angle2 == 0:
        return []

    subdivisions = int((angle2-angle1)/danglemax)+1

    dangle = (angle2-angle1)/subdivisions

    items = []
    for i in range(subdivisions):
        items += _arc_to_bezier(x, y, r, angle1+i*dangle, angle1+(i+1)*dangle, relative)
        relative = True

    p = Path(items)
    return p


# ----------------------------------------------------------------------------
# 
#

class Polygon(Item):
    def __init__(self, pts, fill=None, stroke=None, lw=None,
            texture=None, texture_coords=[(0., 0.), (1., 0.), (0., 1.)]):
        Item.__init__(self)
        assert len(pts)>1
        self.pts = [(x*SCALE_CM_TO_POINT, y*SCALE_CM_TO_POINT) for (x, y) in pts]
        self.fill = fill
        self.stroke = stroke
        if lw is not None:
            lw = lw*SCALE_CM_TO_POINT
        self.lw = lw
        assert texture is None or isinstance(texture, Item)
        self.texture = texture
        if texture is not None:
            pts = [(x*SCALE_CM_TO_POINT, y*SCALE_CM_TO_POINT) for (x, y) in texture_coords]
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]

            import cairo
            m = cairo.Matrix((x1-x0), (y0-y1), (x0-x2), (y2-y0), x0, -y0)
            m.invert()
            self.texm = m

        #self.texture_coords = texture_coords

    def process_cairo(self, cxt):
        pts = self.pts
        cxt.save() # <--------- save

        fill = self.fill #or (0., 0., 0., 1.)
        if fill is not None:
            cxt.set_source_rgba(*fill)
            x, y = pts[0]
            cxt.move_to(x, -y)
            for (x, y) in pts[1:]:
                cxt.line_to(x, -y)
            cxt.close_path()
            cxt.fill()
    
        #cxt.set_line_width(4.0)
        if self.lw is not None:
            cxt.set_line_width(self.lw)

        stroke = self.stroke # or (1., 1., 1., 1.)
        if stroke is not None:
            cxt.set_source_rgba(*stroke)
            x, y = pts[0]
            cxt.move_to(x, -y)
            for (x, y) in pts[1:]:
                cxt.line_to(x, -y)
            cxt.close_path()
            cxt.clip_preserve()
            cxt.stroke()

            #cxt.set_source_rgba(0., 0., 1., 1.)
            #x, y = pts[0]
            #r = 20.0
            #cxt.arc(x, -y, r, 0, 2*pi)
            #cxt.fill()

        import cairo

        texture = self.texture
        if texture is not None:
            x, y = pts[0]
            cxt.move_to(x, -y)
            for (x, y) in pts[1:]:
                cxt.line_to(x, -y)
            cxt.close_path()
            cxt.clip()

            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]

            #rs = 1./SCALE_CM_TO_POINT
            rs = 1.
            #cxt.transform(cairo.Matrix(x1-x0, y0-y1, x0-x2, y2-y0, x0, -y0))
            #cxt.scale(rs, rs)

            cxt.transform(cairo.Matrix(rs*(x1-x0), rs*(y0-y1), rs*(x0-x2), rs*(y2-y0), x0, -y0))
            cxt.transform(self.texm)

            texture.process_cairo(cxt)

        cxt.restore() # <------- restore


class Polymesh(Item):
    def __init__(self, pts, fills):
        Item.__init__(self)
        assert len(pts)>=3
        assert len(pts) == len(fills)
        self.pts = [(x*SCALE_CM_TO_POINT, y*SCALE_CM_TO_POINT) for (x, y) in pts]
        self.fills = fills
        assert len(pts)<=4, len(pts) # this appears to be a limitation of cairo...

    def process_cairo(self, cxt):
        import cairo
        pts = self.pts
        fills = self.fills
        cxt.save() # <--------- save

        pts = list(self.pts)
        if len(pts)==3:
            pts.append(pts[-1])
            fills.append(fills[-1])
        assert len(pts)==4, len(pts) # this appears to be a limitation of cairo...
        m = cairo.MeshPattern()
        m.begin_patch()
        x, y = pts[0]
        m.move_to(x, -y)
        for (x,y) in pts[1:]:
            m.line_to(x, -y)
        for i in range(4):
            m.set_corner_color_rgba(i, *fills[i])
        m.end_patch()

        cxt.set_source(m)
        x, y = pts[0]
        cxt.move_to(x, -y)
        for (x, y) in pts[1:]:
            cxt.line_to(x, -y)
        cxt.close_path()
        cxt.fill()

        cxt.restore() # <------- restore


class Gradient(Item):
    def __init__(self, x, y, radius, rgb0=None, rgb1=None):
        Item.__init__(self)
        self.x = x
        self.y = y
        self.radius = radius
        self.rgb0 = rgb0
        self.rgb1 = rgb1

    def process_cairo(self, cxt):
        import cairo
        x, y, radius = self.x, -self.y, self.radius
        x = x*SCALE_CM_TO_POINT
        y = y*SCALE_CM_TO_POINT
        radius = radius*SCALE_CM_TO_POINT
        cxt.save()
        cx0, cy0 = cx1, cy1 = x, y
        radius0 = 0.2*radius
        radius1 = 1.0*radius
        p = cairo.RadialGradient(cx0, cy0, radius0, cx1, cy1, radius1)
        if self.rgb0 is not None:
            p.add_color_stop_rgba(0, *self.rgb0)
        else:
            p.add_color_stop_rgba(0, 0.9, 0.9, 0.9, 1)
        if self.rgb1 is not None:
            p.add_color_stop_rgba(1, *self.rgb1)
        else:
            p.add_color_stop_rgba(1, 0.6, 0.6, 0.6, 1.)
        cxt.set_source(p)
        cxt.arc(x, y, radius, 0., pi*2)
        cxt.fill()
        cxt.restore()


class Ball(Item):
    def __init__(self, x, y, radius, rgb0=None, rgb1=None):
        Item.__init__(self)
        self.x = x*SCALE_CM_TO_POINT
        self.y = y*SCALE_CM_TO_POINT
        self.radius = radius*SCALE_CM_TO_POINT
        self.rgb0 = rgb0
        self.rgb1 = rgb1

    def process_cairo(self, cxt):
        import cairo
        x, y, radius = self.x, -self.y, self.radius
        cxt.save()
        cxt.set_line_width(0.5)
        cxt.arc(x, y, radius, 0., pi*2)
        cxt.stroke()
        cx0, cy0 = cx1, cy1 = x+0.8*radius, y+0.8*radius
        radius0 = 0.2*radius
        radius1 = 2.0*radius
        p = cairo.RadialGradient(cx0, cy0, radius0, cx1, cy1, radius1)
        if self.rgb0 is not None:
            p.add_color_stop_rgba(0, *self.rgb0)
        else:
            p.add_color_stop_rgba(0, 0.9, 0.9, 0.9, 1)
        if self.rgb1 is not None:
            p.add_color_stop_rgba(1, *self.rgb1)
        else:
            p.add_color_stop_rgba(1, 0.6, 0.6, 0.6, 1.)
        cxt.set_source(p)
        cxt.arc(x, y, radius, 0., pi*2)
        cxt.fill()
        cxt.restore()


# ----------------------------------------------------------------------------
#
#


def test():
    pass


if __name__ == "__main__":
    test()


