#!/usr/bin/env python3

from random import random, choice, seed
from copy import deepcopy
from math import pi, sqrt


from huygens.sat import Expr, System, Listener
from huygens.front import RGBA, Compound, Translate, Deco, Path, Canvas
from huygens.front import path, style, canvas, color
from huygens.turtle import Turtle
from huygens.argv import argv


class Magic(Listener):

    did_layout = False

    def __hash__(self):
        return id(self)

    # Are we like a dict, or a list ??
    # Not sure...

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError



EPSILON = 1e-6


class Box(Magic):

    DEBUG = False

    @classmethod
    def promote(cls, item, align=None, margin=None, xmargin=None, ymargin=None):
        if isinstance(item, Box):
            box = item
        elif isinstance(item, str):
            box = TextBox(item)
        elif isinstance(item, (tuple, list)):
            box = HBox(item)
        elif item is None:
            box = EmptyBox()
        elif isinstance(item, Canvas):
            box = CanBox(item)
        else:
            raise TypeError(repr(item))
        if align is not None:
            box = AlignBox(box, align)
        xmargin = xmargin if xmargin is not None else margin
        ymargin = ymargin if ymargin is not None else xmargin
        if xmargin is not None:
            box = MarginBox(box, xmargin, ymargin)
        return box

    @property
    def width(self):
        return self.left + self.right

    @property
    def height(self):
        return self.top + self.bot

    @property
    def llx(self):
        return self.x - self.left

    @property
    def lly(self):
        return self.y - self.bot

    @property
    def ll(self):
        return self.llx, self.lly

    @property
    def urx(self):
        return self.x + self.right

    @property
    def ury(self):
        return self.y + self.top

    @property
    def ur(self):
        return self.urx, self.ury

    @property
    def midx(self):
        return 0.5*(self.llx + self.urx)

    @property
    def midy(self):
        return 0.5*(self.lly + self.ury)

    @property
    def mid(self):
        return (self.midx, self.midy)

    @property
    def bound(self):
        return self.llx, self.lly, self.urx, self.ury

    def get_align(self, align):
        llx, lly, urx, ury = self.bound
        midx = 0.5*(llx + urx)
        midy = 0.5*(lly + ury)
        if align == "center":
            x, y = midx, midy
        elif align == "north":
            x, y = midx, ury
        elif align == "south":
            x, y = midx, lly
        elif align == "east":
            x, y = urx, midy
        elif align == "west":
            x, y = llx, midy
        elif align == "northeast":
            x, y = urx, ury
        elif align == "northwest":
            x, y = llx, ury
        elif align == "southeast":
            x, y = urx, lly
        elif align == "southwest":
            x, y = llx, lly
        else:
            assert 0, "alignment %r not understood" % align
    
        return x, y

    def contain(self, x, y, system, weight=None):
        system.add(x <= self.urx, weight)
        system.add(x >= self.llx, weight)
        system.add(y <= self.ury, weight)
        system.add(y >= self.lly, weight)

    def on_layout(self, cvs, system):
        #assert not self.did_layout, "already called on_layout"
        assert not self in system.memo, "duplicate box %s"%(self,)
        system.memo.add(self)
        if self.DEBUG:
            print("%s.on_layout" % (self.__class__.__name__,))
        for attr in 'x y left right top bot'.split():
            stem = self.__class__.__name__ + '.' + attr
            expr = getattr(self, attr, None)
            if isinstance(expr, Expr):
                system.listen_expr(self, attr, expr)

            elif attr in self.__dict__:
                pass

            else:
                # We don't try to minimize the absolute coordinate values.
                weight = 1.0 if attr not in 'xy' else 0.0
                vmin = None if attr in 'xy' else 0.
                #v = system.get_var(stem, weight, vmin=vmin)
                v = system.listen_var(self, attr, stem, weight, vmin=vmin)
                setattr(self, attr, v)

        self.did_layout = True

    def assign_variables(self, system):
        # set all our Variable attributes to actual solutions
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            value = getattr(self, attr)
            if not isinstance(value, Expr):
                continue
            value = system[value]
            setattr(self, attr, value)

    def on_render(self, cvs, system):
        if 1:
            x = self.x
            y = self.y
            left = self.left
            right = self.right
            top = self.top
            bot = self.bot
        elif 0:
            x = system[self.x]
            y = system[self.y]
            left = system[self.left]
            right = system[self.right]
            top = system[self.top]
            bot = system[self.bot]

        else:
            self.assign_variables(system)
            x = self.x
            y = self.y
            left = self.left
            right = self.right
            top = self.top
            bot = self.bot

        if not self.DEBUG:
            return
        assert type(x) is float, str(self)
        #cvs.set_line_width(0.5)
        st = [color.rgba(1., 0., 0., 0.5), style.linestyle.solid]
        r = 0.1
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), st)
        cvs.stroke(path.line(x+r, y-r, x-r, y+r), st)
        #bg = RGBA(0.5*random(), 0.5*random(), 0.5*random(), 0.5)
        bg = RGBA(0.5, 0.5, 0., 0.1)
        cvs.fill(path.rect(x-left, y-bot, left+right, top+bot), [bg])
        cvs.stroke(path.rect(x-left, y-bot, left+right, top+bot), st)

    system = None
    def layout(self, cvs, x=0, y=0):
        if self.system is None:
            self.system = System()
        system = self.system
        system.memo = set() # hang this here
        self.on_layout(cvs, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        return system

    def render(self, cvs=None, x=0, y=0):
        if cvs is None:
            cvs = canvas.canvas()
        self.layout(cvs, x, y)
        self.on_render(cvs, self.system)
        self.system.refresh()
        return cvs

    def __add__(self, other):
        other = Box.promote(other)
        return OBox([self, other])


class BoxDeco(Deco):
    def __init__(self, box, t, *args, **kw):
        "decorate a path at time t with a Box"
        box = Box.promote(box, *args, **kw)
        self.box = box
        self.t = t

    def on_decorate(self, pre, path, post):
        assert isinstance(path, Path)
        cvs = canvas.canvas()
        x, y, dx, dy = path.tangent(self.t)
        self.box.render(cvs, x, y)
        post += cvs


#class EmptyBox(Box):
#    def __init__(self, top=0., bot=0., left=0., right=0.):
#        self.top = top
#        self.bot = bot
#        self.left = left
#        self.right = right


class EmptyBox(Box):
    def __init__(self, top=None, bot=None, left=None, right=None):
        if top is not None:
            self.top = top
        if bot is not None:
            self.bot = bot
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right


class HSpaceBox(Box):
    def __init__(self, space):
        self.left = 0
        self.right = space

class VSpaceBox(Box):
    def __init__(self, space):
        self.top = 0
        self.bot = space


class MinBox(Box):
    def __init__(self, min_top=0., min_bot=0., min_left=0., min_right=0.):
        self.min_top = min_top
        self.min_bot = min_bot
        self.min_left = min_left
        self.min_right = min_right

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        system.add(self.top >= self.min_top)
        system.add(self.bot >= self.min_bot)
        system.add(self.left >= self.min_left)
        system.add(self.right >= self.min_right)


class CanBox(Box):
    def __init__(self, cvs):
        bound = cvs.get_bound_cairo()
        bound = bound.scale_point_to_cm()
        self.top = 0.
        self.left = 0.
        self.right = bound.width
        self.bot = bound.height
        x0, y0 = bound.llx, bound.ury # this becomes our top-left
        self.x0 = x0
        self.y0 = y0
        self.cvs = cvs

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        x0, y0 = self.x0, self.y0
        dx, dy = x-x0, y-y0
        item = Compound([Translate(dx, dy), self.cvs])
        cvs.append(item)

    
class StrokeBox(Box):
    def __init__(self, width, height, rgba=(0., 0., 0., 1.)):
        self.top = height
        self.bot = 0.
        self.left = 0.
        self.right = width
        self.rgba = rgba

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        cvs.stroke(path.rect(x, y, self.width, self.height), [RGBA(*self.rgba)])


class FillBox(StrokeBox):
    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        cvs.fill(path.rect(x, y, self.width, self.height), [RGBA(*self.rgba)])


class TextBox(Box):
    def __init__(self, text, weight=99.0):
        self.text = text
        # Use weight higher than the default weight of 1.0.
        self.weight = weight

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        extents = cvs.text_extents(self.text)
        dx, dy, width, height = extents
        system.add(self.left + self.right == width+dx)
        system.add(self.top + self.bot == height, self.weight)
        system.add(self.left == 0)
        assert dy >= 0., dy
        system.add(self.top == dy)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        cvs.text(x, y, self.text)


class ChildBox(Box):
    "Has one child box"
    def __init__(self, child, *args, **kw):
        self.child = Box.promote(child, *args, **kw)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self.child][idx]

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        self.child.on_render(cvs, system)


class MarginBox(ChildBox):
    def __init__(self, child, xmargin, ymargin=None):
        ChildBox.__init__(self, child)
        ymargin = xmargin if ymargin is None else ymargin
        self.xmargin = xmargin
        self.ymargin = ymargin

    def on_layout(self, cvs, system):
        child = self.child
        child.on_layout(cvs, system)
        # inherit the child anchor
        self.x = child.x
        self.y = child.y
        xmargin = self.xmargin
        ymargin = self.ymargin
        self.left = child.left + xmargin
        self.right = child.right + xmargin
        self.top = child.top + ymargin
        self.bot = child.bot + ymargin
        Box.on_layout(self, cvs, system)


class RectBox(MarginBox):
    def __init__(self, child, margin=0, bg=None):
        MarginBox.__init__(self, child, margin)
        self.bg = bg

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        bg = self.bg
        if bg is not None:
            llx = system[self.llx]
            lly = system[self.lly]
            width = system[self.width]
            height = system[self.height]
            cvs.fill(path.rect(llx, lly, width, height), [bg])
        self.child.on_render(cvs, system)


class AlignBox(ChildBox):
    def __init__(self, child, align):
        ChildBox.__init__(self, child)
        self.align = align
        assert align in ("center north south east west northeast "+
            "northwest southeast southwest").split(), "align %r not understood"%align

    def on_layout(self, cvs, system):
        child = self.child
        child.on_layout(cvs, system)
        x, y = child.get_align(self.align)
        self.x = x
        self.y = y
        self.left = x - child.llx
        self.right = child.urx - x
        self.bot = y - child.lly
        self.top = child.ury - y
        Box.on_layout(self, cvs, system)


class SlackBox(ChildBox):
    def __init__(self, child, *args, **kw):
        ChildBox.__init__(self, child, *args, **kw)

    def on_layout(self, cvs, system):
        child = self.child
        child.on_layout(cvs, system)
        # child anchor for self
        self.x = child.x
        self.y = child.y
        Box.on_layout(self, cvs, system)
        system.add(self.top >= child.top)
        system.add(self.bot >= child.bot)
        system.add(self.left >= child.left)
        system.add(self.right >= child.right)


class CompoundBox(Box):
    def __init__(self, boxs, weight=None, align=None):
        assert len(boxs)
        assert type(boxs) is list
        self.boxs = [Box.promote(box, align) for box in boxs]
        self.weight = weight

    def __len__(self):
        return len(self.boxs)

    def __getitem__(self, idx):
        return self.boxs[idx]

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        for box in self.boxs:
            box.on_layout(cvs, system)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        for box in self.boxs:
            box.on_render(cvs, system)

# FAIL:
#
#class MultiBox(CompoundBox):
#    def on_layout(self, cvs, system):
#        #Box.on_layout(self, cvs, system)
#        for box in self.boxs:
#            box.on_layout(cvs, system)
#
#    def on_render(self, cvs, system):
#        # Don't call Box.on_render because I have no shape
#        for box in self.boxs:
#            box.on_render(cvs, system)
#


class MasterBox(CompoundBox):
    def __init__(self, box, boxs, weight=None, align=None):
        CompoundBox.__init__(self, [box] + boxs, weight, align)
        self.box = box # master box
    
    def on_layout(self, cvs, system):
        box = self.box
        box.on_layout(cvs, system)
        # inherit layout from the master box:
        self.x = box.x
        self.y = box.y
        self.left = box.left
        self.right = box.right
        self.top = box.top
        self.bot = box.bot
        Box.on_layout(self, cvs, system)
        for box in self.boxs[1:]:
            box.on_layout(cvs, system)



class OBox(CompoundBox):
    "Overlay boxes on top of each other, with matching anchors"
    strict = False
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(self.y == box.y) # align
            if self.strict:
                system.add(box.left == self.left, self.weight)
                system.add(box.right == self.right, self.weight)
                system.add(box.top == self.top, self.weight)
                system.add(box.bot == self.bot, self.weight)
            else:
                system.add(box.left <= self.left)
                system.add(box.right <= self.right)
                system.add(box.top <= self.top)
                system.add(box.bot <= self.bot)


class HBox(CompoundBox):
    "horizontal compound box: anchor left, layout Box's left to right."
    strict = False
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.left == 0.) # left anchor
        left = self.x
        for box in boxs:
            system.add(self.y == box.y) # align
            system.add(box.x - box.left == left)
            left += box.width
            if self.strict:
                system.add(box.top == self.top, self.weight)
                system.add(box.bot == self.bot, self.weight)
            else:
                system.add(box.top <= self.top)
                system.add(box.bot <= self.bot)
        system.add(self.x + self.width == left)


class StrictHBox(HBox):
    strict = True


class VBox(CompoundBox):
    "vertical compound box: anchor top, layout Box's top down."
    strict = False
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.top == 0.) # top anchor
        y = self.y
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(box.y + box.top == y)
            y -= box.height
            if self.strict:
                system.add(box.left == self.left, self.weight)
                system.add(box.right == self.right, self.weight)
            else:
                system.add(box.left <= self.left)
                system.add(box.right <= self.right)
        system.add(self.y - self.bot == y)


class StrictVBox(VBox):
    strict = True


class TableBox(CompoundBox):
    def __init__(self, rows, hspace=0., vspace=0., grid=False, attrs=[], **kw):
        assert len(rows), "no rows"
        assert len(rows[0]), "no cols"

        rows = [[Box.promote(item, **kw) for item in row] for row in rows]
        assert hspace >= 0.
        assert vspace >= 0.

#       too tricky...
#            # Get original shape
#            m = len(rows) # rows
#            n = len(rows[0]) # cols
#    
#            row = []
#            for i in range(n):
#                space = hspace if i+1<n else 0.
#                row.append(MinBox(0., 0., 0., space))
#            row.append(MinBox(0., 0., 0., 0.))
#            rows.append(row)
#    
#            for i in range(m):
#                space = vspace if i+1<m else 0.
#                rows[i].append(MinBox(0., space, 0., 0.))

        self.hspace = hspace
        self.vspace = vspace
    
        # Get new shape
        m = len(rows) # rows
        n = len(rows[0]) # cols

        boxs = []
        for row in rows:
            assert len(row) == n
            for box in row:
                box = Box.promote(box)
                boxs.append(box)

        self.rows = rows
        self.shape = m, n
        # anchor is top left
        self.top = 0.
        self.left = 0.
        self.grid = grid
        CompoundBox.__init__(self, boxs)

    def __getitem__(self, key):
        assert type(key) is tuple
        assert len(key) == 2
        row, col = key
        return self.rows[row][col]

    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        m, n = self.shape
        rows = self.rows
        xs, ys = {}, {}
        ws, hs = {}, {} # width's, height's
        for i in range(m): # rows
            ys[i] = system.get_var("TableBox.row(%d)"%i, weight=0.)
            hs[i] = system.get_var("TableBox.height(%d)"%i, weight=1.) # minimize
        for j in range(n): # cols
            xs[j] = system.get_var("TableBox.col(%d)"%j, weight=0.)
            ws[j] = system.get_var("TableBox.width(%d)"%j, weight=1.) # minimize
        for i in range(m): # rows
            for j in range(n): # cols
                box = rows[i][j]
                system.add(box.y == ys[i]) # align
                system.add(box.x == xs[j]) # align

        hspace = self.hspace
        for i in range(m): # rows
            x = self.x
            for j in range(n): # cols
                box = rows[i][j]
                system.add(box.x - box.left >= x)
                width = ws[j] # width of this col
                x += width
                system.add(box.x + box.right + hspace <= x)
            system.add(self.x + self.width >= x)

        vspace = self.vspace
        for j in range(n): # cols
            y = self.y
            for i in range(m): # rows
                box = rows[i][j]
                system.add(box.y + box.top <= y)
                height = hs[i]
                y -= height
                system.add(box.y - box.bot - vspace >= y)
            system.add(self.y - self.height <= y)
        self.vs = xs, ys, ws, hs

    def on_render(self, cvs, system):
        CompoundBox.on_render(self, cvs, system)
        if not self.grid:
            return
        m, n = self.shape
        xs, ys, ws, hs = self.vs
        width = system[self.width]
        height = system[self.height]
        x = system[self.x]
        y = system[self.y]
        for j in range(n):
            cvs.stroke(path.line(x, y, x, y-height))
            x += system[ws[j]]
        #cvs.stroke(path.line(x, y, x, y-height))
        x = system[self.x]
        y = system[self.y]
        for i in range(m):
            cvs.stroke(path.line(x, y, x+width, y))
            y -= system[hs[i]]
        x = system[self.x]
        y = system[self.y]
        cvs.stroke(path.rect(x, y-height, width, height))


class PathBox(Box):

    def __init__(self, src, tgt, attrs=[], xweight=0.1, yweight=0.1):
        "stroke a path from src Box to tgt Box"
        assert isinstance(src, Box)
        assert isinstance(tgt, Box)
        self.src = src
        self.tgt = tgt
        assert src is not tgt, "self-arrow not implemented"
        self.attrs = attrs
        self.xweight = xweight
        self.yweight = yweight

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        self.x1 = x1 = system.get_var("PathBox.x1", weight=0.)
        self.y1 = y1 = system.get_var("PathBox.y1", weight=0.)
        x0, y0 = self.x, self.y 
        src = self.src
        tgt = self.tgt
        src.contain(x0, y0, system)
        tgt.contain(x1, y1, system)
        self.contain(x1, y1, system)

        # With a lower weight, try to stay in the middle
        xweight = self.xweight
        yweight = self.yweight
        system.add(x0 == src.midx, xweight)
        system.add(y0 == src.midy, yweight)
        system.add(x1 == tgt.midx, xweight)
        system.add(y1 == tgt.midy, yweight)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x0 = system[self.x]
        y0 = system[self.y]
        x1 = system[self.x1]
        y1 = system[self.y1]
        cvs.stroke(path.line(x0, y0, x1, y1), self.attrs)



class ArrowBox(PathBox):

    #default_astyle = "flat"
    default_astyle = "curve"
    default_size = 0.15
    default_attrs = [style.linewidth.thin, style.linecap.round,
        style.linejoin.round]

    def __init__(self, src, tgt, label=None, label_align=None,
            astyle=None, size=None, pad_head=0.04, pad_tail=0.04,
            attrs=None, xweight=0.1, yweight=0.1):
        assert isinstance(src, Box)
        assert isinstance(tgt, Box)
        self.src = src
        self.tgt = tgt
        assert src is not tgt, "self-arrow not implemented"
        #if label is not None:
        #    label = Box.promote(label)
        self.label = label
        self.label_align = label_align
        if astyle is None:
            astyle = ArrowBox.default_astyle
        self.astyle = astyle
        if size is None:
            size = ArrowBox.default_size
        self.size = size
        if attrs is None:
            attrs = ArrowBox.default_attrs
        self.pad_head = pad_head
        self.pad_tail = pad_tail
        PathBox.__init__(self, src, tgt, attrs, xweight, yweight)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)

        pad_head = self.pad_head
        pad_tail = self.pad_tail

        x0 = system[self.x]
        y0 = system[self.y]
        x1 = system[self.x1]
        y1 = system[self.y1]

        r = sqrt((x1-x0)**2 + (y1-y0)**2)

        turtle = Turtle(x0, y0)
        turtle.lookat(x1, y1)
        turtle.penup()
        turtle.fwd(pad_tail)
        turtle.pendown()
        turtle.fwd(r-pad_head-pad_tail)
        turtle.arrow(size=self.size, astyle=self.astyle)
        turtle.stroke(attrs=self.attrs, cvs=cvs)

        label = self.label
        align = self.label_align
        if label is None:
            return

        dx = x1-x0
        dy = y1-y0
        PIP = 0.04 # <------- MAGIC CONSTANT TODO
        if align is not None:
            pass
        elif abs(dy) < 0.1: # another MAGIC CONSTANT
            align = "south"
        elif abs(dx) < 0.1: # another MAGIC CONSTANT
            align = "west"
            PIP = 0.08
        elif dx*dy > 0.:
            align = "northwest"
            PIP = 0.04
        else:
            align = "southwest"
            PIP = 0.02

        #print("PIP", PIP, align, label, abs(dx), abs(dy))
        if isinstance(label, Box):
            pass
        else:
            label = MarginBox(label, PIP)
            label = AlignBox(label, align)

        # TODO pad_head, pad_tail XXX
        x = 0.5*(x0+x1)
        y = 0.5*(y0+y1)
        label.render(cvs, x, y)
        #cvs.stroke(path.circle(x, y, 0.02), [color.rgb.red])


def test_arrows(): # XXX move to doc

    from huygens import config
    config(text="pdftex")

    Box.DEBUG = False

    tbox = lambda t: MarginBox(TextBox(t), 0.05)
    rows = [
        [r"A", r"B", r"C"],
        [r"D", r"E", r"F"],
        [r"G", r"H", r"I"],
    ]
    boxs = [[tbox("$%s$"%c) for c in row] for row in rows]

    arrows = []
    for di in [-1, 0, 1]:
      for dj in [-1, 0, 1]:
        if di==0 and dj==0:
            continue
        label = r"$x$"
        a = ArrowBox(boxs[1][1], boxs[1+di][1+dj], label=label)
        arrows.append(a)

    r = 1.1
    table = TableBox(boxs, hspace=r, vspace=0.8*r)
    box = MasterBox(table, arrows)

    cvs = canvas.canvas()
    box.render(cvs)

    cvs.writePDFfile("output.pdf")



if __name__ == "__main__":

    test_arrows()

    print("OK\n")



