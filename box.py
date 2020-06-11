#!/usr/bin/env python3

from random import random, choice, seed
from copy import deepcopy
from math import pi


from bruhat.render.sat import Expr, Variable, System
from bruhat.render.front import RGBA, Canvas, Scale, Compound, Translate
from bruhat.render.front import path, style
from bruhat.argv import argv


class Magic(object):

    did_layout = False

    def __hash__(self):
        return id(self)

    # Are we like a dict, or a list ??
    # Not sure...

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError




class Box(Magic):

    DEBUG = False

    @classmethod
    def promote(cls, item, align=None):
        if isinstance(item, Box):
            box = item
        elif isinstance(item, str):
            box = TextBox(item)
        elif isinstance(item, (tuple, list)):
            box = HBox(item)
        elif item is None:
            box = EmptyBox()
        else:
            raise TypeError(repr(item))
        if align is not None:
            box = AlignBox(box, align)
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
    def bound(self):
        return self.llx, self.lly, self.urx, self.ury

    def get_align(self, align):
        llx, lly, urx, ury = self.bound
        xmid = 0.5*(llx + urx)
        ymid = 0.5*(lly + ury)
        if align == "center":
            x, y = xmid, ymid
        elif align == "north":
            x, y = xmid, ury
        elif align == "south":
            x, y = xmid, lly
        elif align == "east":
            x, y = urx, ymid
        elif align == "west":
            x, y = llx, ymid
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

    def on_layout(self, cvs, system):
        #assert not self.did_layout, "already called on_layout"
        assert not self in system.memo, "duplicate box %s"%(self,)
        system.memo.add(self)
        if self.DEBUG:
            print("%s.on_layout" % (self.__class__.__name__,))
        for attr in 'x y left right top bot'.split():
            if attr in self.__dict__:
                continue
            stem = self.__class__.__name__ + '.' + attr

            # We don't try to minimize the absolute coordinate values.
            weight = 1.0 if attr not in 'xy' else 0.0
            vmin = None if attr in 'xy' else 0.
            v = system.get_var(stem, weight, vmin=vmin)
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
        #cvs.set_line_width(0.5)
        cl = RGBA(1., 0., 0., 0.5)
        r = 0.1
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), [cl]) #, style.linewidth.Thick])
        cvs.stroke(path.line(x+r, y-r, x-r, y+r), [cl])
        #bg = RGBA(0.5*random(), 0.5*random(), 0.5*random(), 0.5)
        bg = RGBA(0.5, 0.5, 0., 0.1)
        cvs.fill(path.rect(x-left, y-bot, left+right, top+bot), [bg])
        cvs.stroke(path.rect(x-left, y-bot, left+right, top+bot), [cl])

    def layout(self, cvs, x=0, y=0):
        system = System()
        system.memo = set() # hang this here
        self.on_layout(cvs, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        self.system = system
        return system

    def render(self, cvs, x=0, y=0):
        #save = Box.DEBUG
        #if debug is not None:
        #    Box.DEBUG = debug
        if not self.did_layout:
            self.layout(cvs, x, y)
        self.on_render(cvs, self.system)
        #Box.DEBUG = save


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
    def __init__(self, child):
        self.child = Box.promote(child)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self.child][idx]

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        self.child.on_render(cvs, system)


class MarginBox(ChildBox):
    def __init__(self, child, margin):
        ChildBox.__init__(self, child)
        self.margin = margin

    def SLOW_on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        child = self.child
        child.on_layout(cvs, system)
        system.add(self.x == child.x)
        system.add(self.y == child.y)
        margin = self.margin
        system.add(self.left == child.left + margin)
        system.add(self.right == child.right + margin)
        system.add(self.top == child.top + margin)
        system.add(self.bot == child.bot + margin)

    def on_layout(self, cvs, system):
        child = self.child
        child.on_layout(cvs, system)
        self.x = child.x
        self.y = child.y
        margin = self.margin
        self.left = child.left + margin
        self.right = child.right + margin
        self.top = child.top + margin
        self.bot = child.bot + margin
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
        assert align in ("center north south east west northeast"+
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
    def __init__(self, child):
        ChildBox.__init__(self, child)

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
    "horizontal compound box: anchor left"
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
    "vertical compound box: anchor top"
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
    def __init__(self, rows, grid=False):
        boxs = []
        m = len(rows) # rows
        n = len(rows[0]) # cols
        assert n>0
        for row in rows:
            assert len(row) == n
            for box in row:
                box = Box.promote(box)
                boxs.append(box)
        self.rows = [list(row) for row in rows]
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

        for i in range(m): # rows
            x = self.x
            for j in range(n): # cols
                box = rows[i][j]
                system.add(box.x - box.left >= x)
                width = ws[j] # width of this col
                x += width
                system.add(box.x + box.right <= x)
            system.add(self.x + self.width >= x)

        for j in range(n): # cols
            y = self.y
            for i in range(m): # rows
                box = rows[i][j]
                system.add(box.y + box.top <= y)
                height = hs[i]
                y -= height
                system.add(box.y - box.bot >= y)
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



if __name__ == "__main__":


    print("OK\n")



