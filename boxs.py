#!/usr/bin/env python3

from random import random, choice, seed

from math import pi


from bruhat.render.sat import Variable, System
from bruhat.render.back import RGBA, path, Canvas
from bruhat.argv import argv


def rnd(a, b):
    return (b-a)*random() + a

class Box(object):

    DEBUG = False
    did_layout = False

    @classmethod
    def promote(cls, item):
        if isinstance(item, Box):
            return item
        if isinstance(item, str):
            return TextBox(item)
        if isinstance(item, (tuple, list)):
            return HBox(item)
        raise TypeError(repr(item))

    def on_layout(self, cvs, system):
        assert not self.did_layout, "already called on_layout"
        if self.DEBUG:
            print("%s.on_layout" % (self.__class__.__name__,))
        for attr in 'x y left right top bot'.split():
            if attr in self.__dict__:
                continue
            stem = self.__class__.__name__ + '.' + attr

            # We don't try to minimize the absolute coordinate values.
            weight = 1.0 if attr not in 'xy' else 0.0
            v = system.get_var(stem, weight)
            setattr(self, attr, v)
        self.did_layout = True

    def on_render(self, cvs, system):
        if not self.DEBUG:
            return
        x = system[self.x]
        y = system[self.y]
        left = system[self.left]
        right = system[self.right]
        top = system[self.top]
        bot = system[self.bot]
        #cvs.set_line_width(0.5)
        cl = RGBA(1., 0., 0., 0.5)
        r = 0.1
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), [cl])
        cvs.stroke(path.line(x+r, y-r, x-r, y+r), [cl])
        cvs.fill(path.rect(x-left, y-bot, left+right, top+bot), [RGBA(0.5, 0.5, 0., 0.1)])
        cvs.stroke(path.rect(x-left, y-bot, left+right, top+bot), [cl])

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
    def urx(self):
        return self.x + self.right

    @property
    def ury(self):
        return self.y + self.top

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


    def render(self, cvs, x=0, y=0):
        system = System()
        self.on_layout(cvs, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        self.on_render(cvs, system)


class EmptyBox(Box):
    def __init__(self, top=0., bot=0., left=0., right=0.):
        self.top = top
        self.bot = bot
        self.left = left
        self.right = right


    
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
    def __init__(self, text):
        self.text = text

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        extents = cvs.text_extents(self.text)
        dx, dy, width, height = extents
        system.add(self.left + self.right == width+dx)
        system.add(self.top + self.bot == height)
        system.add(self.left == 0)
        assert dy >= 0., dy
        system.add(self.top == dy)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        cvs.text(x, y, self.text)


class MarginBox(Box):
    def __init__(self, child, margin):
        self.child = Box.promote(child)
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

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        self.child.on_render(cvs, system)


class AlignBox(Box):
    def __init__(self, child, align):
        self.child = Box.promote(child)
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

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        self.child.on_render(cvs, system)


#class VStrutBox(Box): # BLECH
#    def __init__(self, height):
#        self._height = height
#        self.left = 0.
#        self.right = 0.
#        
#    def on_layout(self, cvs, system):
#        Box.on_layout(self, cvs, system)
#        system.add(self.top + self.bot == self._height)


#class StrutBox(Box):
#    def __init__(self, top=None, bot=None, left=None, right=None):
#        self.top = top if top is not None else 0.
#        self.bot = bot if bot is not None else 0.
#        self.left = left if left is not None else 0.
#        self.right = right if right is not None else 0.
        

class CompoundBox(Box):
    def __init__(self, boxs):
        assert len(boxs)
        self.boxs = [Box.promote(box) for box in boxs]

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
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(self.y == box.y) # align
            system.add(box.left <= self.left)
            system.add(box.right <= self.right)
            system.add(box.top <= self.top)
            system.add(box.bot <= self.bot)


class HBox(CompoundBox):
    "horizontal compound box: anchor left"
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.left == 0.) # left anchor
        left = self.x
        for box in boxs:
            system.add(self.y == box.y) # align
            system.add(box.x - box.left == left)
            left += box.width
            system.add(box.top <= self.top)
            system.add(box.bot <= self.bot)
        system.add(self.x + self.width == left)


class VBox(CompoundBox):
    "vertical compound box: anchor top"
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.top == 0.) # top anchor
        y = self.y
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(box.y + box.top == y)
            y -= box.height
            system.add(box.left <= self.left)
            system.add(box.right <= self.right)
        system.add(self.y - self.bot == y)


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



def main():
    
#    box = HBox([
#        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
#        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
#    ])

    #EmptyBox.DEBUG = True
    Box.DEBUG = argv.get("debug") or argv.get("DEBUG")

    if 0:
        box = EmptyBox(1., 1., 1., 1.)
    elif 0:
        box = TableBox([
            [EmptyBox(.4, .1, 0.2, 2.2), EmptyBox(.3, 1.2, .5, 2.5),],
            [EmptyBox(.8, .1, 0.4, 1.2), EmptyBox(.5, 0.4, .5, 1.5),]
        ])
    elif 0:
        a, b = 0.2, 2.0
        rows = []
        for row in range(3):
            row = []
            for col in range(3):
                box = EmptyBox(rnd(a,b), rnd(a,b), rnd(a,b), rnd(a,b))
                row.append(box)
            rows.append(row)

        box = TableBox(rows)
    elif 0:
        box = HBox("geghh xxde xyeey".split())
    elif 1:
        rows = []
        for i in range(3):
            row = []
            for j in range(3):
                box = TextBox(choice("xbcgef")*(i+1)*(j+1))
                #box = MarginBox(box, 0.1)
                #box = AlignBox(box, "north")
                row.append(box)
            #row.append(EmptyBox(bot=1.0))
            rows.append(row)
        box = TableBox(rows)
    elif 1:
        box = OBox([
            EmptyBox(.4, .1, 0., 2.2),
            EmptyBox(.3, 0., .5, 2.5),
            EmptyBox(1., .5, .5, .5),
            FillBox(.2, .2),
        ])
    else:

        a, b = 0.2, 2.0
        rows = []
        for row in range(2):
            boxs = []
            for col in range(2):
                top = rnd(a, b)
                bot = rnd(a, b)
                left = rnd(a, b)
                right = rnd(a, b)
                box = EmptyBox(top, bot, left, right)
                boxs.append(box)
            boxs.append(TextBox("Hig%d !"%row))
            #boxs.append(TextBox(r"$\to$"))
            box = HBox(boxs)
            rows.append(box)
        box = VBox(rows)

    cvs = Canvas()
    box.render(cvs, 0., 0.)
    cvs.writePDFfile("output.pdf")
    #cvs.writeSVGfile("output.svg")


if __name__ == "__main__":

    seed(0)
    main()

    print("OK\n")



