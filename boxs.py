#!/usr/bin/env python3

from random import random, seed

from math import pi


from bruhat.render.sat import Variable, System
from bruhat.render.back import RGBA, path, Canvas
from bruhat.argv import argv


def rnd(a, b):
    return (b-a)*random() + a

class Box(object):

    DEBUG = True
    fixed = False

    def on_layout(self, cvs, system):
        assert not self.fixed, "already called on_layout"
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
        self.fixed = True

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
        cvs.stroke(path.rect(x-left, y-bot, left+right, top+bot), [cl])

    @property
    def width(self):
        return self.left + self.right

    @property
    def height(self):
        return self.top + self.bot

    def render(self, cvs, x=0, y=0):
        system = System()
        self.on_layout(cvs, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        self.on_render(cvs, system)


class EmptyBox(Box):
    def __init__(self, top, bot, left, right):
        self.top = top
        self.bot = bot
        self.left = left
        self.right = right
        self.DEBUG = True

    def on_layout(self, cvs, system):
        #assert "x" not in self.__dict__
        Box.on_layout(self, cvs, system)


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


class CompoundBox(Box):
    def __init__(self, boxs):
        assert len(boxs)
        self.boxs = list(boxs)

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        for box in self.boxs:
            box.on_layout(cvs, system)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        for box in self.boxs:
            box.on_render(cvs, system)


class HBox(CompoundBox):
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.left == 0.)
        left = self.x
        for box in boxs:
            system.add(self.y == box.y) # align
            system.add(box.x - box.left == left)
            left += box.width
            system.add(box.top <= self.top)
            system.add(box.bot <= self.bot)
        system.add(self.x + self.width == left)


class VBox(CompoundBox):
    def on_layout(self, cvs, system):
        CompoundBox.on_layout(self, cvs, system)
        boxs = self.boxs
        system.add(self.top == 0.)
        y = self.y
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(box.y + box.top == y)
            y -= box.height
            system.add(box.left <= self.left)
            system.add(box.right <= self.right)
        system.add(self.y - self.bot == y)


def main():
    
#    box = HBox([
#        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
#        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
#    ])

    if 0:
        box = EmptyBox(10., 10., 10., 10.)
        cvs = Canvas()
        box.render(cvs, 0., 0.)
        cvs.writePDFfile("outbox.pdf")

    elif 0:
        box = VBox([
            EmptyBox(40., 10., 0., 20.),
            EmptyBox(30., 0., 5., 25.),
            EmptyBox(10., 5., 5., 5.),
        ])

        #box = VBox([HBox([EmptyBox(10., 10., 10., 10.)])])
    
        cvs = Canvas()
        box.render(cvs, 0., 0.)
        cvs.writePDFfile("outbox.pdf")
        #return
    
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
            boxs.append(TextBox("hig%d !"%row))
            #boxs.append(TextBox(r"$\to$"))
            box = HBox(boxs)
            rows.append(box)
        box = VBox(rows)

        cvs = Canvas()
        box.render(cvs, 0., 0.)
        cvs.writePDFfile("output.pdf")
        cvs.writeSVGfile("output.svg")


if __name__ == "__main__":

    seed(0)
    main()

    print("OK\n")



