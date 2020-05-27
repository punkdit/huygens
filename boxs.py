#!/usr/bin/env python3

from random import random
from math import pi


from bruhat.render.sat import Variable, System
from bruhat.render import back
from bruhat.argv import argv


def rnd(a, b):
    return (b-a)*random() + a

class Box(object):

    DEBUG = True
    fixed = False

    def on_layout(self, cxt, system):
        assert not self.fixed, "already called on_layout"
        if self.DEBUG:
            print("%s.on_layout" % (self.__class__.__name__,))
        for attr in 'x y left right top bot'.split():
            if attr in self.__dict__:
                continue
            stem = self.__class__.__name__ + '.' + attr
            v = system.get_var(stem)
            setattr(self, attr, v)
        self.fixed = True

    def on_render(self, cxt, system):
        if not self.DEBUG:
            return
        cxt.save()
        cxt.set_source_rgba(1., 0., 0., 0.5)
        x = system[self.x]
        y = system[self.y]
        left = system[self.left]
        right = system[self.right]
        top = system[self.top]
        bot = system[self.bot]
        cxt.set_line_width(0.5)
        #cxt.move_to(x, y)
        #cxt.arc(x, y, 1.0, 0, 2*pi)
        #cxt.stroke()
        r = 1.4
        cxt.move_to(x-r, y-r)
        cxt.line_to(x+r, y+r)
        cxt.stroke()
        cxt.move_to(x+r, y-r)
        cxt.line_to(x-r, y+r)
        cxt.stroke()
        cxt.rectangle(x-left, y-top, left+right, top+bot)
        cxt.stroke()
        cxt.restore()

    @property
    def width(self):
        return self.left + self.right

    @property
    def height(self):
        return self.top + self.bot

    def render(self, cxt, x=0, y=0):
        system = System()
        self.on_layout(cxt, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        self.on_render(cxt, system)


class EmptyBox(Box):
    def __init__(self, top, bot, left, right):
        self.top = top
        self.bot = bot
        self.left = left
        self.right = right
        self.DEBUG = True

    def on_layout(self, cxt, system):
        #assert "x" not in self.__dict__
        Box.on_layout(self, cxt, system)


class TextBox(Box):
    def __init__(self, text):
        self.text = text

    def on_layout(self, cxt, system):
        Box.on_layout(self, cxt, system)
        extents = cxt.text_extents(self.text)
        (dx, dy, width, height, _, _) = extents
        system.add(self.left + self.right == width+dx)
        system.add(self.top + self.bot == height)
        system.add(self.left == 0)
        assert dy <= 0., dy
        system.add(self.top == -dy)

    def on_render(self, cxt, system):
        Box.on_render(self, cxt, system)
        x = system[self.x]
        y = system[self.y]
        cxt.move_to(x, y)
        cxt.show_text(self.text)


class CompoundBox(Box):
    def __init__(self, boxs):
        assert len(boxs)
        self.boxs = list(boxs)

    def on_layout(self, cxt, system):
        Box.on_layout(self, cxt, system)
        for box in self.boxs:
            box.on_layout(cxt, system)

    def on_render(self, cxt, system):
        Box.on_render(self, cxt, system)
        for box in self.boxs:
            box.on_render(cxt, system)


class HBox(CompoundBox):
    def on_layout(self, cxt, system):
        CompoundBox.on_layout(self, cxt, system)
        boxs = self.boxs
        left = self.x
        for box in boxs:
            system.add(self.y == box.y) # align
            system.add(box.x - box.left == left)
            left += box.width
            system.add(box.top <= self.top)
            system.add(box.bot <= self.bot)
        system.add(self.x + self.width == left)
        system.add(self.left == 0.)


class VBox(CompoundBox):
    def on_layout(self, cxt, system):
        CompoundBox.on_layout(self, cxt, system)
        boxs = self.boxs
        top = self.y
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(box.y - box.top == top)
            top += box.height
            system.add(box.left <= self.left)
            system.add(box.right <= self.right)
        system.add(self.y + self.height == top)
        system.add(self.top == 0.)


def main():
    W, H = 200, 200
    #surface = cairo.SVGSurface("example.svg", W, H)
    surface = cairo.PDFSurface("example.pdf", W, H)
    cxt = cairo.Context(surface)
    #print(' '.join(dir(cxt)))
    
#    box = HBox([
#        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
#        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
#    ])

    
    if 1:
        a, b = 10, 20
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
            boxs.append(TextBox("hi !"))
            box = HBox(boxs)
            rows.append(box)
        box = VBox(rows)

#    box = VBox([
#        EmptyBox(20, 5, 5, 18),
#        EmptyBox(10, 8, 9, 16),
#    ])

    #box = TextBox('xyyxy !')
    box.render(cxt, W/2., H/2.)

    surface.finish()


if __name__ == "__main__":

    main()

    print("OK\n")



