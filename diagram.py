#!/usr/bin/env python3

"""
string diagram's
"""

from math import pi

from bruhat.render.back import canvas, path
from bruhat.render.boxs import (Box, EmptyBox, HBox, VBox, AlignBox, 
    StrictVBox, StrictHBox, TextBox)



SIZE = 1.0

Box.DEBUG = False


class Dia(object): # Mixin
    def __init__(self, n_top=0, n_bot=0, n_left=0, n_right=0, 
        min_width=SIZE, min_height=SIZE):
        self.n_top = n_top
        self.n_bot = n_bot
        self.n_left = n_left
        self.n_right = n_right
        self.min_width = min_width
        self.min_height = min_height

    def on_layout(self, cvs, system):
        pass

    def on_render(self, cvs, system):
        pass

    def __mul__(self, other):
        return VDia([self, other])

    def __matmul__(self, other):
        return HDia([self, other])


class Atom(Dia):

    DEBUG = False

    def on_layout(self, cvs, system):
        get_var = lambda stem : system.get_var(stem, weight=0.)
        self.x_top = [get_var("x_top_%d"%i) for i in range(self.n_top)]
        self.x_bot = [get_var("x_bot%d"%i) for i in range(self.n_bot)]
        self.y_left = [get_var("y_left%d"%i) for i in range(self.n_left)]
        self.y_right = [get_var("y_right%d"%i) for i in range(self.n_right)]

        system.add(self.width >= self.min_width)
        system.add(self.height >= self.min_height)

        # left to right
        x0 = self.llx
        for x1 in self.x_top:
            system.add(x0 <= x1)
            x0 = x1
        if len(self.x_top):
            system.add(x0 <= self.urx)

        # left to right
        x0 = self.llx
        for x1 in self.x_bot:
            system.add(x0 <= x1)
            x0 = x1
        if len(self.x_bot):
            system.add(x0 <= self.urx)

        # TODO: top to bot for the left and right ports XXX

    def on_render(self, cvs, system):
        pass


class HDia(StrictHBox, Dia):
    "left to right arrangement of dias"
    def __init__(self, dias):
        assert len(dias)
        for dia in dias:
            assert isinstance(dia, Dia)
        i = 0
        while i+1<len(dias):
            left = dias[i]
            right = dias[i+1]
            assert left.n_right == right.n_left, "Dia mismatch at %d"%i
            i+=1
        n_top = sum(dia.n_top for dia in dias)
        n_bot = sum(dia.n_bot for dia in dias)
        n_left = dias[0].n_left
        n_right = dias[-1].n_right
        Dia.__init__(self, n_top, n_bot, n_left, n_right)
        #self.boxs = list(dias)
        StrictHBox.__init__(self, dias)

    def on_layout(self, cvs, system):
        HBox.on_layout(self, cvs, system)
        Dia.on_layout(self, cvs, system)
        dias = self.boxs

        # steal layout variables from the children
        x_top = []
        x_bot = []
        for dia in dias:
            x_top += dia.x_top
            x_bot += dia.x_bot
        y_left = dias[0].y_left
        y_right = dias[-1].y_right
        assert len(x_top) == self.n_top
        assert len(x_bot) == self.n_bot
        assert len(y_left) == self.n_left
        assert len(y_right) == self.n_right
        self.x_top = x_top
        self.x_bot = x_bot
        self.y_left = y_left
        self.y_right = y_right
                
        # join children left to right
        i = 0
        while i+1<len(dias):
            left = dias[i]
            right = dias[i+1]
            n = left.n_right # == right.n_left
            for j in range(n):
                system.add(left.y_right[j] == right.y_left[j])
            i+=1


class VDia(StrictVBox, Dia):
    "top down arrangement of dias"
    def __init__(self, dias):
        assert len(dias)
        for dia in dias:
            assert isinstance(dia, Dia)
        i = 0
        while i+1<len(dias):
            top = dias[i]
            bot = dias[i+1]
            assert top.n_bot == bot.n_top, "Dia mismatch at %d"%i
            i+=1
        n_left = sum(dia.n_left for dia in dias)
        n_right = sum(dia.n_right for dia in dias)
        n_top = dias[0].n_top
        n_bot = dias[-1].n_bot
        Dia.__init__(self, n_top, n_bot, n_left, n_right)
        #self.boxs = list(dias)
        StrictVBox.__init__(self, dias)

    def on_layout(self, cvs, system):
        VBox.on_layout(self, cvs, system)
        Dia.on_layout(self, cvs, system)
        dias = self.boxs

        # steal layout variables from the children
        y_left = []
        y_right = []
        for dia in dias:
            y_left += dia.y_left
            y_right += dia.y_right
        x_top = dias[0].x_top
        x_bot = dias[-1].x_bot
        assert len(x_top) == self.n_top
        assert len(x_bot) == self.n_bot
        assert len(y_left) == self.n_left
        assert len(y_right) == self.n_right
        self.x_top = x_top
        self.x_bot = x_bot
        self.y_left = y_left
        self.y_right = y_right
                
        # join children top to bot
        i = 0
        while i+1<len(dias):
            top = dias[i]
            bot = dias[i+1]
            n = top.n_bot
            for j in range(n):
                system.add(top.x_bot[j] == bot.x_top[j])
            i+=1


class VWire(Box, Atom):
    def __init__(self, min_width=0.5*SIZE, min_height=0.5*SIZE):
        Atom.__init__(self, n_top=1, n_bot=1, 
            min_width=min_width, min_height=min_height)

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)
        system.add(self.x_bot[0] == self.x_top[0])
        x = 0.5*(self.llx + self.urx)
        system.add(self.x_bot[0] == x, 1.0)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)
        x = system[self.x]
        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        x0 = system[self.x_bot[0]]
        cvs.stroke(path.line(x0, y-bot, x0, y+top))


class Cap(Box, Atom):
    def __init__(self):
        Atom.__init__(self, n_bot=2, min_height=0.5*SIZE)

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)
        x0 = self.x_bot[0]
        x1 = self.x_bot[1]
        system.add(x0 == self.llx + (1./4)*self.width, weight=1.0)
        system.add(x1 == self.llx + (3./4)*self.width, weight=1.0)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)

        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        y0 = y-bot
        x0 = system[self.x_bot[0]]
        x1 = system[self.x_bot[1]]
        x2 = 0.5*(x0+x1)
        radius = 0.5*(x1-x0)
        cvs.stroke(path.arc(x2, y0, radius, 0., pi))


class Cup(Box, Atom):
    def __init__(self):
        Atom.__init__(self, n_top=2, min_height=0.5*SIZE)

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)
        x0 = self.x_top[0]
        x1 = self.x_top[1]
        system.add(x0 == self.llx + (1./4)*self.width, weight=1.0)
        system.add(x1 == self.llx + (3./4)*self.width, weight=1.0)

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)

        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        y0 = y+top
        x0 = system[self.x_top[0]]
        x1 = system[self.x_top[1]]
        x2 = 0.5*(x0+x1)
        radius = 0.5*(x1-x0)
        cvs.stroke(path.arc(x2, y0, radius, pi, 0.))


def test():
    Box.DEBUG = False

    top = HDia([VWire(), Cap()])
    #mid = HDia([VWire(), VWire(), VWire()])
    bot = HDia([Cup(), VWire()])
    lsnake = VDia([top, bot])

    top = HDia([Cap(), VWire()])
    bot = HDia([VWire(), Cup()])
    rsnake = VDia([top, bot])

    boxs = [lsnake, "$=$", VWire(min_height=SIZE), "$=$", rsnake]
    boxs = [AlignBox(box, "center") for box in boxs]
    dia = HBox(boxs)

    if 0:
        boxs = ['a', '$=$']
        dia = TextBox("$a=b$")
        dia = HBox([dia, "$=$"])

    cvs = canvas.canvas()
    dia.render(cvs)
    cvs.writePDFfile("test_diagram.pdf")


def test_boxs():


    r = 1.0
    a = EmptyBox(top=r, bot=r)
    b = EmptyBox(top=r, bot=r)
    c = EmptyBox(left=r, right=r)
#    box = StrictVBox([a, c])
    box = VBox([a, c])
    
    cvs = canvas.canvas()
    box.render(cvs)
    cvs.writePDFfile("test_diagram.pdf")



if __name__ == "__main__":

    test()
    print("OK\n")


