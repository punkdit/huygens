#!/usr/bin/env python3

"""
string diagram's
"""

from math import pi

from bruhat.render.front import canvas, path, trafo
from bruhat.render.boxs import (Box, EmptyBox, HBox, VBox, AlignBox, 
    StrictVBox, StrictHBox, TextBox)



SIZE = 2.0


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
    def __init__(self, weight=1.0):
        Atom.__init__(self, n_bot=2, min_height=0.5*SIZE)
        self.weight = weight

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)
        x0 = self.x_bot[0]
        x1 = self.x_bot[1]
        system.add(x0 == self.llx + (1./4)*self.width, weight=self.weight)
        system.add(x1 == self.llx + (3./4)*self.width, weight=self.weight)
        system.add(self.height >= x1-x0)

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
    def __init__(self, weight=1.0):
        Atom.__init__(self, n_top=2, min_height=0.5*SIZE)
        self.weight = weight

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)
        x0 = self.x_top[0]
        x1 = self.x_top[1]
        system.add(x0 == self.llx + (1./4)*self.width, weight=self.weight)
        system.add(x1 == self.llx + (3./4)*self.width, weight=self.weight)
        system.add(2*self.height >= x1-x0)

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


class Multi(Box, Atom):
    def __init__(self, n_top, n_bot, weight=1.0):
        min_width = 0.5*SIZE
        if n_top > 1 or n_bot > 1:
            min_width = 1.0*SIZE
        min_height = 0.5*SIZE
        self.weight = weight
        Atom.__init__(self, n_top=n_top, n_bot=n_bot,
            min_width=min_width, min_height=min_height)

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        Atom.on_layout(self, cvs, system)

        width = self.width
        llx = self.llx

        n_top = self.n_top
        x_top = self.x_top
        if n_top>0:
            dx = width/(2*n_top)
            x = llx + dx
            for i in range(n_top):
                system.add(x_top[i] == x, weight=self.weight)
                x += 2*dx

        n_bot = self.n_bot
        x_bot = self.x_bot
        if n_bot>0:
            dx = width/(2*n_bot)
            x = llx + dx
            for i in range(n_bot):
                system.add(x_bot[i] == x, weight=self.weight)
                x += 2*dx

        #n = n_top + n_bot
        #if n==0:
        #    return
        #middle = system.get_var("middle", weight=0.)
        #system.add(middle == (1./n)*sum(x_top + x_bot))
        #self.middle = middle

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)


class Spider(Multi):
    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)

        n = self.n_top + self.n_bot
        if n==0:
            return

        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        y_top = y+top
        y_bot = y-bot
        x_mid, y_mid = self.get_align("center")
        x0, y0 = system[x_mid], system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]
        x0 = (1./n)*sum(x_top + x_bot)

        conv = lambda x0, x1, t: (1.-t)*x0 + t*x1

        y3 = y_top
        for x3 in x_top:
            x2, y2 = x3, conv(y3, y0, 0.3)
            x1, y1 = conv(x0, x3, 0.7), conv(y3, y0, 0.7)
            cvs.stroke(path.curve(x0, y0, x1, y1, x2, y2, x3, y3))

        y3 = y_bot
        for x3 in x_bot:
            x2, y2 = x3, conv(y3, y0, 0.3)
            x1, y1 = conv(x0, x3, 0.7), conv(y3, y0, 0.7)
            cvs.stroke(path.curve(x0, y0, x1, y1, x2, y2, x3, y3))

        cvs.fill(path.circle(x0, y0, 0.04))


class Relation(Multi):
    def __init__(self, n_top, n_bot, rels, weight=1.0):
        min_width = 0.5*SIZE
        if n_top > 1 or n_bot > 1:
            min_width = 1.0*SIZE
        min_height = 0.5*SIZE
        self.weight = weight
        for (a, b) in rels:
            assert 0<=a<n_top
            assert 0<=b<n_bot
        assert len(set(rels)) == len(rels)
        self.rels = list(rels)
        Atom.__init__(self, n_top=n_top, n_bot=n_bot,
            min_width=min_width, min_height=min_height)

    def on_render(self, cvs, system):
        Multi.on_render(self, cvs, system)

        n = self.n_top + self.n_bot
        if n==0:
            return

        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        y_top = y+top
        y_bot = y-bot
        x_mid, y_mid = self.get_align("center")
        x_mid, y_mid = system[x_mid], system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]
        x_avg = (1./n)*sum(x_top + x_bot)

        conv = lambda x0, x1, t: (1.-t)*x0 + t*x1

        for (i_top, i_bot) in self.rels:
            x0 = x_bot[i_bot]
            x1 = x_top[i_top]
            cvs.stroke(path.curve(x0, y_bot, x0, y_mid, x1, y_mid, x1, y_top))


class Braid(Multi):
    def __init__(self, inverse=False, space=0.8, weight=1.0):
        Multi.__init__(self, 2, 2, weight)
        self.inverse = inverse
        self.space = space

    def on_layout(self, cvs, system):
        Multi.on_layout(self, cvs, system)
        # need this extra constraint
        for i in [0, 1]:
            system.add(self.x_bot[i] == self.x_top[i])

    def on_render(self, cvs, system):
        Multi.on_render(self, cvs, system)

        y = system[self.y]
        top = system[self.top]
        bot = system[self.bot]
        y_top = y+top
        y_bot = y-bot
        x_mid, y_mid = self.get_align("center")
        x_mid, y_mid = system[x_mid], system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]

        conv = lambda x0, x1, t=0.5: (1.-t)*x0 + t*x1

        alpha = self.space
        under = self.inverse
        x0, x1 = x_bot[1], x_top[0]

        for i0, i1 in [(0, 1), (1, 0)]:
            x0, x1 = x_bot[i0], x_top[i1]

            if under:
                # I think this is de Casteljau subdivision...
                xt, yt = conv(x0, x1, 0.25), conv(y_bot, y_mid, 0.75) # control point
                p = path.curve(
                    x0, y_bot, 
                    x0, conv(y_bot, y_mid),
                    xt, yt,
                    conv(xt, x_mid, alpha), conv(yt, y_mid, alpha))
                cvs.stroke(p)
        
                xt, yt = conv(x0, x1, 0.75), conv(y_mid, y_top, 0.25) # control point
                p = path.curve(
                    conv(xt, x_mid, alpha), conv(yt, y_mid, alpha),
                    xt, yt,
                    x1, conv(y_mid, y_top),
                    x1, y_top)
                cvs.stroke(p)

            else:
                x0 = x_bot[0]
                x1 = x_top[1]
                p = path.curve(x0, y_bot, x0, y_mid, x1, y_mid, x1, y_top)
                cvs.stroke(p)

            under = not under



def test_snake():
    Box.DEBUG = True

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

    cvs = canvas.canvas()
    dia.render(cvs)
    cvs.writePDFfile("test_snake.pdf")


def test():
    Box.DEBUG = True

    # Note: __mul__ composes top-down, like VBox.

    box = Spider(2, 2) @ VWire()
    #box = box * (VWire() @ Spider(2, 2))
    box = box * (Spider(3, 1))
    box = box @ VWire()
    #box = box * Cup()
    box = box * Spider(2, 0, weight=0.9)
    #box = (Cap() @ Cap()) * box
    box = (Cap() @ Spider(0, 1) @ Spider(0, 1)) * box

    cvs = canvas.canvas()
    #cvs.append(trafo.rotate(-pi/4))
    #cvs.append(trafo.scale(1., -1.))
    #cvs.append(trafo.rotate(pi/4))
    #cvs.append(trafo.rotate(pi/2))
    box.render(cvs)
    cvs.writePDFfile("test_diagram.pdf")


def test_relation():
    Box.DEBUG = True

    # Note: __mul__ composes top-down, like VBox.

    box = Relation(3, 4, [(0, 0), (1, 0), (2, 1), (1, 2), (2, 3)])
    box = box * (Cup() @ Cup())
    box = (VWire() @ Spider(0, 1) @ VWire()) * box
    box = Cap() * box

    cvs = canvas.canvas()
    box.render(cvs)
    cvs.writePDFfile("test_diagram.pdf")


def test():
    Box.DEBUG = False

    # Note: __mul__ composes top-down, like VBox.

    Id = VWire

    s12 = lambda : Braid() @ Id()
    s23 = lambda : Id() @ Braid()

    lhs = s12() * s23() * s12()
    #lhs = lhs @ Braid()
    rhs = s23() * s12() * s23()
    box = HBox([lhs, "$=$", rhs], align="center")

    cvs = canvas.canvas()
    box.render(cvs)
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

    test_snake()
    test()
    print("OK\n")


