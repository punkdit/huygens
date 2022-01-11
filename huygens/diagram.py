#!/usr/bin/env python3

"""
string diagram's
"""

from math import pi
from functools import reduce
from operator import add

from huygens.base import Base
from huygens.front import canvas, path, trafo, style, color, deco
from huygens.box import (Box, EmptyBox, HBox, VBox, AlignBox, 
    StrictVBox, StrictHBox, TextBox, RectBox, MarginBox, BoxDeco, CanBox)

PIP = 0.001
SCALE_SIZE = 2.0

class Config(object):

    stack = []

    def __init__(self, hflow=None, vflow=None, size=None, **kw):
        if size is not None:
            size *= SCALE_SIZE
        if self.stack:
            hflow = get_hflow(hflow)
            vflow = get_vflow(vflow)
            size = get_size(size)
        assert hflow in "left right".split()
        assert vflow in "up down".split()
        self.hflow = hflow
        self.vflow = vflow
        self.size = size
        self.__dict__.update(kw)
        self.stack.append(self)

    @classmethod
    def get_size(cls, size=None):
        conf = cls.stack[-1]
        size = conf.size if size is None else size
        return size

    @classmethod
    def get_hflow(cls, hflow=None):
        conf = cls.stack[-1]
        hflow = conf.hflow if hflow is None else hflow
        return hflow

    @classmethod
    def get_vflow(cls, vflow=None):
        conf = cls.stack[-1]
        vflow = conf.vflow if vflow is None else vflow
        return vflow

    @classmethod
    def restore(cls):
        assert cls.stack, "empty stack!"
        conf = cls.stack.pop()

    #def __del__(self):
    #    self.restore()


#config = Config.config
config = Config
config(hflow="right", vflow="down", size=1.0)
restore = Config.restore
get_size = Config.get_size
get_hflow = Config.get_hflow
get_vflow = Config.get_vflow


conv = lambda x0, x1, t=0.5: (1.-t)*x0 + t*x1


class Dia(Base): # Mixin
    def __init__(self, n_top=0, n_bot=0, n_left=0, n_right=0, 
        min_width=None, min_height=None):
        self.n_top = n_top
        self.n_bot = n_bot
        self.n_left = n_left
        self.n_right = n_right
        self.min_width = get_size(min_width)
        self.min_height = get_size(min_height)

    def on_layout(self, cvs, system):
        pass

    def on_render(self, cvs, system):
        pass

    def __mul__(self, other):
        vflow = get_vflow()
        if vflow == "down":
            return VDia([self, other])
        elif vflow == "up":
            return VDia([other, self])
        else:
            assert 0, vflow

    def __matmul__(self, other):
        hflow = get_hflow()
        if hflow == "right":
            return HDia([self, other])
        elif hflow == "left":
            return HDia([other, self])
        else:
            assert 0, hflow

    def get_hunits(self):
        return 1

    def get_vunits(self):
        return 1



class Atom(Dia):
    """
        Has ports at the top and bot: n_top, n_bot
        TODO: n_left and n_right ports
    """

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
        assert self.n_left == 0, "not implemented"
        assert self.n_right == 0, "not implemented"

    def on_render(self, cvs, system):
        pass


class HDia(StrictHBox, Dia):
    "left to right arrangement of dias"
    def __init__(self, dias):
        assert len(dias)
        for dia in dias:
            assert isinstance(dia, Dia), type(dia)
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

#    # this is just too tricky ... better to just set attributes in on_layout
#    @property
#    def x_top(self):
#        x_top = reduce(add, [dia.x_top for dia in self.boxs])
#        return x_top
#
#    @property
#    def x_bot(self):
#        x_bot = reduce(add, [dia.x_bot for dia in self.boxs])
#        return x_bot

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
        self.x_top = x_top # could also make this a property ...
        self.x_bot = x_bot # could also make this a property ...
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

    def get_hunits(self):
        hunits = sum([dia.get_hunits() for dia in self.boxs])
        return hunits

    def get_vunits(self):
        vunits = max([dia.get_vunits() for dia in self.boxs])
        return vunits


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

#    # this is just too tricky ... better to just set attributes in on_layout
#    @property
#    def x_top(self):
#        return self.boxs[0].x_top
#
#    @property
#    def x_bot(self):
#        return self.boxs[-1].x_bot

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
        self.x_top = x_top # could also make this a property ...
        self.x_bot = x_bot # could also make this a property ...
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

    def get_hunits(self):
        #for dia in self.boxs:
        #    dia.get_hunits()
        hunits = max([dia.get_hunits() for dia in self.boxs])
        return hunits

    def get_vunits(self):
        vunits = sum([dia.get_vunits() for dia in self.boxs])
        return vunits



class VWire(Box, Atom):
    def __init__(self, min_width=None, min_height=None, attrs=[]):
        min_width = 0.5*get_size(min_width)
        min_height = 0.5*get_size(min_height)
        Atom.__init__(self, n_top=1, n_bot=1, 
            min_width=min_width, min_height=min_height)
        self.attrs = attrs

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
        top = system[self.top] + PIP
        bot = system[self.bot] + PIP
        x0 = system[self.x_bot[0]]
        #cvs.stroke(path.line(x0, y-bot, x0, y+top), self.attrs)
        cvs.stroke(path.line(x0, y+top, x0, y-bot), self.attrs)


class Cap(Box, Atom):
    def __init__(self, weight=1.0):
        Atom.__init__(self, n_bot=2, min_height=0.5*get_size())
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
        bot = system[self.bot] + PIP
        y0 = y-bot
        x0 = system[self.x_bot[0]]
        x1 = system[self.x_bot[1]]
        x2 = 0.5*(x0+x1)
        radius = 0.5*(x1-x0)
        cvs.stroke(path.arc(x2, y0, radius, 0., pi))


class Cup(Box, Atom):
    def __init__(self, weight=1.0):
        Atom.__init__(self, n_top=2, min_height=0.5*get_size())
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
        top = system[self.top] + PIP
        bot = system[self.bot]
        y0 = y+top
        x0 = system[self.x_top[0]]
        x1 = system[self.x_top[1]]
        x2 = 0.5*(x0+x1)
        radius = 0.5*(x1-x0)
        cvs.stroke(path.arc(x2, y0, radius, pi, 0.))


class Multi(Box, Atom):
    """
    Even layout of top and bot ports.
    """
    def __init__(self, n_top, n_bot,
            weight=1.0, min_width=None, min_height=None,
            top_attrs=None, bot_attrs=None, **kw
        ):
        if get_vflow() == "up":
            n_top, n_bot = n_bot, n_top # i hope i don't regret this
        if top_attrs is None:
            top_attrs = [[] for i in range(n_top)]
        if bot_attrs is None:
            bot_attrs = [[] for i in range(n_bot)]
        if min_width is None:
            min_width = 0.5*get_size()
            if n_top > 1 or n_bot > 1:
                min_width = 1.0*get_size()
        if min_height is None:
            min_height = 0.5*get_size()
        self.weight = weight
        self.top_attrs = top_attrs
        self.bot_attrs = bot_attrs
        assert len(top_attrs) == n_top
        assert len(bot_attrs) == n_bot
        Atom.__init__(self, n_top=n_top, n_bot=n_bot,
            min_width=min_width, min_height=min_height)

    def get_shape(self):
        return (self.__class__.__name__, self.n_top, self.n_bot)

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
    def __init__(self, *args, **kw):
        Multi.__init__(self, *args, **kw)
        pip = kw.get("pip")
        if pip is not None:
            pip_align = kw.get("pip_align")
            pip = Box.promote(pip, pip_align)
        self.clamp = kw.get("clamp", 0.3)
        self.pip = pip
        self.trace = {}

#    def on_layout(self, cvs, system):
#        Multi.on_layout(self, cvs, system)
#        get_var = system.get_var
#        pipx = get_var("pipx")
#        system.add(x_bot[i] == x, weight=self.weight)
#        self.pipx = pipx

    # XXX the pip should be at the anchor... right now
    # the anchor is just free to float XXX

    def _get_pipx(self, x_top, x_bot): # override in TBone below
        n_top, n_bot = self.n_top, self.n_bot
        clamp = self.clamp
        if n_bot == 1 and n_top > 1:
            x0 = x_bot[0]
            xmin, xmax = x_top[0], x_top[-1]
            assert xmin <= xmax, "wup?"
            x0 = max(x0, conv(xmin, xmax, clamp)) # clamp
            x0 = min(x0, conv(xmin, xmax, 1.0-clamp)) # clamp
        elif n_top == 1 and n_bot > 1:
            x0 = x_top[0]
            xmin, xmax = x_bot[0], x_bot[-1]
            assert xmin <= xmax, "wup?"
            x0 = max(x0, conv(xmin, xmax, clamp)) # clamp
            x0 = min(x0, conv(xmin, xmax, 1.0-clamp)) # clamp
        else:
            n = n_top + n_bot
            x0 = (1./n)*sum(x_top + x_bot)
        return x0

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        Atom.on_render(self, cvs, system)

        n = self.n_top + self.n_bot
        if n==0:
            return

        y = system[self.y]
        top = system[self.top] + PIP
        bot = system[self.bot] + PIP
        y_top = y+top
        y_bot = y-bot
        x_mid, y_mid = self.get_align("center")
        x0, y0 = system[x_mid], system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]
        x0 = self._get_pipx(x_top, x_bot)

        conv = lambda x0, x1, t: (1.-t)*x0 + t*x1

        top_attrs = self.top_attrs
        bot_attrs = self.bot_attrs

        trace = self.trace
        trace["top"] = []
        trace["bot"] = []

        y3 = y_top
        for x3, attrs in zip(x_top, top_attrs):
            x2, y2 = x3, conv(y3, y0, 0.3)
            x1, y1 = conv(x0, x3, 0.7), conv(y3, y0, 0.7)
            p = path.curve(x3, y3, x2, y2, x1, y1, x0, y0)
            trace["top"].append(p)
            cvs.stroke(p, attrs)

        y3 = y_bot
        for x3, attrs in zip(x_bot, bot_attrs):
            x2, y2 = x3, conv(y3, y0, 0.3)
            x1, y1 = conv(x0, x3, 0.7), conv(y3, y0, 0.7)
            p = path.curve(x0, y0, x1, y1, x2, y2, x3, y3)
            trace["bot"].append(p)
            cvs.stroke(p, attrs)

        #cvs.fill(path.circle(x0, y0, 0.04))
        if self.pip is not None:
            self.pip.render(cvs, x0, y0)


class TBone(Spider):
    def __init__(self, *args, **kw):
        Spider.__init__(self, *args, **kw)
        connect = kw.get("connect")
        assert connect is not None, "please specify connect"
        assert len(connect) == 2, "connect top index to bot index"
        self.connect = connect

    def _get_pipx(self, x_top, x_bot):
        connect = self.connect
        i, j = connect
        x0 = 0.5*(x_top[i] + x_bot[j])
        return x0

    def on_layout(self, cvs, system):
        Spider.on_layout(self, cvs, system)
        connect = self.connect
        i, j = connect
        system.add(self.x_top[i] == self.x_bot[j])


class Relation(Multi):
    def __init__(self, n_top, n_bot, toptop=[], topbot=[], botbot=[], weight=1.0, 
            attrs=[], top_attrs=None, bot_attrs=None):
        if get_vflow() == "up":
            n_top, n_bot = n_bot, n_top # i hope i don't regret this
        min_width = 0.5*get_size()
        if n_top > 1 or n_bot > 1:
            min_width = 1.0*get_size()
        if top_attrs is None:
            top_attrs = [[]]*n_top
        if bot_attrs is None:
            bot_attrs = [[]]*n_bot
        assert len(top_attrs) == n_top
        assert len(bot_attrs) == n_bot
        min_height = 0.5*get_size()
        self.weight = weight
        for (a, b) in topbot:
            assert 0<=a<n_top
            assert 0<=b<n_bot
        for (a, b) in toptop:
            assert 0<=a<n_top
            assert 0<=b<n_top
        for (a, b) in botbot:
            assert 0<=a<n_bot
            assert 0<=b<n_bot
        assert len(set(topbot)) == len(topbot), "non unique relation"
        assert len(set(toptop)) == len(toptop), "non unique relation"
        assert len(set(botbot)) == len(botbot), "non unique relation"
        self.topbot = list(topbot)
        self.toptop = list(toptop)
        self.botbot = list(botbot)
        self.attrs = attrs # applies to all strings
        self.top_attrs = list(top_attrs) # applies to top strings
        self.bot_attrs = list(bot_attrs) # applies to bot strings
        Atom.__init__(self, n_top=n_top, n_bot=n_bot,
            min_width=min_width, min_height=min_height)

    def on_layout(self, cvs, system):
        Multi.on_layout(self, cvs, system)
        system.add(self.height >= 0.5*self.width)

    def on_render(self, cvs, system):
        Multi.on_render(self, cvs, system)

        n = self.n_top + self.n_bot
        if n==0:
            return

        y = system[self.y]
        top = system[self.top] + PIP
        bot = system[self.bot] + PIP
        y_top = y+top
        y_bot = y-bot
        x_mid, y_mid = self.get_align("center")
        x_mid, y_mid = system[x_mid], system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]
        x_avg = (1./n)*sum(x_top + x_bot)

        attrs = self.attrs
        top_attrs = self.top_attrs
        bot_attrs = self.bot_attrs
        conv = lambda x0, x1, t=0.5: (1.-t)*x0 + t*x1

        for (i_top, j_top) in self.toptop:
            x0 = x_top[i_top]
            x1 = x_top[j_top]
            #cvs.stroke(path.curve(x0, y_bot, x0, y_mid, x1, y_mid, x1, y_top))
            x = conv(x0, x1)
            r = 0.5*(x1-x0)
            cvs.stroke(path.arc(x, y_top, r, pi, 2*pi), 
                attrs + top_attrs[i_top])

        for (i_top, i_bot) in self.topbot:
            x0 = x_bot[i_bot]
            x1 = x_top[i_top]
            cvs.stroke(path.curve(x0, y_bot, x0, y_mid, x1, y_mid, x1, y_top), 
                attrs + top_attrs[i_top] + bot_attrs[i_bot])

        for (i_bot, j_bot) in self.botbot:
            x0 = x_bot[i_bot]
            x1 = x_bot[j_bot]
            x = conv(x0, x1)
            r = 0.5*(x1-x0)
            cvs.stroke(path.arc(x, y_bot, r, 0, pi), 
                attrs + bot_attrs[i_bot])



class Braid(Multi):
    def __init__(self, inverse=False, space=0.5, weight=1.0, min_width=None, min_height=None):
        min_width = get_size(min_width)
        min_height = 0.5*get_size(min_height)
        Multi.__init__(self, 2, 2, weight, min_width=min_width, min_height=min_height)
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
        top = system[self.top] + PIP
        bot = system[self.bot] + PIP
        y_top = y+top
        y_bot = y-bot
        y_mid = conv(self.lly, self.ury)
        y_mid = system[y_mid]

        x_top = [system[x] for x in self.x_top]
        x_bot = [system[x] for x in self.x_bot]
        x_mid = conv(x_bot[0], x_bot[1])

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
                p = path.curve(x0, y_bot, x0, y_mid, x1, y_mid, x1, y_top)
                cvs.stroke(p)

            under = not under


def test():
    from huygens import config
    config("pdftex")

    #Box.DEBUG=True

    st_dashed = [style.linestyle.dashed]
    st_arrow = [deco.marrow()]
    st_rarrow = [deco.marrow(reverse=True)]

    #box = Spider(1, 2)
    
    box = TBone(1, 2, connect=[0,0], weight=0.5)
    box = box * (VWire() @ Spider(1, 2))
    box = box * (Spider(2, 1) @ VWire())
    
    cvs = canvas.canvas()
    box.render(cvs)
    cvs.writePDFfile("output.pdf")



if __name__ == "__main__":

    test()

    print("OK")



