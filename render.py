#!/usr/bin/env python3

"""
Expose a drawing api ...
"""

#from math import pi


# ----------------------------------------------------------------------------
# Item's go in Path's
#

class Item(object):
    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.__dict__)
    __repr__ = __str__

    def process_cairo(self, cxt):
        pass


class ClosePath(Item):
    def process_cairo(self, cxt):
        cxt.close_path()


class Moveto(Item):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process_cairo(self, cxt):
        cxt.move_to(self.x, self.y)


class Lineto(Item):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process_cairo(self, cxt):
        cxt.line_to(self.x, self.y)


class Curveto(Item):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def process_cairo(self, cxt):
        cxt.curve_to(self.x0, self.y0, 
            self.x1, self.y1, self.x2, self.y2)


class RMoveto(Item):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def process_cairo(self, cxt):
        cxt.rel_move_to(self.dx, self.dy)


class RLineto(Item):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def process_cairo(self, cxt):
        cxt.rel_line_to(self.dx, self.dy)


class RCurveto(Item):
    def __init__(self, dx0, dy0, dx1, dy1, dx2, dy2):
        self.dx0 = dx0
        self.dy0 = dy0
        self.dx1 = dx1
        self.dy1 = dy1
        self.dx2 = dx2
        self.dy2 = dy2

    def process_cairo(self, cxt):
        cxt.rel_curve_to(self.dx0, self.dy0, 
            self.dx1, self.dy1, self.dx2, self.dy2)


class Arc(Item):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in degrees"
        self.x = x
        self.y = y
        self.r = r
        self.angle1 = angle1
        self.angle2 = angle2

    def process_cairo(self, cxt):
        cxt.arc(self.x, self.y, self.r, 2*pi*self.angle1, 2*pi*self.angle2)


class Arcn(Arc):
    def process_cairo(self, cxt):
        cxt.arc_negative(self.x, self.y, self.r, 2*pi*self.angle1, 2*pi*self.angle2)


# ----------------------------------------------------------------------------
# Path's : a list of Item's
#

class Path(object):
    def __init__(self, items):
        for item in items:
            assert isinstance(item, Item)
        self.items = list(items)


class Line(Path):
    def __init__(self, x0, y0, x1, y1):
        Path.__init__(self, [
            Moveto(x0, y0), 
            Lineto(x1, y1)])


class Curve(Path):
    def __init__(self, x0, y0, x1, y1, x2, y2, x3, y3):
        Path.__init__(self, [
            Moveto(x0, y0), 
            Curveto(x1, y1, x2, y2, x3, y3)])
    

class Rect(Path):
    def __init__(self, x, y, width, height):
        Path.__init__(self, [
            Moveto(x, y), 
            Lineto(x+width, y),
            Lineto(x+width, y+height),
            Lineto(x, y+height),
            ClosePath()])


class Circle(Path):
    def __init__(self, x, y, r):
        Path.__init__(self, [
            Moveto(x+r, y),
            Arc(x, y, r, 0, 360),
            ClosePath()])
        


