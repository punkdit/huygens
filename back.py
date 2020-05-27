#!/usr/bin/env python3

"""
Expose a _drawing api with multiple (hopefully) backens.
"""

#from math import pi

class StringAble(object):
    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.__dict__)
    __repr__ = __str__


# ----------------------------------------------------------------------------
# Item's go in Path's
#

class Item(StringAble):

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
        cxt.curve_to(self.x0, self.y0, self.x1, self.y1, self.x2, self.y2)


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

class Path(StringAble):
    def __init__(self, items):
        for item in items:
            assert isinstance(item, Item)
        self.items = list(items)

    def process_cairo(self, cxt):
        for item in self.items:
            item.process_cairo(cxt)


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
        

# ----------------------------------------------------------------------------
# Deco
#


class Deco(StringAble):
    def pre_process_cairo(self, cxt):
        pass

    def post_process_cairo(self, cxt):
        pass


class Stroke(Deco):
    def post_process_cairo(self, cxt):
        cxt.stroke()

class Fill(Deco):
    def post_process_cairo(self, cxt):
        cxt.fill()

class RGBA(Deco):
    def __init__(self, r, g, b, a=1.0):
        self.cl = (r, g, b, a)

    def pre_process_cairo(self, cxt):
        cxt.set_source_rgba(*self.cl)

RGB = RGBA
    


# ----------------------------------------------------------------------------
# Canvas
#


class Drawable(StringAble):
    pass


class Text(Drawable):
    def __init__(self, x, y, text, decos=[]):
        for deco in decos:
            assert isinstance(deco, Deco)
        self.x = x
        self.y = y
        self.text = text
        self.decos = list(decos)

    def process_cairo(self, cxt):
        decos = self.decos
        cxt.save()
        for deco in decos:
            deco.pre_process_cairo(cxt)
        cxt.move_to(self.x, self.y)
        cxt.show_text(self.text)
        for deco in decos:
            deco.post_process_cairo(cxt)
        cxt.restore()


class DecoPath(Drawable):
    "A Path and a list of Deco's"
    def __init__(self, path, decos=[]):
        assert isinstance(path, Path)
        for deco in decos:
            assert isinstance(deco, Deco)
        self.path = path
        self.decos = list(decos)

    def process_cairo(self, cxt):
        decos = self.decos
        path = self.path
        cxt.save()
        for deco in decos:
            deco.pre_process_cairo(cxt)
        self.path.process_cairo(cxt)
        for deco in decos:
            deco.post_process_cairo(cxt)
        cxt.restore()


class Canvas(StringAble):
    "A list of Drawable's"
    def __init__(self, draws=[]):
        for draw in draws:
            assert isinstance(draw, Drawable)
        self.draws = list(draws)

    def append(self, draw):
        assert isinstance(draw, Drawable)
        self.draws.append(draw)

    def stroke(self, path, decos=[]):
        decos = list(decos)
        decos.append(Stroke())
        draw = DecoPath(path, decos)
        self.append(draw)

    def fill(self, path, decos=[]):
        decos = list(decos)
        decos.append(Fill())
        draw = DecoPath(path, decos)
        self.append(draw)

    def text_extents(self, text):
        import cairo
        surface = cairo.PDFSurface("/dev/null", 0, 0)
        cxt = cairo.Context(surface)
        return cxt.text_extents(text)

    def text(self, x, y, text, decos=[]):
        #print("Canvas.text", x, y, text)
        draw = Text(x, y, text, decos)
        self.append(draw)

    def writePDFfile(self, name):
        #print(self)
        import cairo
        W, H = 200, 200
        assert name.endswith(".pdf")
        surface = cairo.PDFSurface(name, W, H)
        cxt = cairo.Context(surface)
        cxt.set_line_width(0.5)
        for draw in self.draws:
            draw.process_cairo(cxt)
        surface.finish()

    def writeSVGfile(self, name):
        assert name.endswith(".svg")
        surface = cairo.SVGSurface(name, W, H)



# ----------------------------------------------------------------------------
# namespaces
#

class NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

path = NS(line = Line, curve = Curve, rect = Rect, circle = Circle)


# ----------------------------------------------------------------------------
# test
#

def test():

    cvs = Canvas()

    cvs.stroke(path.line(10, 10, 50, 50))

    cvs.writePDFfile("output.pdf")



if __name__ == "__main__":
    test()





