#!/usr/bin/env python3

"""
Expose a _drawing api with multiple (hopefully) backends.

The api uses the first quadrant coordinate system:

        y 
        ^
        |
        |
        | positive x and y
        |    
        |
      --+-------------------> x
        |
    
"""

#from math import pi


CAIRO_SCALE = 72.0/2.54 # convert cm's to points at 72 dpi.


class Base(object):
    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.__dict__)
    __repr__ = __str__


def n_min(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def n_max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


class Bound(Base):
    def __init__(self, llx=None, lly=None, urx=None, ury=None):
        self.llx = llx
        self.lly = lly
        self.urx = urx
        self.ury = ury

    def union(self, other):
        "union"
        llx = n_min(self.llx, other.llx)
        lly = n_min(self.lly, other.lly)
        urx = n_max(self.urx, other.urx)
        ury = n_max(self.ury, other.ury)
        return Bound(llx, lly, urx, ury)
    __add__ = union

    def update(self, other):
        "union update"
        self.llx = n_min(self.llx, other.llx)
        self.lly = n_min(self.lly, other.lly)
        self.urx = n_max(self.urx, other.urx)
        self.ury = n_max(self.ury, other.ury)

    @property
    def width(self):
        return self.urx - self.llx

    @property
    def height(self):
        return self.ury - self.lly




# ----------------------------------------------------------------------------
# Item's go in Path's
#

class Item(Base):

    def get_bound(self):
        return Bound()

    def process_cairo(self, cxt):
        pass


class ClosePath(Item):
    def process_cairo(self, cxt):
        cxt.close_path()


class Moveto(Item):
    def __init__(self, x, y):
        self.x = CAIRO_SCALE*x
        self.y = CAIRO_SCALE*y

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.move_to(self.x, -self.y)


class Lineto(Item):
    def __init__(self, x, y):
        self.x = CAIRO_SCALE*x
        self.y = CAIRO_SCALE*y

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.line_to(self.x, -self.y)


class Curveto(Item):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = CAIRO_SCALE*x0
        self.y0 = CAIRO_SCALE*y0
        self.x1 = CAIRO_SCALE*x1
        self.y1 = CAIRO_SCALE*y1
        self.x2 = CAIRO_SCALE*x2
        self.y2 = CAIRO_SCALE*y2

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.curve_to(self.x0, -self.y0, self.x1, -self.y1, self.x2, -self.y2)


class RMoveto(Item):
    def __init__(self, dx, dy):
        self.dx = CAIRO_SCALE*dx
        self.dy = CAIRO_SCALE*dy

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_move_to(self.dx, -self.dy)


class RLineto(Item):
    def __init__(self, dx, dy):
        self.dx = CAIRO_SCALE*dx
        self.dy = CAIRO_SCALE*dy

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_line_to(self.dx, -self.dy)


class RCurveto(Item):
    def __init__(self, dx0, dy0, dx1, dy1, dx2, dy2):
        self.dx0 = CAIRO_SCALE*dx0
        self.dy0 = CAIRO_SCALE*dy0
        self.dx1 = CAIRO_SCALE*dx1
        self.dy1 = CAIRO_SCALE*dy1
        self.dx2 = CAIRO_SCALE*dx2
        self.dy2 = CAIRO_SCALE*dy2

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_curve_to(self.dx0, -self.dy0, 
            self.dx1, -self.dy1, self.dx2, -self.dy2)


class Arc(Item):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in degrees"
        self.x = CAIRO_SCALE*x
        self.y = CAIRO_SCALE*y
        self.r = CAIRO_SCALE*r
        self.angle1 = angle1
        self.angle2 = angle2

    def get_bound(self):
        return Bound(self.x-r, self.y-r, self.x+r, self.y+r) # XXX TODO XXX

    def process_cairo(self, cxt):
        cxt.arc(self.x, -self.y, self.r, 2*pi*self.angle1, 2*pi*self.angle2)


class Arcn(Arc):
    def process_cairo(self, cxt):
        cxt.arc_negative(self.x, -self.y, self.r, 2*pi*self.angle1, 2*pi*self.angle2)


# ----------------------------------------------------------------------------
# Path's : a list of Item's
#

class Path(Base):
    def __init__(self, items):
        for item in items:
            assert isinstance(item, Item)
        self.items = list(items)

    def get_bound(self):
        b = Bound()
        for item in self.items:
            b.update(item.get_bound())
        return b

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


class Deco(Base):
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


def text_extents_cairo(text):
    import cairo
    surface = cairo.PDFSurface("/dev/null", 0, 0)
    #surface = cairo.RecordingSurface("/dev/null", 0, 0) # only in cairo 1.11.0
    cxt = cairo.Context(surface)
    ex = cxt.text_extents(text)
    surface.finish()
    return ex


class Drawable(Base):
    pass


class Text(Drawable):
    def __init__(self, x, y, text, decos=[]):
        for deco in decos:
            assert isinstance(deco, Deco)
        self.x = CAIRO_SCALE*x
        self.y = CAIRO_SCALE*y
        self.text = text
        self.decos = list(decos)

    def get_bound(self):
        extents = text_extents_cairo(self.text)
        (dx, dy, width, height, _, _) = extents
        b = Bound(self.x, self.y, self.x+width, self.y+height) # XXX FIX FIX XXX
        return b

    def process_cairo(self, cxt):
        decos = self.decos
        cxt.save()
        for deco in decos:
            deco.pre_process_cairo(cxt)
        cxt.move_to(self.x, -self.y)
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

    def get_bound(self):
        return self.path.get_bound()

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


class Canvas(Base):
    "A list of Drawable's"
    def __init__(self, draws=[]):
        for draw in draws:
            assert isinstance(draw, Drawable)
        self.draws = list(draws)

    def get_bound(self):
        bound = Bound()
        for draw in self.draws:
            bound.update(draw.get_bound())
        return bound

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
        dx, dy, width, height, _, _ = text_extents_cairo(text)
        return (dx/CAIRO_SCALE, dy/CAIRO_SCALE, width/CAIRO_SCALE, height/CAIRO_SCALE)

    def text(self, x, y, text, decos=[]):
        #print("Canvas.text", x, y, text)
        draw = Text(x, y, text, decos)
        self.append(draw)

    def writePDFfile(self, name):
        #print(self)
        assert name.endswith(".pdf")

        bound = self.get_bound()
        print(bound)

        SCALE = 1.0

        import cairo

        if 0:
            W, H = 200., 200. # point == 1/72 inch
            surface = cairo.PDFSurface(name, W, H)

            dx = 0
            dy = H
            surface.set_device_offset(dx, dy)

        else:
            W = bound.width
            H = bound.height
            surface = cairo.PDFSurface(name, W, H)

            dx = 0 - bound.llx
            dy = H + bound.lly
            surface.set_device_offset(dx, dy)

        cxt = cairo.Context(surface)
        cxt.scale(SCALE, SCALE)
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





