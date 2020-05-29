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

from math import pi, sqrt


# simple namespace class
class NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


SCALE_CM_TO_POINT = 72.0/2.54 # convert cm's to points at 72 dpi.


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


class MoveTo_Pt(Item):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.move_to(self.x, -self.y)


class MoveTo(MoveTo_Pt):
    def __init__(self, x, y):
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y


class LineTo_Pt(Item):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.line_to(self.x, -self.y)


class LineTo(LineTo_Pt):
    def __init__(self, x, y):
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y


class CurveTo_Pt(Item):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        cxt.curve_to(self.x0, -self.y0, self.x1, -self.y1, self.x2, -self.y2)


class CurveTo(CurveTo_Pt):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = SCALE_CM_TO_POINT*x0
        self.y0 = SCALE_CM_TO_POINT*y0
        self.x1 = SCALE_CM_TO_POINT*x1
        self.y1 = SCALE_CM_TO_POINT*y1
        self.x2 = SCALE_CM_TO_POINT*x2
        self.y2 = SCALE_CM_TO_POINT*y2


class RMoveTo(Item):
    def __init__(self, dx, dy):
        self.dx = SCALE_CM_TO_POINT*dx
        self.dy = SCALE_CM_TO_POINT*dy

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_move_to(self.dx, -self.dy)


class RLineTo(Item):
    def __init__(self, dx, dy):
        self.dx = SCALE_CM_TO_POINT*dx
        self.dy = SCALE_CM_TO_POINT*dy

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_line_to(self.dx, -self.dy)


class RCurveTo(Item):
    def __init__(self, dx0, dy0, dx1, dy1, dx2, dy2):
        self.dx0 = SCALE_CM_TO_POINT*dx0
        self.dy0 = SCALE_CM_TO_POINT*dy0
        self.dx1 = SCALE_CM_TO_POINT*dx1
        self.dy1 = SCALE_CM_TO_POINT*dy1
        self.dx2 = SCALE_CM_TO_POINT*dx2
        self.dy2 = SCALE_CM_TO_POINT*dy2

    def get_bound(self):
        assert 0, "???"

    def process_cairo(self, cxt):
        cxt.rel_curve_to(self.dx0, -self.dy0, 
            self.dx1, -self.dy1, self.dx2, -self.dy2)


class Arc(Item):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in degrees"
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y
        self.r = SCALE_CM_TO_POINT*r
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
            MoveTo(x0, y0), 
            LineTo(x1, y1)])


class Curve(Path):
    def __init__(self, x0, y0, x1, y1, x2, y2, x3, y3):
        Path.__init__(self, [
            MoveTo(x0, y0), 
            CurveTo(x1, y1, x2, y2, x3, y3)])
    

class Rect(Path):
    def __init__(self, x, y, width, height):
        Path.__init__(self, [
            MoveTo(x, y), 
            LineTo(x+width, y),
            LineTo(x+width, y+height),
            LineTo(x, y+height),
            ClosePath()])


class Circle(Path):
    def __init__(self, x, y, r):
        Path.__init__(self, [
            MoveTo(x+r, y),
            Arc(x, y, r, 0, 360),
            ClosePath()])
        

path = NS(line = Line, curve = Curve, rect = Rect, circle = Circle)


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


class FillPreserve(Deco):
    def post_process_cairo(self, cxt):
        cxt.fill_preserve()


class RGBA(Deco):
    def __init__(self, r, g, b, a=1.0):
        self.cl = (r, g, b, a)

    def pre_process_cairo(self, cxt):
        cxt.set_source_rgba(*self.cl)

RGB = RGBA


class LineWidth_Pt(Deco):
    def __init__(self, w):
        self.w = w

    def pre_process_cairo(self, cxt):
        cxt.set_line_width(self.w)


class LineWidth(LineWidth_Pt):
    def __init__(self, w):
        self.w = w*SCALE_CM_TO_POINT


# cairo constants:
#    cairo.LINE_CAP_BUTT           
#    cairo.LINE_CAP_ROUND         
#    cairo.LINE_CAP_SQUARE       
#    cairo.LINE_JOIN_BEVEL      
#    cairo.LINE_JOIN_MITER     
#    cairo.LINE_JOIN_ROUND    


class LineCap(Deco):
    def __init__(self, desc):
        assert type(desc) is str
        assert desc.lower() in "butt round square".split()
        self.desc = desc.upper()

    def pre_process_cairo(self, cxt):
        import cairo
        cap = getattr(cairo, "LINE_CAP_%s"%(self.desc,))
        cxt.set_line_cap(cap)


class LineJoin(Deco):
    def __init__(self, desc):
        assert type(desc) is str
        assert desc.lower() in "bevel miter round".split()
        self.desc = desc.upper()

    def pre_process_cairo(self, cxt):
        import cairo
        cap = getattr(cairo, "LINE_JOIN_%s"%(self.desc,))
        cxt.set_line_join(cap)


_defaultlinewidth = 0.02 # cm

style = NS(
    linecap = NS(
        butt = LineCap("butt"),
        round = LineCap("round"), 
        square = LineCap("square")),
    linejoin = NS(
        bevel = LineJoin("bevel"),
        miter = LineJoin("miter"),
        round = LineJoin("round")),
    linewidth = NS(
        THIN = LineWidth(_defaultlinewidth/sqrt(32)),
        THIn = LineWidth(_defaultlinewidth/sqrt(16)),
        THin = LineWidth(_defaultlinewidth/sqrt(8)),
        Thin = LineWidth(_defaultlinewidth/sqrt(4)),
        thin = LineWidth(_defaultlinewidth/sqrt(2)),
        normal = LineWidth(_defaultlinewidth),
        thick = LineWidth(_defaultlinewidth*sqrt(2)),
        Thick = LineWidth(_defaultlinewidth*sqrt(4)),
        THick = LineWidth(_defaultlinewidth*sqrt(8)),
        THIck = LineWidth(_defaultlinewidth*sqrt(16)),
        THICk = LineWidth(_defaultlinewidth*sqrt(32)),
        THICK = LineWidth(_defaultlinewidth*sqrt(64)),
    ))


class TextSize(Deco):
    def __init__(self, size):
        #self.size = float(size) * SCALE_CM_TO_POINT
        self.size = size # ???

    def pre_process_cairo(self, cxt):
        pass
        #cxt.set_font_size(self.size)


class TextAlign(Deco):
    def __init__(self, desc):
        self.desc = desc


text = NS(
    size = NS(
        tiny = TextSize(-4),
        script = TextSize(-3),
        footnote = TextSize(-2),
        small = TextSize(-1),
        normal = TextSize(0),
        large = TextSize(1),
        Large = TextSize(2),
        LARGE = TextSize(3),
        huge = TextSize(4),
        Huge = TextSize(5)),
    halign = NS(
        left = TextAlign("left"),
        center = TextAlign("center"),
        right = TextAlign("right"),
        clear = TextAlign("clear"),
        boxleft = TextAlign("boxleft"),
        boxcenter = TextAlign("boxcenter"),
        boxright = TextAlign("boxright"),
        flushleft = TextAlign("flushleft"),
        flushcenter = TextAlign("flushcenter"),
        flushright = TextAlign("flushright")),
    valign = NS(
        top = TextAlign("top"),
        middle = TextAlign("middle"),
        bottom = TextAlign("bottom")))



#linestyle.solid = linestyle(linecap.butt, dash([]))
#linestyle.dashed = linestyle(linecap.butt, dash([2]))
#linestyle.dotted = linestyle(linecap.round, dash([0, 2]))
#linestyle.dashdotted = linestyle(linecap.round, dash([0, 2, 2, 2]))


class Translate(Deco):
    def __init__(self, dx, dy):
        self.dx = dx*SCALE_CM_TO_POINT
        self.dy = dy*SCALE_CM_TO_POINT

    def pre_process_cairo(self, cxt):
        cxt.translate(self.dx, self.dy)


class Scale(Deco):
    def __init__(self, sx, sy):
        self.sx = float(sx)
        self.sy = float(sy)

    def pre_process_cairo(self, cxt):
        cxt.scale(self.sx, self.sy)


trafo = NS(translate = Translate, scale = Scale)





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
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y
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
        return (dx/SCALE_CM_TO_POINT, dy/SCALE_CM_TO_POINT, 
            width/SCALE_CM_TO_POINT, height/SCALE_CM_TO_POINT)

    def text(self, x, y, text, decos=[]):
        #print("Canvas.text", x, y, text)
        draw = Text(x, y, text, decos)
        self.append(draw)

    def writePDFfile(self, name):
        #print(self)
        assert name.endswith(".pdf")

        bound = self.get_bound()
        #print(bound)

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
        cxt.set_line_width(_defaultlinewidth * SCALE_CM_TO_POINT)
        for draw in self.draws:
            draw.process_cairo(cxt)
        surface.finish()

    def writeSVGfile(self, name):
        assert name.endswith(".svg")
        surface = cairo.SVGSurface(name, W, H)





# ----------------------------------------------------------------------------
# test
#

def test():

    cvs = Canvas()

    cvs.stroke(path.line(10, 10, 50, 50))

    cvs.writePDFfile("output.pdf")



if __name__ == "__main__":
    test()





