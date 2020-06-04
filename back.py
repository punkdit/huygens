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

from math import pi, sqrt, sin, cos, sqrt, floor

from bruhat.render.base import EPSILON, NS, SCALE_CM_TO_POINT, Base, Matrix
from bruhat.render.text import make_text


# ----------------------------------------------------------------------------
# 
#

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
        assert urx is None or llx is None or urx-llx >= 0., (llx, lly, urx, ury)
        assert ury is None or lly is None or ury-lly >= 0., (llx, lly, urx, ury)
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

    def nonempty(self):
        return (self.llx is not None or self.lly is not None or 
            self.urx is not None or self.ury is not None)

    def is_empty(self):
        return (self.llx is None and self.lly is None and 
            self.urx is None and self.ury is None)

    def __getitem__(self, idx):
        return (self.llx, self.lly, self.urx, self.ury)[idx]

    @property
    def width(self):
        return self.urx - self.llx

    @property
    def height(self):
        return self.ury - self.lly




# ----------------------------------------------------------------------------
# 
#


class Visitor(object):
    def on_item(self, item):
        pass

class DumpVisitor(Visitor):
    def on_item(self, item):
        print('\t%s'%item)


# XXX this probably should be a full fledged Context XXX
class BoundVisitor(Visitor):
    def __init__(self):
        self.pos = None
        self.lw = _defaultlinewidth*SCALE_CM_TO_POINT
        self.bound = Bound()

    def on_item(self, item):
        tp = item.__class__.__name__ # ARGGGHHH!!!
        if tp == "Translate_Pt":
            assert 0, "%s not implemented"%item
        elif tp == "Scale":
            assert 0, "%s not implemented"%item
        elif tp in "Compound Path": # save, restore not implemented
            assert 0, "%s not implemented"%item
        elif tp == "MoveTo_Pt":
            self.pos = item.x, item.y
        elif tp in "Stroke Fill":
            self.pos = None
        elif tp in "LineWidth_Pt":
            self.lw = item.lw
        elif tp in "LineTo_Pt CurveTo_Pt":
            b = item.get_bound()
            if b.is_empty():
                assert 0
            # add a generous linewidth... does not work very well... XXX
            r = 0.5*self.lw
            b.llx -= r
            b.lly -= r
            b.urx += r
            b.ury += r
            self.bound.update(b)
            if self.pos is not None:
                x, y = self.pos
                self.bound.update(Bound(x, y, x, y))
        else:
            pass
        #print("BoundVisitor.on_item", self.bound)



# ----------------------------------------------------------------------------
# 
#


class Item(Base):

    DEBUG = False

    def get_bound(self):
        return Bound()

    def process_cairo(self, cxt):
        pass

    def visit(self, visitor):
        visitor.on_item(self)

    def dump(self):
        visitor = DumpVisitor()
        self.visit(visitor)



class ClosePath(Item):
    def process_cairo(self, cxt):
        cxt.close_path()


class MoveTo_Pt(Item):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        if self.DEBUG:
            print("ctx.move_to", self.x, self.y)
        cxt.move_to(self.x, -self.y)


class MoveTo(MoveTo_Pt):
    def __init__(self, x, y):
        x, y = float(x), float(y)
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y


class LineTo_Pt(Item):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_bound(self):
        return Bound(self.x, self.y, self.x, self.y)

    def process_cairo(self, cxt):
        if self.DEBUG:
            print("ctx.line_to", self.x, self.y)
        cxt.line_to(self.x, -self.y)


class LineTo(LineTo_Pt):
    def __init__(self, x, y):
        x, y = float(x), float(y)
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
        x0 = min([self.x0, self.x1, self.x2])
        x1 = max([self.x0, self.x1, self.x2])
        y0 = min([self.y0, self.y1, self.y2])
        y1 = max([self.y0, self.y1, self.y2])
        return Bound(x0, y0, x1, y1)

    def process_cairo(self, cxt):
        if self.DEBUG:
            print("ctx.curve_to", self.x0, self.y0, self.x1, self.y1, self.x2, self.y2)
        cxt.curve_to(self.x0, -self.y0, self.x1, -self.y1, self.x2, -self.y2)


class CurveTo(CurveTo_Pt):
    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = SCALE_CM_TO_POINT*x0
        self.y0 = SCALE_CM_TO_POINT*y0
        self.x1 = SCALE_CM_TO_POINT*x1
        self.y1 = SCALE_CM_TO_POINT*y1
        self.x2 = SCALE_CM_TO_POINT*x2
        self.y2 = SCALE_CM_TO_POINT*y2


class Arc_Pt(Item):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in radians"
        self.x = x
        self.y = y
        self.r = r
        self.angle1 = angle1
        self.angle2 = angle2

    def get_bound(self):
        r = self.r
        return Bound(self.x-r, self.y-r, self.x+r, self.y+r) # XXX TODO XXX

    def process_cairo(self, cxt):
        cxt.arc_negative(self.x, -self.y, self.r, 2*pi-self.angle1, 2*pi-self.angle2)


class Arc(Arc_Pt):
    def __init__(self, x, y, r, angle1, angle2):
        "angle in radians"
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y
        self.r = SCALE_CM_TO_POINT*r
        self.angle1 = angle1
        self.angle2 = angle2


class _ArcnMixin(object):
    def process_cairo(self, cxt):
        cxt.arc(self.x, -self.y, self.r, 2*pi-self.angle1, 2*pi-self.angle2)


class Arcn(Arc, _ArcnMixin):
    pass


class Arcn_Pt(Arc_Pt, _ArcnMixin):
    pass


# ----------------------------------------------------------------------------
#
#

class Compound(Item):
    def __init__(self, *args):
        items = []
        for arg in args:
            if type(arg) is list:
                items += arg
            else:
                items.append(arg)
        self.items = items

    def append(self, item):
        items = self.items
        items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def visit(self, visitor):
        for item in self.items:
            item.visit(visitor)

    def get_bound(self):
        visitor = BoundVisitor()
        self.visit(visitor)
        return visitor.bound

    def get_bound_cairo(self):
        import cairo
        surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        cxt = cairo.Context(surface)
        self.process_cairo(cxt)
        extents = surface.ink_extents()
        (ulx, uly, width, height) = extents
        llx, lly = ulx, -uly-height
        urx, ury = llx+width, lly+height
        return Bound(llx, lly, urx, ury)

    def process_cairo(self, cxt):
        cxt.save()
        for item in self.items:
            item.process_cairo(cxt)
        cxt.restore()


class Path(Compound):
    def reversed(self, pos=None):
        if len(self)==0:
            return self
        #print("Path.reversed")
        #for item in self:
        #    print('\t', item)
        n = len(self)
        assert n%2 == 0, "not implemented"
        items = []
        idx = n-2
        while idx>=0:
            item = self[idx]
            assert isinstance(item, LineTo_Pt), "not implemented"
            x, y = item.x, item.y
            item = self[idx+1]
            assert isinstance(item, CurveTo_Pt), "not implemented"
            x0, y0 = item.x0, item.y0
            x1, y1 = item.x1, item.y1
            x2, y2 = item.x2, item.y2
            items.append(LineTo_Pt(x2, y2))
            items.append(CurveTo_Pt(x1, y1, x0, y0, x, y))
            idx -= 2
        p = Path(items)
        return p



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
            Arc(x, y, r, 0, 2*pi),
            ClosePath()])
        

path = NS(
    line=Line, curve=Curve, rect=Rect, circle=Circle, path=Path,
    arc=Arc, arcn=Arcn, moveto=MoveTo, lineto=LineTo, 
    curveto=CurveTo, closepath=ClosePath)


# ----------------------------------------------------------------------------
# Deco
#


class Deco(Item):
    pass


class Stroke(Deco):
    def process_cairo(self, cxt):
        cxt.stroke()


class Fill(Deco):
    def process_cairo(self, cxt):
        cxt.fill()


class FillPreserve(Deco):
    def process_cairo(self, cxt):
        cxt.fill_preserve()


class RGBA(Deco):
    def __init__(self, r, g, b, a=1.0):
        self.cl = (r, g, b, a)

    def process_cairo(self, cxt):
        cxt.set_source_rgba(*self.cl)

RGB = RGBA
RGB.red = RGB(1., 0., 0.)
RGB.green = RGB(0., 1., 0.)
RGB.blue = RGB(0., 0., 1.)
RGB.white = RGB(1., 1., 1.)
RGB.black = RGB(0., 0., 0.)

color = NS(rgb=RGBA)


class LineWidth_Pt(Deco):
    def __init__(self, lw):
        self.lw = lw

    def process_cairo(self, cxt):
        cxt.set_line_width(self.lw)


class LineWidth(LineWidth_Pt):
    def __init__(self, lw):
        self.lw = lw*SCALE_CM_TO_POINT


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

    def process_cairo(self, cxt):
        import cairo
        cap = getattr(cairo, "LINE_CAP_%s"%(self.desc,))
        cxt.set_line_cap(cap)


class LineJoin(Deco):
    def __init__(self, desc):
        assert type(desc) is str
        assert desc.lower() in "bevel miter round".split()
        self.desc = desc.upper()

    def process_cairo(self, cxt):
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

    def process_cairo(self, cxt):
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


class Translate_Pt(Deco):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def process_cairo(self, cxt):
        cxt.translate(self.dx, -self.dy)


class Translate(Translate_Pt):
    def __init__(self, dx, dy):
        self.dx = dx*SCALE_CM_TO_POINT
        self.dy = dy*SCALE_CM_TO_POINT


class Scale(Deco):
    def __init__(self, sx, sy=None):
        if sy is None:
            sy = sx
        self.sx = float(sx)
        self.sy = float(sy)

    def process_cairo(self, cxt):
        cxt.scale(self.sx, self.sy)


trafo = NS(translate = Translate, scale = Scale)


# ----------------------------------------------------------------------------
# 
#


def text_extents_cairo(text):
    import cairo
    surface = cairo.PDFSurface("/dev/null", 0, 0)
    # only in cairo 1.11.0:
    #surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    cxt = cairo.Context(surface)
    ex = cxt.text_extents(text)
    surface.finish()
    return ex


class CairoText(Item):
    def __init__(self, x, y, text):
        self.x = SCALE_CM_TO_POINT*x
        self.y = SCALE_CM_TO_POINT*y
        self.text = text

    def get_bound(self):
        extents = text_extents_cairo(self.text)
        (dx, dy, width, height, _, _) = extents
        b = Bound(self.x, self.y, self.x+width, self.y+height) # XXX FIX FIX XXX
        return b

    def process_cairo(self, cxt):
        cxt.save()
        cxt.move_to(self.x, -self.y)
        cxt.show_text(self.text)
        cxt.restore()


class Text(Compound):

    _baseline = None
    @classmethod
    def _get_baseline(cls):
        if cls._baseline is None:
            item = make_text("x") # measure this font using "x"
            bound = item.get_bound()
            #print("_get_baseline", bound)
            cls._baseline = bound.lly
        return cls._baseline

    def __init__(self, x, y, text):
        assert text
        self.x = x = SCALE_CM_TO_POINT*x
        self.y = y = SCALE_CM_TO_POINT*y
        self.text = text
        item = make_text(text)
        bound = item.get_bound()
        assert not bound.is_empty(), bound
        llx, lly = bound.llx, bound.lly
        #print("Text.__init__", x, y, bound)
        y0 = self._get_baseline()
        self.bot = y0 - bound.lly
        trafo = Translate_Pt(x-llx, y-lly-self.bot)
        #print("Text.__init__ trafo:", trafo)
        self.bound = Bound(x, y-self.bot, x+bound.width, y+bound.height-self.bot)
        items = list(item.items)
        items = [trafo] + items
        Compound.__init__(self, items)

    def get_bound(self):
        return self.bound



class Canvas(Compound):

    def stroke(self, path, decos=[]):
        item = Compound(decos, path, Stroke())
        self.append(item)

    def fill(self, path, decos=[]):
        item = Compound(decos, path, Fill())
        self.append(item)

#    def text_extents(self, text):
#        dx, dy, width, height, _, _ = text_extents_cairo(text)
#        return (dx/SCALE_CM_TO_POINT, -dy/SCALE_CM_TO_POINT,  # <-- sign flip
#            width/SCALE_CM_TO_POINT, height/SCALE_CM_TO_POINT)

    def text_extents(self, text):
        item = Text(0., 0., text)
        #item.dump()
        bound = item.get_bound()
        #print("text_extents", text, bound)
        #dx, dy = bound.urx, bound.ury
        llx, lly, urx, ury = bound
        #return (0., dy/SCALE_CM_TO_POINT, dx/SCALE_CM_TO_POINT, dy/SCALE_CM_TO_POINT)
        llx /= SCALE_CM_TO_POINT
        lly /= SCALE_CM_TO_POINT
        urx /= SCALE_CM_TO_POINT
        ury /= SCALE_CM_TO_POINT
        #print("text_extents", text, (0., ury, urx-llx, ury-lly))
        return (0., ury, urx-llx, ury-lly)

    def text(self, x, y, text, decos=[]):
        #print("Canvas.text", x, y, text)
        item = Compound(decos, Text(x, y, text))
        self.append(item)

    def _write_cairo(self, method, name):

        #self.dump()
        #bound = self.get_bound()
        #print("_write_cairo: self.get_bound()", bound)

        from bruhat.render.flatten import Flatten
        cxt = Flatten()
        self.process_cairo(cxt)
        item = Compound(cxt.paths)
        #print("Flatten:")
        #item.dump()
        bound = item.get_bound()
        #print("_write_cairo: item.get_bound()", bound)
        assert not bound.is_empty()

        import cairo

        W = bound.width
        H = bound.height
        surface = method(name, W, H)

        dx = 0 - bound.llx
        dy = H + bound.lly
        surface.set_device_offset(dx, dy)

        cxt = cairo.Context(surface)
        cxt.set_line_width(_defaultlinewidth * SCALE_CM_TO_POINT)
        self.process_cairo(cxt)
        #item.process_cairo(cxt)
        surface.finish()

    def writePDFfile(self, name):
        assert name.endswith(".pdf")
        import cairo
        method = cairo.PDFSurface
        self._write_cairo(method, name)

    def writeSVGfile(self, name):
        assert name.endswith(".svg")
        import cairo
        method = cairo.SVGSurface
        self._write_cairo(method, name)


canvas = NS(canvas=Canvas)



# ----------------------------------------------------------------------------
# Some good code copied from PyX
#


def arc_to_curve_pt(x_pt, y_pt, r_pt, angle1, angle2):
    dangle = angle2-angle1

    if dangle==0:
        return None

    x0_pt, y0_pt = x_pt+r_pt*cos(angle1), y_pt+r_pt*sin(angle1)
    x3_pt, y3_pt = x_pt+r_pt*cos(angle2), y_pt+r_pt*sin(angle2)

    l = r_pt*4*(1-cos(dangle/2))/(3*sin(dangle/2))

    x1_pt, y1_pt = x0_pt-l*sin(angle1), y0_pt+l*cos(angle1)
    x2_pt, y2_pt = x3_pt+l*sin(angle2), y3_pt-l*cos(angle2)

    items = [
        LineTo_Pt(x0_pt, y0_pt), 
        CurveTo_Pt(x1_pt, y1_pt, x2_pt, y2_pt, x3_pt, y3_pt)]
    return items


def arc_to_bezier(x, y, r, angle1, angle2, danglemax=0.5*pi):
    x_pt = x*SCALE_CM_TO_POINT
    y_pt = y*SCALE_CM_TO_POINT
    r_pt = r*SCALE_CM_TO_POINT
    p = arc_to_bezier_pt(x_pt, y_pt, r_pt, angle1, angle2, danglemax)
    return p


def arc_to_bezier_pt(x_pt, y_pt, r_pt, angle1, angle2, danglemax=0.5*pi):
    if angle2<angle1:
        angle2 = angle2 + (floor((angle1-angle2)/(2*pi))+1)*2*pi
    elif angle2>angle1+2*pi:
        angle2 = angle2 - (floor((angle2-angle1)/(2*pi))-1)*2*pi

    if r_pt == 0 or angle1-angle2 == 0:
        return []

    subdivisions = int((angle2-angle1)/danglemax)+1

    dangle = (angle2-angle1)/subdivisions

    items = []
    for i in range(subdivisions):
        items += arc_to_curve_pt(x_pt, y_pt, r_pt, angle1+i*dangle, angle1+(i+1)*dangle)

    p = path.path(items)
    return p


# ----------------------------------------------------------------------------
#
#


def test():

    cvs = canvas.canvas()

    def cross(x, y):
        r = 0.1
        st = [color.rgb.blue, style.linewidth.THick, style.linecap.round]
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), st)
        cvs.stroke(path.line(x-r, y+r, x+r, y-r), st)

    p = path.path([
        path.moveto(0., 0.),
        path.arc(0., 0., 1., 0., 0.5*pi),
        path.lineto(-1., 1.),
        path.arc(-1., 0., 1., 0.5*pi, 1.0*pi),
        path.arc(-1.5, 0., 0.5, 1.0*pi, 2.0*pi),
        path.closepath()
    ])

    items = (
    [ 
        path.moveto(0., 0.),
        path.arc(0., 0., 1., 0., 0.5*pi),
        path.lineto(-1., 1.), path.arc(-1., 0., 1., 0.5*pi, 1.0*pi),
        path.arc(-1.5, 0., 0.5, 1.0*pi, 2.0*pi), path.closepath() ])
    p = path.path(items)

    cvs.fill(p, [color.rgb.red, trafo.scale(0.8, 0.8)])
    cvs.stroke(p, [color.rgb.black, style.linewidth.THick])

    cross(0., 0.)
    cross(-1.2, 1.2)

    if 0:
        x, y, r, angle1, angle2 = 0., 0., 1., 0., 0.5*pi
        p = arc_to_bezier(x, y, r, angle1, angle2, danglemax=pi/2.)
        cvs.stroke(p, [color.rgb.white])
    
        x, y, r, angle1, angle2 = 0., 0., 1., -0.5*pi, 0.
        p = arc_to_bezier(x, y, r, angle1, angle2, danglemax=pi/2.)
        cvs.stroke(p, [color.rgb.red])

    cvs.writePDFfile("output.pdf")

    print("OK")


def test():

    cvs = canvas.canvas()

    def cross(x, y):
        r = 0.1
        st = [color.rgb.blue, style.linewidth.normal, style.linecap.round]
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), st)
        cvs.stroke(path.line(x-r, y+r, x+r, y-r), st)

    #cvs.append(Translate(1., 1.))
    cross(0., 0.)

    cvs.text(0., 0., "hey there!")

    cvs.writePDFfile("output.pdf")

    print("OK\n")


if __name__ == "__main__":
    test()





