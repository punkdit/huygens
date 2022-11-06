#!/usr/bin/env python3

"""
Expose a _drawing api similar to the pyx api.

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

import os
from tempfile import mkstemp
from math import pi, sqrt, sin, cos, sqrt, floor

from huygens.base import EPSILON, NS, SCALE_CM_TO_POINT
from huygens.back import *
from huygens.back import _defaultlinewidth
from huygens.flatten import Flatten


class BoxDeco(Deco): 
    def __init__(self, box, t=0.5):
        assert 0<=t<=1.
        self.box = box
        self.t = t

    def on_decorate(self, pre, path, post):
        t = self.t
        x, y, dx, dy = path.tangent(t)
        cvs = canvas.canvas()
        self.box.render(cvs, x, y)
        post.append(cvs)


# move this somewhere ... 
class ArrowDeco(Deco): # PathDeco ?
    def __init__(self, astyle="curve", t=1.0, size=0.15, angle=30., 
            round=False, reverse=False, **kw):
        assert 0<=t<=1.
        assert astyle in "hook dart curve feather bar flat".split()
        self.astyle = astyle
        self.t = t
        self.size = size
        self.angle = angle
        self.round = round
        self.reverse = reverse

#    # I think we can just do all this with **kw and __dict__ magic...
#    def __call__(self, astyle=None, t=None, size=None, angle=None, 
#            round=None, reverse=None):
#        astyle = self.astyle if astyle is None else astyle
#        t = self.t if t is None else t
#        size = self.size if size is None else size
#        angle = self.angle if angle is None else angle
#        round = self.round if round is None else round
#        reverse = self.reverse if reverse is None else reverse
#        return ArrowDeco(astyle, t, size, angle, round, reverse)

    def on_decorate(self, pre, path, post):
        from huygens.turtle import Turtle
        assert isinstance(path, Path), "don't know how to decorate %s"%(path,)
        t = self.t
        if self.reverse:
            t = 1.-t
        x, y, dx, dy = path.tangent(t)
        if self.reverse:
            dx, dy = -dx, -dy
        # Ugh, not pretty but it works...
        turtle = Turtle(x, y)
        turtle.lookat(x+dx, y+dy)
        astyle = self.astyle
        turtle.arrow(self.size, self.angle, astyle)
        post.append(style.linestyle.solid) # ???
        if astyle == "dart":
            if self.round:
                post.append(style.linejoin.round)
            post += turtle.mkpath(closepath=True)
            post.append(FillPreserve())
            post.append(Stroke())
        elif astyle == "curve" or astyle == "feather":
            post.append(style.linejoin.round)
            post.append(style.linecap.round)
            post += turtle.mkpath()
            post.append(Stroke())
        else:
            if self.round:
                post.append(style.linejoin.round)
                post.append(style.linecap.round)
            else:
                post.append(style.linejoin.miter)
                post.append(style.linecap.butt)
            post += turtle.mkpath()
            post.append(Stroke())


# ----------------------------------------------------------------------------
# 
#


# Note: arc/arcn use radians, unlike pyx which uses degrees 
path = NS(
    line=Line, curve=Curve, rect=Rect, circle=Circle, path=Path,
    arc=Arc, arcn=Arcn, moveto=MoveTo, lineto=LineTo, 
    curveto=CurveTo, closepath=ClosePath)

# XXX Arc is PathItem, we need a Path for path.arc XXX
# Use arc=arc_to_bezier ?

def polygon(points): # XXX there's also Polygon(Item) class XXX
    assert len(points)
    pth = [path.moveto(*points[0])]
    for point in points[1:]:
        pth.append(path.lineto(*point))
    pth.append(path.closepath())
    pth = path.path(pth)
    return pth

path.polygon = polygon
    

RGB = RGBA
RGB.red = RGB(1., 0., 0.)
RGB.green = RGB(0., 1., 0.)
RGB.blue = RGB(0., 0., 1.)
RGB.white = RGB(1., 1., 1.)
RGB.grey = RGB.gray = RGB(0.8, 0.8, 0.8)
RGB.black = RGB(0., 0., 0.)

def rgbhex(spec):
    assert type(spec) is str
    if spec.startswith("#"):
        spec = spec[1:]
    assert len(spec)==6
    r, g, b = [1.*int(spec[2*i:2*i+2], 16)/255. for i in range(3)]
    return RGB(r, g, b)

color = NS(rgb=RGBA, rgba=RGBA, rgbhex=rgbhex)


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


text = NS(
    size = NS(
        tiny = TextSize(-4, "tiny"),
        script = TextSize(-3, "script"),
        footnote = TextSize(-2, "footnote"),
        small = TextSize(-1, "small"),
        normal = TextSize(0, "normalsize"),
        large = TextSize(1, "large"),
        Large = TextSize(2, "Large"),
        LARGE = TextSize(3, "LARGE"),
        huge = TextSize(4, "huge"),
        Huge = TextSize(5, "Huge")),
    halign = NS(
        left = TextHAlign("left"),
        center = TextHAlign("center"),
        right = TextHAlign("right"),
        clear = TextHAlign("clear"),
        boxleft = TextHAlign("boxleft"),
        boxcenter = TextHAlign("boxcenter"),
        boxright = TextHAlign("boxright"),
        flushleft = TextHAlign("flushleft"),
        flushcenter = TextHAlign("flushcenter"),
        flushright = TextHAlign("flushright")),
    valign = NS(
        top = TextVAlign("top"),
        middle = TextVAlign("middle"),
        bottom = TextVAlign("bottom")))




linestyle = NS(
    solid = CompoundDeco([style.linecap.butt, LineDash([])]),
    dashed = CompoundDeco([style.linecap.butt, LineDash([2])]),
    dotted = CompoundDeco([style.linecap.round, LineDash([0, 2])]),
    dashdotted = CompoundDeco([style.linecap.round, LineDash([0, 2, 2, 2])]))

style.linestyle = linestyle

trafo = NS(translate = Translate, scale = Scale, rotate = Rotate)

_base = 0.1 
deco = NS()
_default_astyle = "curve"
deco.barrow = ArrowDeco(_default_astyle, 0.0)
deco.marrow = ArrowDeco(_default_astyle, 0.5) # mid arrow
deco.earrow = ArrowDeco(_default_astyle, 1.0)
deco.earrow.large = ArrowDeco(_default_astyle, 1.0, _base*sqrt(2))
deco.earrow.Large = ArrowDeco(_default_astyle, 1.0, _base*sqrt(4))
deco.earrow.LArge = ArrowDeco(_default_astyle, 1.0, _base*sqrt(8))
deco.earrow.LARge = ArrowDeco(_default_astyle, 1.0, _base*sqrt(16))
deco.earrow.LARGe = ArrowDeco(_default_astyle, 1.0, _base*sqrt(32))
deco.earrow.LARGE = ArrowDeco(_default_astyle, 1.0, _base*sqrt(64))

#bbox = Bound


class Canvas(Compound):

    def __init__(self, *args, **kw):
        Compound.__init__(self, *args, **kw)

        # Used in diagram.py for recording paths
        # (Lord help me, this is getting hairy.)
        #self.trace = {} 

    def translate(self, x, y):
        self.append(Translate(x, y))
        return self # yes...

    def scale(self, sx, sy=None, x=0., y=0.):
        self.append(Scale(x, y))
        return self # yes...

    def rotate(self, angle, x=0., y=0.):
        self.append(Rotate(angle, x, y))
        return self # yes...

    # Argh! This method already means something else for list's...
    def insert(self, x, y, cvs): 
        cvs = Canvas([Translate(x, y), cvs])
        self.append(cvs)
        return self 

    def stroke(self, path, decos=[], address=None):
        assert type(decos) is list
        assert isinstance(path, Item), repr(path)
        path.address = address
        pre = Compound()
        post = Compound()
        for deco in decos:
            deco.on_decorate(pre, path, post)
        pre.append(path)
        pre.append(Stroke())
        self.append(pre + post)
        return self # yes...

    def fill(self, path, decos=[], address=None):
        assert type(decos) is list
        assert isinstance(path, Item), repr(path)
        #item = Compound(decos, path, Fill())
        #self.append(item)
        #path.address = address
        pre = Compound()
        post = Compound()
        for deco in decos:
            deco.on_decorate(pre, path, post)
        pre.append(path)
        pre.append(Fill())
        self.append(pre + post)
        return self # yes...

    def paint(self, decos=[]):
        item = Compound(decos + [Paint()])
        self.append(item)
        return self

    def clip(self, path):
        self.append(path)
        self.append(Clip())
        return self # yes...

#    def text_extents(self, text):
#        dx, dy, width, height, _, _ = text_extents_cairo(text)
#        return (dx/SCALE_CM_TO_POINT, -dy/SCALE_CM_TO_POINT,  # <-- sign flip
#            width/SCALE_CM_TO_POINT, height/SCALE_CM_TO_POINT)

    def text_extents(self, text):
        item = Text(0., 0., text)
        return item.text_extents()

    def text(self, x, y, text, decos=[]):
        assert type(decos) is list
        # just hack this color deco...
        color = None 
        for deco in decos:
            if isinstance(deco, RGBA): 
                color = deco
        item = Text(0., 0., text, color)
        _, ury, width, height = item.text_extents()
        tr = trafo.translate(x, y)
        pre = Compound([tr])
        post = Compound()
        for deco in decos:
            if not isinstance(deco, RGBA):
                deco.on_decorate(pre, item, post)
        pre.append(item)
        if 0:
            # DEBUG with a rectangle around the text...
            #self.stroke(path.circle(x, y, 0.05))
            p = path.rect(0., ury-height, width, height)
            post.append(p)
            post.append(Stroke())
        self.append(pre + post)
        return self # yes...

    def image(self, name, x=0, y=0):
        im = PNGImage(name, x, y)
        self.append(im)
        return self # yes...

    def _write_cairo(self, method, name=None, trim=0):

        if 0:
            #self.dump()
            #bound = self.get_bound()
            #print("_write_cairo: self.get_bound()", bound)
    
            cxt = Flatten()
            self.process_cairo(cxt)
            item = Compound(cxt.paths)
            #print("Flatten:")
            #item.dump()
            bound = item.get_bound()
            #print("_write_cairo: item.get_bound()", bound)
            assert not bound.is_empty()

        else:
            bound = self.get_bound_cairo()
        #print("_write_cairo", bound)

        import cairo

        W = bound.width - 2*trim
        H = bound.height - 2*trim
        surface = method(name, W, H)

        dx = 0 - bound.llx - trim
        dy = H + bound.lly + trim
        surface.set_device_offset(dx, dy)

        cxt = cairo.Context(surface)
        cxt.set_line_width(_defaultlinewidth * SCALE_CM_TO_POINT)
        self.process_cairo(cxt)
        #item.process_cairo(cxt)
        return surface

    def writePDFfile(self, name):
        if name=="/dev/null" or name.endswith(".pdf"):
            pass
        else:
            name = name + ".pdf"
        import cairo
        method = cairo.PDFSurface
        surface = self._write_cairo(method, name)
        surface.finish()

    def writeSVGfile(self, name):
        if name=="/dev/null" or name.endswith(".svg"):
            pass
        else:
            name = name + ".svg"
        import cairo
        method = cairo.SVGSurface
        surface = self._write_cairo(method, name)
        surface.finish()

    def writePNGfile(self, name, trim=1):
        if name=="/dev/null" or name.endswith(".png"):
            pass
        else:
            name = name + ".png"
        import cairo
        def method(name, W, H):
            W = int(round(W))
            H = int(round(H))
            #surface = cairo.ImageSurface(cairo.Format.RGB24, W, H)
            surface = cairo.ImageSurface(cairo.Format.ARGB32, W, H)
            return surface
        surface = self._write_cairo(method, name, trim)
        surface.write_to_png(name)
        surface.finish()

    def get_data(self):
        import cairo, numpy
        def method(name, W, H):
            W = int(round(W))
            H = int(round(H))
            surface = cairo.ImageSurface(cairo.Format.ARGB32, W, H)
            return surface
        surface = self._write_cairo(method)
        width = surface.get_width()
        height = surface.get_height()
        buf = surface.get_data()
        data = numpy.ndarray(shape=(height, width, 4), dtype=numpy.uint8, buffer=buf)
        data = data.copy()
        surface.finish()
        return data

    # svg output does not work on github  :P 
    def _repr_svg_(self):
        fd, name = mkstemp(suffix=".svg")
        self.writeSVGfile(name)
        os.close(fd)
        s = open(name).read()
        return s
    
    def _repr_png_(self):
        fd, name = mkstemp(suffix=".png")
        self.writePNGfile(name)
        os.close(fd)
        s = open(name, 'rb').read()
        return s
    


canvas = NS(canvas=Canvas)



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


def test():

    cvs = canvas.canvas()

    st_thin = [style.linewidth.thin]

    cvs.stroke(path.line(0,    2, 0,    3), st_thin+[ArrowDeco("curve")])
    cvs.stroke(path.line(-0.5, 2, -0.5, 3), st_thin+[ArrowDeco("feather")])
    cvs.stroke(path.line(-1.0, 2, -1.0, 3), st_thin+[ArrowDeco("dart")])

    attrs = st_thin+[
        ArrowDeco("dart", 0.1),
        ArrowDeco("curve", 0.3),
        ArrowDeco("feather", 0.6),
        ArrowDeco("bar", 0.8),
        ArrowDeco("flat"),
    ]

    p = path.rect(-1, 0, 1, 1)
    cvs.stroke(p, attrs)

    tr = trafo.translate(-0.3, 0.5)
    attrs.append(ArrowDeco("hook", 0.))

    p = path.curve(3, 0, 1, 0, 2, 2, 1, 3)
    cvs.stroke(p, [tr]+attrs)

    tr = trafo.translate(0.5, 0)
    p = path.path([
        path.moveto(2.5, 2),
        path.arcn(2, 2, 0.5, 0., 0.5*pi)])
    cvs.stroke(p, [tr, color.rgb.red]+attrs)

    cvs.writePDFfile("output-arrow.pdf")




if __name__ == "__main__":
    test()

    print("OK")





