#!/usr/bin/env python3

"""
Interpret all transforms, thereby flattening cairo operations
to just a few primitives, such as move_to line_to and curve_to.
"""

import sys
from math import sin, cos, pi

from huygens import argv
from huygens import back
from huygens.base import Context, SCALE_CM_TO_POINT

# Internally huygens uses cm units, even though we are exposing a cairo interface:

class Flatten(Context):

    def __init__(self):
        Context.__init__(self)
        self.path = back.Compound()
        self.paths = []

    def move_to(self, x, y):
        x = x/SCALE_CM_TO_POINT # scale to cm units
        y = y/SCALE_CM_TO_POINT # scale to cm units
        x, y = self.matrix(x, y)
        item = back.MoveTo(x, -y)
        self.path.append(item)
        self.pos = x, y

    def line_to(self, x, y):
        x = x/SCALE_CM_TO_POINT # etc..
        y = y/SCALE_CM_TO_POINT
        x, y = self.matrix(x, y)
        item = back.LineTo(x, -y)
        self.path.append(item)
        self.pos = x, y

    def curve_to(self, x0, y0, x1, y1, x2, y2):
        #print("curve_to", x0, y0, x1, y1, x2, y2)
        x0 = x0/SCALE_CM_TO_POINT
        y0 = y0/SCALE_CM_TO_POINT
        x1 = x1/SCALE_CM_TO_POINT
        y1 = y1/SCALE_CM_TO_POINT
        x2 = x2/SCALE_CM_TO_POINT
        y2 = y2/SCALE_CM_TO_POINT
        #print("\t", x0, y0, x1, y1, x2, y2)
        x0, y0 = self.matrix(x0, y0)
        x1, y1 = self.matrix(x1, y1)
        x2, y2 = self.matrix(x2, y2)
        item = back.CurveTo(x0, -y0, x1, -y1, x2, -y2)
        self.path.append(item)
        self.pos = x2, y2

    def rel_move_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        dx = dx/SCALE_CM_TO_POINT
        dy = dy/SCALE_CM_TO_POINT
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        x, y = x+dx, y+dy
        item = back.MoveTo(x, -y)
        self.path.append(item)
        self.pos = x, y

    def rel_line_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        dx = dx/SCALE_CM_TO_POINT
        dy = dy/SCALE_CM_TO_POINT
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        x, y = x+dx, y+dy
        item = back.LineTo(x, -y)
        self.path.append(item)
        self.pos = x, y

    def rel_curve_to(self, dx0, dy0, dx1, dy1, dx2, dy2):
        assert self.pos is not None, "no current point"
        dx0 = dx0/SCALE_CM_TO_POINT
        dy0 = dy0/SCALE_CM_TO_POINT
        dx1 = dx1/SCALE_CM_TO_POINT
        dy1 = dy1/SCALE_CM_TO_POINT
        dx2 = dx2/SCALE_CM_TO_POINT
        dy2 = dy2/SCALE_CM_TO_POINT
        x, y = self.pos
        dx0, dy0 = self.matrix.transform_distance(dx0, dy0)
        dx1, dy1 = self.matrix.transform_distance(dx1, dy1)
        dx2, dy2 = self.matrix.transform_distance(dx2, dy2)
        x0, y0 = x+dx0, y+dy0
        x1, y1 = x+dx1, y+dy1
        x2, y2 = x+dx2, y+dy2
        item = back.CurveTo(x0, -y0, x1, -y1, x2, -y2)
        self.path.append(item)
        self.pos = x2, y2

    def arc(self, x, y, radius, angle1, angle2):
        # stay in user space coordinates
        x = x/SCALE_CM_TO_POINT
        y = y/SCALE_CM_TO_POINT
        radius = radius/SCALE_CM_TO_POINT
        if self.pos is None:
            x1, y1 = x+radius*cos(angle1), y+radius*sin(angle1)
            self.move_to(x1, y1)
        p = back.arc_to_bezier(x, -y, radius, -angle2, -angle1)
        p = p.reversed()
        p.process_cairo(self)

    def arc_negative(self, x, y, radius, angle1, angle2):
        # stay in user space coordinates
        x = x/SCALE_CM_TO_POINT
        y = y/SCALE_CM_TO_POINT
        radius = radius/SCALE_CM_TO_POINT
        if self.pos is None:
            x1, y1 = x+radius*cos(angle1), y+radius*sin(angle1)
            self.move_to(x1, y1)
        p = back.arc_to_bezier(x, -y, radius, -angle1, -angle2)
        p.process_cairo(self)

    def close_path(self):
        item = back.ClosePath()
        self.path.append(item)

    def set_source_rgba(self, r, g, b, a):
        deco = back.RGBA(r, g, b, a)
        self.path.append(deco)

    def set_line_width(self, w):
        w /= SCALE_CM_TO_POINT
        wx, wy = self.matrix.transform_distance(w, w)
        w = wx
        deco = back.LineWidth(w)
        self.path.append(deco)

    def stroke(self):
        deco = back.Stroke()
        self.path.append(deco)
        self.paths.append(self.path)
        self.path = back.Compound()
        self.pos = None

    def fill_preserve(self):
        deco = back.FillPreserve()
        self.path.append(deco)

    def fill(self):
        deco = back.Fill()
        self.path.append(deco)
        self.paths.append(self.path)
        self.path = back.Compound()
        self.pos = None

    def set_line_cap(self, arg):
        import cairo
        for desc in "butt round square".split():
            if getattr(cairo, "LINE_CAP_"+(desc.upper())) == arg:
                self.path.append(back.LineCap(desc))
                break
        else:
            assert 0

    def ignore(self, *args, **kw):
        pass

    def set_fill_rule(self, arg):
        print("TODO: Flatten.set_fill_rule", arg)
        self.set_fill_rule = self.ignore


def test():

    def draw_test(cxt):
        cxt.move_to(10, 10)
        cxt.line_to(50, 50)

        cxt.translate(100., 0.)
        cxt.scale(0.8, 0.7)

        cxt.move_to(10., 10.)
        cxt.line_to(100., 100.)
        cxt.arc(200., 200., 80., 0., 1.1*pi)
        cxt.rotate(0.3)

        cxt.scale(0.7, 1.2)
        cxt.translate(50., 50.)

        cxt.line_to(300., 300.)
        cxt.rotate(-0.3)
        cxt.arc_negative(400., 300., 60., 0., -1.8*pi)
        cxt.line_to(600.-10, 400.-10)
        cxt.stroke()

    def _draw_test(cxt):
        cxt.move_to(10, 10)
        cxt.line_to(50, 50)
        cxt.stroke()

    import cairo

    W, H = 600., 400. # point == 1/72 inch

    # black line should follow the red line.
    surface = cairo.PDFSurface("output.pdf", W, H)
    context = cairo.Context(surface)

    context.save()
    context.set_source_rgba(1., 0., 0., 0.5)
    context.set_line_width(10.)
    draw_test(context)
    context.restore()

    cxt = Flatten()
    draw_test(cxt)
    for path in cxt.paths:
        path.process_cairo(context)

    surface.finish()

    print("OK")



if __name__ == "__main__":

    test()


