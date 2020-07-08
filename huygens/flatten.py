#!/usr/bin/env python3

"""
Interpret all transforms, thereby flattening cairo operations
to just a few primitives, such as move_to line_to and curve_to.
"""

import sys
from math import sin, cos, pi

from huygens import argv
from huygens import back
from huygens.base import Context


class Flatten(Context):

    def __init__(self):
        Context.__init__(self)
        self.path = back.Compound()
        self.paths = []

    def move_to(self, x, y):
        x, y = self.matrix(x, y)
        item = back.MoveTo_Pt(x, -y)
        self.path.append(item)
        self.pos = x, y

    def line_to(self, x, y):
        x, y = self.matrix(x, y)
        item = back.LineTo_Pt(x, -y)
        self.path.append(item)
        self.pos = x, y

    def curve_to(self, x0, y0, x1, y1, x2, y2):
        x0, y0 = self.matrix(x0, y0)
        x1, y1 = self.matrix(x1, y1)
        x2, y2 = self.matrix(x2, y2)
        item = back.CurveTo_Pt(x0, -y0, x1, -y1, x2, -y2)
        self.path.append(item)
        self.pos = x2, y2

    def rel_move_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        x, y = x+dx, y+dy
        item = back.MoveTo_Pt(x, -y)
        self.path.append(item)
        self.pos = x, y

    def rel_line_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        x, y = x+dx, y+dy
        item = back.LineTo_Pt(x, -y)
        self.path.append(item)
        self.pos = x, y

    def rel_curve_to(self, dx0, dy0, dx1, dy1, dx2, dy2):
        assert self.pos is not None, "no current point"
        x, y = self.pos
        dx0, dy0 = self.matrix.transform_distance(dx0, dy0)
        dx1, dy1 = self.matrix.transform_distance(dx1, dy1)
        dx2, dy2 = self.matrix.transform_distance(dx2, dy2)
        x0, y0 = x+dx0, y+dy0
        x1, y1 = x+dx1, y+dy1
        x2, y2 = x+dx2, y+dy2
        item = back.CurveTo_Pt(x0, -y0, x1, -y1, x2, -y2)
        self.path.append(item)
        self.pos = x2, y2

    def arc(self, x, y, radius, angle1, angle2):
        # stay in user space coordinates
        if self.pos is None:
            x1, y1 = x+radius*cos(angle1), y+radius*sin(angle1)
            self.move_to(x1, y1)
        p = back.arc_to_bezier_pt(x, -y, radius, -angle2, -angle1)
        p = p.reversed()
        p.process_cairo(self)

    def arc_negative(self, x, y, radius, angle1, angle2):
        # stay in user space coordinates
        if self.pos is None:
            x1, y1 = x+radius*cos(angle1), y+radius*sin(angle1)
            self.move_to(x1, y1)
        p = back.arc_to_bezier_pt(x, -y, radius, -angle1, -angle2)
        p.process_cairo(self)

    def close_path(self):
        item = back.ClosePath()
        self.path.append(item)

    def set_source_rgba(self, r, g, b, a):
        deco = back.RGBA(r, g, b, a)
        self.path.append(deco)

    def set_line_width(self, w):
        deco = back.LineWidth_Pt(w)
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



def test():

    def draw_test(cxt):
        cxt.translate(100., 0.)
        cxt.scale(0.8, 0.7)

        cxt.move_to(10., 10.)
        cxt.line_to(100., 100.)
        cxt.arc(200., 200., 80., 0., 1.1*pi)

        cxt.scale(0.7, 1.2)
        cxt.translate(50., 50.)

        cxt.line_to(300., 300.)
        cxt.arc_negative(400., 300., 60., 0., -1.8*pi)
        cxt.line_to(600.-10, 400.-10)
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


