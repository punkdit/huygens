#!/usr/bin/env python3

"""
Hijack cairosvg to load svg into our internal data structures.
"""

import sys
from math import sin, cos, pi

from cairosvg.parser import Tree
from cairosvg.surface import Surface

from bruhat.argv import argv
from bruhat.render import back
from bruhat.render.base import Context


class Reflector(Context):

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



class DummySurf(Surface):

    def __init__(self, tree, output, dpi):
    
        W, H = 600., 200. # point == 1/72 inch

        self.context = Reflector()

        self.dpi = dpi

        self._old_parent_node = self.parent_node = None
        self.output = output
        self.font_size = None

        self.context_width = W
        self.context_height = H

        self.cursor_position = [0, 0]
        self.cursor_d_position = [0, 0]
        self.text_path_width = 0
        self.stroke_and_fill = True

        self.tree_cache = {(tree.url, tree.get('id')): tree}

        self.markers = {}
        self.gradients = {}
        self.patterns = {}
        self.masks = {}
        self.paths = {}
        self.filters = {}

        self.map_rgba = None
        self.map_image = None

        self.draw(tree)
    
        #surface.finish()

        self.paths = self.context.paths


def loadsvg(name, dpi=72.):
    assert name.endswith(".svg")
    s = open(name).read()
    tree = Tree(bytestring=s)
    dummy = DummySurf(tree, None, dpi)
    item = back.Compound(dummy.paths)
    return item


def test():

    def draw_test(cxt):
        cxt.move_to(10., 10.)
        cxt.line_to(100., 100.)
        cxt.arc(200., 200., 80., 0., 1.1*pi)
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

    cxt = Reflector()
    draw_test(cxt)
    for path in cxt.paths:
        path.process_cairo(context)

    surface.finish()

    print("OK")



if __name__ == "__main__":

    if argv.test:
        test()

    else:

        name = argv.next()
        s = open(name).read()
        tree = Tree(bytestring=s)
        my = DummySurf(tree, None, 72.)
        cvs = back.Canvas(my.paths)
        cvs.writePDFfile("output.pdf")






