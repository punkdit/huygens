#!/usr/bin/env python3

"""
Hijack cairosvg to load svg into our internal data structures.
"""

import sys

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
        dx, dy = self.offset
        x, y = (x+dx, y+dy)
        item = back.MoveTo_Pt(x, -y)
        path = self.path
        path.append(item)
        self.pos = x, y

    def line_to(self, x, y):
        dx, dy = self.offset
        x += dx
        y += dy
        item = back.LineTo_Pt(x, -y)
        self.path.append(item)
        self.pos = (x, y)

    def curve_to(self, x0, y0, x1, y1, x2, y2):
        dx, dy = self.offset
        x0 += dx
        y0 += dy
        x1 += dx
        y1 += dy
        x2 += dx
        y2 += dy
        item = back.CurveTo_Pt(x0, -y0, x1, -y1, x2, -y2)
        self.path.append(item)
        self.pos = (x2, y2)

    def close_path(self):
        item = back.ClosePath()
        self.path.append(item)

    def set_source_rgba(self, r, g, b, a):
        deco = back.RGBA(r, g, b, a)
        self.path.append(deco)

    def set_line_width(self, w):
        deco = back.LineWidth_Pt(w)
        self.path.append(deco)

    def fill_preserve(self):
        deco = back.FillPreserve()
        self.path.append(deco)

    def stroke(self):
        deco = back.Stroke()
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
    import cairo

    W, H = 600., 200. # point == 1/72 inch

    surface = cairo.PDFSurface("test_out.pdf", W, H)
    context = cairo.Context(surface)

    context.move_to(91.93, 81.96)
    context.translate(91.93, 81.96)
    context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    #context.set_line_width(1.0)

    context.move_to(0, 0)
    context.move_to(0.921875, -0.921875)
    context.curve_to(0.921875, -0.46875, 0.984375, -0.640625, 0.140625, -0.640625)
    context.line_to(0.140625, 0.015625)
    context.curve_to(0.671875, -0.015625, 1.171875, -0.03125, 1.453125, -0.03125)
    context.curve_to(1.703125, -0.03125, 2.21875, -0.015625, 2.734375, 0.015625)
    context.line_to(2.734375, -0.640625)
    context.curve_to(1.890625, -0.640625, 1.953125, -0.46875, 1.953125, -0.921875)
    context.line_to(1.953125, -2.75)
    context.curve_to(1.953125, -3.78125, 2.5, -4.1875, 3.125, -4.1875)
    context.curve_to(3.765625, -4.1875, 3.6875, -3.8125, 3.6875, -3.234375)
    context.line_to(3.6875, -0.921875)
    context.curve_to(3.6875, -0.46875, 3.765625, -0.640625, 2.90625, -0.640625)
    context.line_to(2.90625, 0.015625)
    context.curve_to(3.4375, -0.015625, 3.953125, -0.03125, 4.21875, -0.03125)
    context.curve_to(4.46875, -0.03125, 5.0, -0.015625, 5.5, 0.015625)
    context.line_to(5.5, -0.640625)
    context.curve_to(4.8125, -0.640625, 4.734375, -0.46875, 4.71875, -0.765625)
    context.line_to(4.71875, -2.671875)
    context.curve_to(4.71875, -3.53125, 4.671875, -3.953125, 4.359375, -4.3125)
    context.curve_to(4.234375, -4.484375, 3.78125, -4.734375, 3.203125, -4.734375)
    context.curve_to(2.359375, -4.734375, 1.75, -3.96875, 1.578125, -3.59375)
    context.line_to(1.921875, -3.59375)
    context.line_to(1.921875, -7.265625)
    context.line_to(0.140625, -7.125)
    context.line_to(0.140625, -6.5)
    context.curve_to(1.015625, -6.5, 0.921875, -6.59375, 0.921875, -6.09375)
    context.close_path()
    context.move_to(0.921875, -0.921875)

    context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    context.fill_preserve()
    context.set_line_width(1.0)
    context.set_source_rgba(0, 0, 0, 0.0)
    context.stroke()

    surface.finish()




if __name__ == "__main__":

    if argv.test:
        test()

    else:

        name = argv.next()
        s = open(name).read()
        tree = Tree(bytestring=s)
        my = DummySurf(tree, None, 72.)
        cvs = back.Canvas(my.paths)
        cvs.writePDFfile("test_load.pdf")






