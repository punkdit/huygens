#!/usr/bin/env python3

import sys

from cairosvg.parser import Tree
from cairosvg.surface import Surface

from bruhat.argv import argv
from bruhat.render import back
from bruhat.render.back import SCALE_CM_TO_POINT

POINT_TO_CM = 1./SCALE_CM_TO_POINT


class State(object):
    def __init__(self, **kw):
        self.state = dict(kw)


class Method(object):
    def __init__(self, context, name):
        self.context = context
        self.name = name
        
    def __call__(self, *args, **kw):
        assert not kw
        self.context.log(self.name, *args)
        return self.context


class Context(object):

    save_attrs = 'pos offset'.split()

    def __init__(self):
        self.stack = []
        self.pos = None # current point
        self.offset = 0., 0. # translate
        self.path = []
        self.paths = []

    def save(self):
        #self.log("save")
        state = {}
        for k in self.save_attrs:
            state[k] = getattr(self, k)
        self.stack.append(state)

    def restore(self):
        #self.log("restore")
        state = self.stack.pop()
        self.__dict__.update(state)

    def __str__(self):
        return "Context(%r)"%(self.name,)
    __repr__ = __str__

    def __getattr__(self, attr):
        return Method(self, attr)

    def log(self, method, *args):
        INDENT = "  "*len(self.stack)
        print("%scontext.%s(%s)"%(INDENT, method, ', '.join(str(a) for a in args)))

    def scale(self, sx, sy):
        assert abs(sx-1.0)<1e-6, "TODO"
        assert abs(sy-1.0)<1e-6, "TODO"

    def get_current_point(self):
        self.log("get_current_point()")
        return self.pos

    def translate(self, dx, dy):
        x, y = self.offset
        self.offset = (x+dx, y+dy)

    def move_to(self, x, y):
        dx, dy = self.offset
        pos = (x+dx, y+dy)
        self.pos = pos

    def line_to(self, x, y):
        dx, dy = self.offset
        x += dx
        y += dy
        item = back.LineTo_Pt(x, y)
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
        item = back.CurveTo_Pt(x0, y0, x1, y1, x2, y2)
        self.path.append(item)
        self.pos = (x2, y2)

    def close_path(self):
        item = back.ClosePath()
        self.path.append(item)

    def set_source_rgba(self, r, g, b, a):
        deco = back.RGBA(r, g, b, a)

    def set_line_width(self, w):
        deco = back.LineWidth_Pt(r, g, b, a)

    def fill_preserve(self):
        deco = back.FillPreserve()

    def stroke(self):
        deco = back.Stroke()
        self.paths.append(self.path)
        self.path = []
        self.pos = None

    def has_current_point(self):
        return self.pos is not None

    def get_font_options(self):
        return self

    def set_font_options(self, fo):
        pass

    def set_hint_style(self, x): # font_options
        pass

    def set_hint_metrics(self, x): # font_options
        pass

    def set_miter_limit(self, x):
        pass

    def set_antialias(self, x):
        pass


class MySurf(Surface):

    #def _create_surface(self, width, height): # FAIL
    #    surface = cairo.PDFSurface("test_out.pdf", width, height)
    #    return surface, width, height

    def __init__(self, tree, output, dpi):
    
        W, H = 600., 200. # point == 1/72 inch

        if 0:
            surface = cairo.PDFSurface("test_out.pdf", W, H)
            self.context = cairo.Context(surface)

        else:
            self.context = Context()

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



def test():
    import cairo

    W, H = 600., 200. # point == 1/72 inch

    surface = cairo.PDFSurface("test_out.pdf", W, H)
    context = cairo.Context(surface)

    context.move_to(91.93, 81.96)
    context.translate(91.93, 81.96)
    context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    #context.set_line_width(1.0)


    if 1:
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

    if 1:
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
    
        my = MySurf(tree, None, 72.)






