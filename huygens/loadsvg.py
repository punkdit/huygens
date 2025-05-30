#!/usr/bin/env python3

"""
Hijack cairosvg to load svg into our internal data structures.

Does not work on all svg files, but seems good enough for text.
"""

from cairosvg.parser import Tree
from cairosvg.surface import Surface

from huygens.argv import argv
from huygens import back
from huygens.flatten import Flatten


class DummySurf(Surface):

    def __init__(self, tree, output, dpi, context=None):
    
        W, H = 600., 200. # point == 1/72 inch

        if context is None:
            context = Flatten()
        self.context = context

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


class SkipColors(Flatten):
    def set_source_rgba(self, r, g, b, a):
        pass


def loadsvg(name, dpi=72., keep_colors=True):
    assert name.endswith(".svg")
    s = open(name).read()
    tree = Tree(bytestring=s)
    if keep_colors:
        context = Flatten()
    else:
        context = SkipColors()
    dummy = DummySurf(tree, None, dpi, context=context)
    item = back.Compound(dummy.paths)
    return item



def loadtex(filename, width=1.):
    # XXX does not work very well... XXX
    from huygens.front import Canvas, Translate, Scale
    tex = loadsvg(filename)
    bb = tex.get_bound_box()
    llx, lly, urx, ury = bb

    conv = lambda a,b:(a+b)/2
    x0, y0 = conv(llx, urx), conv(lly, ury)
    w, h = urx-llx, ury-lly

    X0, Y0 = 0., 0.
    cvs = Canvas()
    cvs.append(Translate(X0-x0, Y0-y0))
    cvs.append(Scale(width/w, width/w, x0, y0))
    cvs.append(tex)
    return cvs




if __name__ == "__main__":

    name = argv.next()
    s = open(name).read()
    tree = Tree(bytestring=s)
    my = DummySurf(tree, None, 72.)
    from huygens.front import Canvas
    cvs = Canvas(my.paths)
    if 0:
      for item in cvs:
        print(item.__class__)
        for sub in item:
            print(sub)
        print()
    cvs.writePDFfile("output.pdf")
    cvs.writeSVGfile("output.svg")






