#!/usr/bin/env python3
"""
copied from arrowtheory repo
"""

import sys, os
from math import *

import pyx.bbox
from pyx import canvas, path, deco, trafo, style, text, color, deformer, unit
from pyx.color import rgb

from bruhat.argv import argv
from turtle import Turtle


if __name__ == "__main__":
    text.set(cls=text.LatexRunner, texenc='utf-8')
    text.preamble(r"\usepackage{amsmath,amsfonts,amssymb}")

north = [text.halign.boxcenter, text.valign.top]
northeast = [text.halign.boxright, text.valign.top]
northwest = [text.halign.boxleft, text.valign.top]
south = [text.halign.boxcenter, text.valign.bottom]
southeast = [text.halign.boxright, text.valign.bottom]
southwest = [text.halign.boxleft, text.valign.bottom]
east = [text.halign.boxright, text.valign.middle]
west = [text.halign.boxleft, text.valign.middle]
center = [text.halign.boxcenter, text.valign.middle]

black = rgb(0., 0., 0.) 
blue = rgb(0.1, 0.1, 0.9)
red = lred = rgb(0.8, 0.4, 0.2)
st_red = [red]
green = rgb(0.0, 0.6, 0.0)
white = rgb(1., 1., 1.) 
#shade = rgb(0.75, 0.55, 0)
grey = rgb(0.75, 0.75, 0.75)
yellow = rgb(1., 1., 0.)

st_dashed = [style.linestyle.dashed]
st_dotted = [style.linestyle.dotted]
st_round = [style.linecap.round]

st_thick = [style.linewidth.thick]
st_Thick = [style.linewidth.Thick]
st_THick = [style.linewidth.THick]
st_THICK = [style.linewidth.THICK]
#print dir(style.linewidth)

st_font = [text.size.Large]


def save(name):
    filt = ''
    
    if sys.argv[1:]:
        filt = sys.argv[1]

    if filt and filt not in name:
        return
    name = 'images_trick/'+name
    c.writePDFfile(name)
    c.writeSVGfile(name)
    print(name)


#def _X_save(name):
#    name = 'images/'+name
#    if 'pdf' in sys.argv:
#        c.writePDFfile(name)
#    if 'svg' in sys.argv:
#        c.writeSVGfile(name)


#def test():
#    c = canvas.canvas()
#
#    c.stroke(path.line(0, 0, 1, 1))
#    c.stroke(path.line(1, 1, 0, 0))
#
#    b = pyx.bbox.empty()
#    for item in c.items:
#        b1 = item.bbox()
#        print(b1.llx_pt)
#        print(b.llx_pt)
#        b += b1
#    print(b.llx_pt)


def cross(cvs, x, y, r=0.1, deco=[]):
    cvs.stroke(path.line(x-r, y, x+r, y), deco+st_THick)
    cvs.stroke(path.line(x, y-r, x, y+r), deco+st_THick)


def get_box_anchor(extent, align):
    (x0, y0, x1, y1) = extent

    xmid = 0.5*(x0 + x1)
    ymid = 0.5*(y0 + y1)
    if align == "center":
        dx, dy = xmid, ymid
    elif align == "north":
        dx, dy = xmid, y1
    elif align == "south":
        dx, dy = xmid, y0
    elif align == "east":
        dx, dy = x1, ymid
    elif align == "west":
        dx, dy = x0, ymid
    elif align == "northeast":
        dx, dy = x1, y1
    elif align == "northwest":
        dx, dy = x0, y1
    elif align == "southeast":
        dx, dy = x1, y0
    elif align == "southwest":
        dx, dy = x0, y0
    else:
        assert 0, "alignment %r not understood" % align

    return dx, dy


class Box(object):

    debug = bool(argv.debug)

    # Store absolute bounding box (llx, lly, urx, ury)
    # after a render.
    pos = None

    @classmethod
    def promote(self, item):
        if isinstance(item, Box):
            return item
        if isinstance(item, canvas.canvas):
            return CanBox(item)
        if isinstance(item, str):
            return TextBox(item)
        if isinstance(item, (tuple, list)):
            return HBox(item)
        assert 0, item

    def render(self, anchor_x=0., anchor_y=0., cvs=None, name=None):
        if cvs is None:
            cvs = canvas.canvas()
        self.on_render(anchor_x, anchor_y, cvs)

        (x0, y0, x1, y1) = self.get_extent()
        self.pos = (anchor_x+x0, anchor_y+y0, anchor_x+x1, anchor_y+y1)
    
        if self.debug:
            cross(cvs, anchor_x, anchor_y, 0.1)
            cvs.stroke(path.line(anchor_x, anchor_y, anchor_x+x0, anchor_y+y0), [deco.earrow()])
            cvs.stroke(path.line(anchor_x, anchor_y, anchor_x+x1, anchor_y+y1), [deco.earrow()])
        
        if name is not None:
            cvs.writePDFfile(name)
            cvs.writeSVGfile(name)
        return cvs

    def save(self, name):
        self.render(name=name)

    def on_render(self, anchor_x, anchor_y, cvs):
        return None # ?

    def get_extent(self):
        return (0., 0., 0., 0.)

    def get_pos(self):
        assert self.pos is not None, "need to render me first."
        pos = self.pos
        return pos

    def get_anchor(self, align):
        pos = self.get_pos()
        dx, dy = get_box_anchor(pos, align)
        return dx, dy


class SpaceBox(Box):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def get_extent(self):
        return (0., 0., self.w, self.h)



class CanBox(Box):
    "Default anchor for a CanBox is at (0, 0) "
    def __init__(self, cvs=None, ax=0., ay=0.):
        if cvs is None:
            cvs = canvas.canvas()
        self.cvs = cvs
        self.anchor = (ax, ay)

    def __getattr__(self, attr):
        return getattr(self.csv, attr)

    def _get_bb(self):
        bbox = self.cvs.bbox()
        #bb = (bbox.llx_pt, bbox.lly_pt, bbox.urx_pt, bbox.ury_pt)
        conv = lambda x : unit.tocm(unit.length(x, unit="pt"))
        (x0, y0, x1, y1) = (
            conv(bbox.llx_pt), conv(bbox.lly_pt), 
            conv(bbox.urx_pt), conv(bbox.ury_pt))
        return (x0, y0, x1, y1)

    def get_extent(self):
        "lower-left and upper-right points, relative to anchor point"
        ax, ay = self.anchor
        (x0, y0, x1, y1) = self._get_bb()
        return (x0-ax, y0-ay, x1-ax, y1-ay)

    def on_render(self, anchor_x, anchor_y, cvs):
        (x0, y0, x1, y1) = self.get_extent()
        ax, ay = self.anchor
        if self.debug:
            cross(self.cvs, ax, ay, 0.2, st_red)
        #print("CanBox.on_render:", anchor_x-x0, anchor_y-y0)
        cvs.insert(self.cvs, [trafo.translate(anchor_x-ax, anchor_y-ay)])


class TextBox(CanBox):
    "anchor is in the center of the text"
    def __init__(self, text, scale=1.):
        cvs = canvas.canvas()
        cvs.text(0, 0, text, center+[trafo.scale(sx=scale, sy=scale)])
        CanBox.__init__(self, cvs)


class CompoundBox(Box):
    def __init__(self, boxs):
        self.boxs = [Box.promote(box) for box in boxs]

    def __len__(self):
        return len(self.boxs)

    def __getitem__(self, idx):
        return self.boxs[idx]

    def get_extent(self):
        layout, extent = self.get_layout(0., 0.)
        return extent

    def on_render(self, anchor_x, anchor_y, cvs):
        layout, extent = self.get_layout(anchor_x, anchor_y)
        for idx, box in enumerate(self.boxs):
            x, y = layout[idx]
            box.render(x, y, cvs)


class MarginBox(Box):
    def __init__(self, child, margin=0.1, st=[grey]):
        self.child = Box.promote(child)
        self.margin = margin
        self.st = st

    def get_extent(self):
        child = self.child
        (x0, y0, x1, y1) = child.get_extent()
        return (x0, y0, x1, y1)

    def on_render(self, anchor_x, anchor_y, cvs):
        child = self.child
        (x0, y0, x1, y1) = child.get_extent()
        m = self.margin
        p = path.rect(anchor_x+x0-m, anchor_y+y0-m, x1-x0+2*m, y1-y0+2*m)
        st = self.st
        #cvs.stroke(p, st+st_Thick)
        cvs.fill(p, st)
        child.on_render(anchor_x, anchor_y, cvs)


class OBox(CompoundBox):
    "overlay boxes on top of each other"

    def get_extent(self):
        assert self.boxs
        box = self.boxs[0]
        (x0, y0, x1, y1) = box.get_extent()
        for box in self.boxs[1:]:
            (_x0, _y0, _x1, _y1) = box.get_extent()
            x0 = min(x0, _x0)
            y0 = min(y0, _y0)
            x1 = max(x1, _x1)
            y1 = max(y1, _y1)
        return (x0, y0, x1, y1)

    def on_render(self, anchor_x, anchor_y, cvs):
        for box in self.boxs:
            box.render(anchor_x, anchor_y, cvs)


class HBox(CompoundBox):
    "horizontal align boxes left to right"
    def __init__(self, boxs, anchor="west", pad=0.1):
        self.boxs = [Box.promote(box) for box in boxs]
        self.anchor = anchor
        assert anchor == "west", "not implemented"
        self.pad = pad

    def get_layout(self, anchor_x, anchor_y):
        y_bot = y_top = 0.
        y = 0.
        x = 0.
        layout = {}
        for idx, box in enumerate(self.boxs):
            (x0, y0, x1, y1) = box.get_extent()
            y_bot = min(y_bot, y0)
            y_top = max(y_top, y1)
            layout[idx] = (x+anchor_x-x0, y+anchor_y)
            x += x1-x0 # width of this box
            x += self.pad
        extent = (0., y_bot, x-self.pad, y_top)
        return layout, extent



class VBox(CompoundBox):
    "vertical align boxes top to bottom"
    def __init__(self, boxs, anchor="north", pad=0.1):
        self.boxs = [Box.promote(box) for box in boxs]
        self.anchor = anchor
        assert anchor == "north", "not implemented"
        self.pad = pad

    def get_layout(self, anchor_x, anchor_y):
        x_bot = x_top = 0.
        x = 0.
        y = 0.
        layout = {}
        for idx, box in enumerate(self.boxs):
            (x0, y0, x1, y1) = box.get_extent()
            x_bot = min(x_bot, x0)
            x_top = max(x_top, x1)
            layout[idx] = (x+anchor_x, y+anchor_y-y1)
            y -= y1-y0 # height of this box
            y -= self.pad
        y += self.pad
        extent = (x_bot, y, x_top, 0.)
        return layout, extent


class OffsetBox(CompoundBox):
    def __init__(self, box, offset_x, offset_y):
        box = Box.promote(box)
        CompoundBox.__init__(self, [box])
        self.offset_x = offset_x
        self.offset_y = offset_y

    def get_extent(self):
        box = self.boxs[0]
        (x0, y0, x1, y1) = box.get_extent()
        dx = self.offset_x
        dy = self.offset_y
        return (x0+dx, y0+dy, x1+dx, y1+dy)

    def on_render(self, anchor_x, anchor_y, cvs):
        box = self.boxs[0]
        x = self.offset_x + anchor_x
        y = self.offset_y + anchor_y
        box.render(x, y, cvs)


class AnchorBox(OffsetBox):
    def __init__(self, box, align):
        #assert align in "center north south east west northeast northwest southeast southwest".split()
        box = Box.promote(box)
        self.align = align
        extent = box.get_extent()
        dx, dy = get_box_anchor(extent, align)

        OffsetBox.__init__(self, box, -dx, -dy)


class ChildAlignBox(OffsetBox):
    "use the anchor of child at idx"
    def __init__(self, box, idx=0):
        box = Box.promote(box)
        assert isinstance(box, CompoundBox)
        assert 0<=idx<len(box)

        layout, extent = box.get_layout(0., 0.)

        child = box[idx]
        dx, dy = layout[child]

        OffsetBox.__init__(self, box, -dx, -dy)



class RectBox(Box):
    def __init__(self, child, margin=0.1, extra=[]):
        Box.__init__(self)
        self.child = Box.promote(child)
        self.margin = margin
        self.extra = list(extra)

    def get_extent(self):
        child = self.child
        (x0, y0, x1, y1) = child.get_extent()
        m = self.margin
        return (x0-m, y0-m, x1+m, y1+m)

    def draw_rect(self, cvs, x0, y0, w, h, r=0.):
        w -= 2*r
        h -= 2*r
        t = Turtle(x0, y0+r)
        t.fwd(h)
        t.right(90., r)
        t.fwd(w)
        t.right(90., r)
        t.fwd(h)
        t.right(90., r)
        t.fwd(w)
        t.right(90., r)
        t.stroke(closepath=True, cvs=cvs, extra=self.extra)

    def on_render(self, anchor_x, anchor_y, cvs):
        child = self.child
        x = anchor_x
        y = anchor_y
        child.render(x, y, cvs)
        (x0, y0, x1, y1) = child.get_extent()
        m = self.margin
        w = x1-x0+2*m
        h = y1-y0+2*m
        self.draw_rect(cvs, anchor_x+x0-m, anchor_y+y0-m, w, h, m)




        
def test():
    cvs = canvas.canvas()
    cvs.fill(path.circle(0., 0., 1.), [green])
    a = CanBox(cvs)

    b = TextBox("$=$", 2.0)

    cvs = canvas.canvas()
    cvs.fill(path.circle(0., 0., 0.5), [yellow])
    c = CanBox(cvs)

    box = HBox([a, b, c])

    box.render(name="test")



def main():

    from bruhat.graphs import cycle_graph
    graph = list(cycle_graph(4, 0.6))[1]
    #graph.draw([-4, 2, 0, 2], "images_trick/square")

    boxs = [
        TextBox(r"$2\times$", 2.),
        graph.draw([-2, 1, 0, 1]),
        TextBox(r"$=$", 2.),
        graph.draw([-4, 2, 0, 2]),
    ]

    box = HBox(boxs, pad=0.3)

    c = box.render(name="images_trick/pic-abig")

    # ------------------------------------------------------------- #

    boxs = [
        graph.draw([-2, 1, 0, 1]),
        TextBox(r"$+$", 2.),
        graph.draw([1, 0, 1, -2]),
        TextBox(r"$=$", 2.),
        graph.draw([-1, 1, 1, -1]),
    ]

    box = HBox(boxs, pad=0.3)

    c = box.render(name="images_trick/pic-ontop")

    # ------------------------------------------------------------- #

    a = graph.draw(   [-2, 1, 0, 1])
    b = graph.draw(   [-1, 0, 0, 2])
    ab = graph.draw(  [-3, 1, 0, 3])
    aa = graph.draw(  [-4, 2, 0, 2])
    bb = graph.draw(  [-2, 0, 0, 4])
    aabb = graph.draw([-6, 2, 0, 6])

    HBox([
        TextBox(r"$2\times\Biggl($", 2.), a, TextBox(r"$+$", 2.), b, TextBox(r"$\Biggr)$", 2.),
    ], pad=0.3).render(name="images_trick/pic-wavedist-1")

    VBox([
        ChildAlignBox(
        HBox([
            TextBox(r"$2\times\Biggl($", 2.), a, TextBox(r"$+$", 2.), b, TextBox(r"$\Biggr)$", 2.),
            TextBox("$=$", 2.),
            TextBox(r"$2\ \times$", 2.), a, TextBox(r"$+$", 2.), TextBox(r"$2\ \times$", 2.), b,
        ], pad=0.3), 5),
        ChildAlignBox(
        HBox([
            TextBox("$=$", 2.),
            aa, TextBox(r"$+$", 2.), bb,
        ])),
        ChildAlignBox(HBox([TextBox("$=$", 2.), aabb])),
    ]).render(name="images_trick/pic-wavedist-3")

    VBox([
        ChildAlignBox(
        HBox([
            TextBox(r"$2\times\Biggl($", 2.), a, TextBox(r"$+$", 2.), b, TextBox(r"$\Biggr)$", 2.),
            TextBox("$=$", 2.),
            TextBox(r"$2\ \times$", 2.), ab,
        ], pad=0.3), 5),
        ChildAlignBox(HBox([TextBox("$=$", 2.), aabb])),
    ]).render(name="images_trick/pic-wavedist-2")





if __name__ == "__main__":

    if argv.test:
        test()

    else:
        main()


