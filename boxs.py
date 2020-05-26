#!/usr/bin/env python3

from random import random
from math import pi

import cairo

# Context attr's
"""
append_path arc arc_negative clip clip_extents clip_preserve
close_path copy_clip_rectangle_list copy_page copy_path
copy_path_flat curve_to device_to_user device_to_user_distance
fill fill_extents fill_preserve font_extents get_antialias
get_current_point get_dash get_dash_count get_fill_rule
get_font_face get_font_matrix get_font_options get_group_target
get_line_cap get_line_join get_line_width get_matrix
get_miter_limit get_operator get_scaled_font get_source
get_target get_tolerance glyph_extents glyph_path has_current_point
identity_matrix in_fill in_stroke line_to mask mask_surface
move_to new_path new_sub_path paint paint_with_alpha
path_extents pop_group pop_group_to_source push_group
push_group_with_content rectangle rel_curve_to rel_line_to
rel_move_to reset_clip restore rotate save scale select_font_face
set_antialias set_dash set_fill_rule set_font_face set_font_matrix
set_font_options set_font_size set_line_cap set_line_join
set_line_width set_matrix set_miter_limit set_operator
set_scaled_font set_source set_source_rgb set_source_rgba
set_source_surface set_tolerance show_glyphs show_page
show_text stroke stroke_extents stroke_preserve text_extents
text_path transform translate user_to_device user_to_device_distance
"""

from bruhat.render.sat import Variable, System
from bruhat.argv import argv

def rnd(a, b):
    return (b-a)*random() + a

class Box(object):

    DEBUG = True
    fixed = False

    def on_layout(self, cxt, system):
        assert not self.fixed, "already called on_layout"
        if self.DEBUG:
            print("%s.on_layout" % (self.__class__.__name__,))
        for attr in 'x y left right top bot'.split():
            if attr in self.__dict__:
                continue
            stem = self.__class__.__name__ + '.' + attr
            v = system.get_var(stem)
            setattr(self, attr, v)
        self.fixed = True

    def on_render(self, cxt, system):
        if not self.DEBUG:
            return
        cxt.save()
        cxt.set_source_rgba(1., 0., 0., 0.5)
        x = system[self.x]
        y = system[self.y]
        left = system[self.left]
        right = system[self.right]
        top = system[self.top]
        bot = system[self.bot]
        cxt.set_line_width(0.5)
        #cxt.move_to(x, y)
        #cxt.arc(x, y, 1.0, 0, 2*pi)
        #cxt.stroke()
        r = 1.4
        cxt.move_to(x-r, y-r)
        cxt.line_to(x+r, y+r)
        cxt.stroke()
        cxt.move_to(x+r, y-r)
        cxt.line_to(x-r, y+r)
        cxt.stroke()
        cxt.rectangle(x-left, y-top, left+right, top+bot)
        cxt.stroke()
        cxt.restore()

    @property
    def width(self):
        return self.left + self.right

    @property
    def height(self):
        return self.top + self.bot

    def render(self, cxt, x=0, y=0):
        system = System()
        self.on_layout(cxt, system)
        system.add(self.x == x)
        system.add(self.y == y)
        system.solve()
        self.on_render(cxt, system)


class EmptyBox(Box):
    def __init__(self, top, bot, left, right):
        self.top = top
        self.bot = bot
        self.left = left
        self.right = right
        self.DEBUG = True

    def on_layout(self, cxt, system):
        #assert "x" not in self.__dict__
        Box.on_layout(self, cxt, system)


class TextBox(Box):
    def __init__(self, text):
        self.text = text

    def on_layout(self, cxt, system):
        Box.on_layout(self, cxt, system)
        extents = cxt.text_extents(self.text)
        (dx, dy, width, height, _, _) = extents
        system.add(self.left + self.right == width+dx)
        system.add(self.top + self.bot == height)
        system.add(self.left == 0)
        assert dy <= 0., dy
        system.add(self.top == -dy)

    def on_render(self, cxt, system):
        Box.on_render(self, cxt, system)
        x = system[self.x]
        y = system[self.y]
        cxt.move_to(x, y)
        cxt.show_text(self.text)


class CompoundBox(Box):
    def __init__(self, boxs):
        assert len(boxs)
        self.boxs = list(boxs)

    def on_layout(self, cxt, system):
        Box.on_layout(self, cxt, system)
        for box in self.boxs:
            box.on_layout(cxt, system)

    def on_render(self, cxt, system):
        Box.on_render(self, cxt, system)
        for box in self.boxs:
            box.on_render(cxt, system)


class HBox(CompoundBox):
    def on_layout(self, cxt, system):
        CompoundBox.on_layout(self, cxt, system)
        boxs = self.boxs
        left = self.x
        for box in boxs:
            system.add(self.y == box.y) # align
            system.add(box.x - box.left == left)
            left += box.width
            system.add(box.top <= self.top)
            system.add(box.bot <= self.bot)
        system.add(self.x + self.width == left)
        system.add(self.left == 0.)


class VBox(CompoundBox):
    def on_layout(self, cxt, system):
        CompoundBox.on_layout(self, cxt, system)
        boxs = self.boxs
        top = self.y
        for box in boxs:
            system.add(self.x == box.x) # align
            system.add(box.y - box.top == top)
            top += box.height
            system.add(box.left <= self.left)
            system.add(box.right <= self.right)
        system.add(self.y + self.height == top)
        system.add(self.top == 0.)


def main():
    W, H = 200, 200
    #surface = cairo.SVGSurface("example.svg", W, H)
    surface = cairo.PDFSurface("example.pdf", W, H)
    cxt = cairo.Context(surface)
    #print(' '.join(dir(cxt)))
    
#    box = HBox([
#        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
#        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
#    ])

    
    if 1:
        a, b = 10, 20
        rows = []
        for row in range(2):
            boxs = []
            for col in range(2):
                top = rnd(a, b)
                bot = rnd(a, b)
                left = rnd(a, b)
                right = rnd(a, b)
                box = EmptyBox(top, bot, left, right)
                boxs.append(box)
            boxs.append(TextBox("hi !"))
            box = HBox(boxs)
            rows.append(box)
        box = VBox(rows)

#    box = VBox([
#        EmptyBox(20, 5, 5, 18),
#        EmptyBox(10, 8, 9, 16),
#    ])

    #box = TextBox('xyyxy !')
    box.render(cxt, W/2., H/2.)

    surface.finish()


def test():
    W, H = 200, 200
    #surface = cairo.SVGSurface("example.svg", W, H)
    surface = cairo.PDFSurface("example.pdf", W, H)
    cxt = cairo.Context(surface)
    #print(' '.join(dir(cxt)))
    
    #m = cairo.Matrix()
    #m.scale(1., -1.)
    #cxt.transform(m)
    
    cxt.move_to(10, 0)
    cxt.line_to(10, H)
    cxt.stroke()
    cxt.move_to(0, 10)
    cxt.line_to(W, 10)
    cxt.stroke()
    
    #print(' '.join(dir(surface)))
    
    
    s = 'ggghi there!'
    ex = cxt.text_extents(s)
    (x_bearing, y_bearing, width, height, x_advance, y_advance) = ex
    print(ex)
    
    x, y = W/2, H/2
    cxt.move_to(x, y)
    cxt.show_text(s)
    
    cxt.set_source_rgba(0.2, 1.0, 0.2, 0.6)
    #cxt.rectangle(x, y, 5, 5)
    cxt.move_to(x, y)
    cxt.arc(x, y, 1.0, 0., 2*pi)
    cxt.stroke()
    
    cxt.save()
    cxt.set_line_width(0.1)
    cxt.set_source_rgba(1, 0.2, 0.2, 0.2)
    cxt.rectangle(x+x_bearing, y+y_bearing, width, height)
    cxt.stroke()
    cxt.restore()


def test_curve(cxt):
    x, y, x1, y1 = 0.1, 0.5, 0.4, 0.9
    x2, y2, x3, y3 = 0.6, 0.1, 0.9, 0.5
    cxt.scale(200, 200)
    cxt.set_line_width(0.04)
    cxt.move_to(x, y)
    cxt.curve_to(x1, y1, x2, y2, x3, y3)
    cxt.stroke()
    cxt.set_source_rgba(1, 0.2, 0.2, 0.6)
    cxt.set_line_width(0.02)
    cxt.move_to(x, y)
    cxt.line_to(x1, y1)
    cxt.move_to(x2, y2)
    cxt.line_to(x3, y3)
    cxt.stroke()

if __name__ == "__main__":

    #test()
    main()

    print("OK\n")



