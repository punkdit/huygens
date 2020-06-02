#!/usr/bin/env python3

from math import pi, sqrt, sin, cos

EPSILON = 1e-6

# simple namespace class
class NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


SCALE_CM_TO_POINT = 72.0/2.54 # convert cm's to points at 72 dpi.


class Base(object):
    def __str__(self):
        attrs = list(self.__dict__.keys())
        attrs.sort()
        attrs = ', '.join("%s=%s"%(k, self.__dict__[k]) for k in attrs)
        return "%s(%s)"%(self.__class__.__name__, attrs)
    __repr__ = __str__


# ----------------------------------------------------------------------------
# See: https://www.cairographics.org/cookbook/matrix_transform/
#

class Matrix(Base):
    "Affine transformation matrix"
    def __init__(self, xx=1.0, yx=0.0, xy=0.0, yy=1.0, x0=0.0, y0=0.0):
        self.xx = xx
        self.yx = yx
        self.xy = xy
        self.yy = yy
        self.x0 = x0 # x translate
        self.y0 = y0 # y translate

    def __call__(self, _x, _y):
        x = self.xx * self.x + self.xy * self.y + self.x0
        y = self.yx * self.x + self.yy * self.y + self.y0
        return x, y

    def __eq__(self, other):
        return sum(abs(self[i]-other[i]) for i in range(6)) < EPSILON

    def __ne__(self, other):
        return sum(abs(self[i]-other[i]) for i in range(6)) > EPSILON

    def __getitem__(self, idx):
        return [self.xx, self.yx, self.xy, self.yy, self.x0, self.y0][idx]

    def multiply(self, other):
        "algebraic order: first do other then do self"
        xx = self.xx*other.xx + self.xy*other.yx
        yx = self.yx*other.xx + self.yy*other.yx
        xy = self.xx*other.xy + self.xy*other.yy
        yy = self.xy*other.xy + self.yy*other.yy
        x0 = self.xx*other.x0 + self.xy*other.y0 + self.x0
        y0 = self.yx*other.x0 + self.yy*other.y0 + self.y0
        return Matrix(xx, yx, xy, yy, x0, y0)
    __mul__ = multiply

    @classmethod
    def translate(cls, dx, dy):
        return cls(1., 0., 0., 1., dx, dy)

    @classmethod
    def scale(cls, sx, sy):
        return cls(sx, 0., 0., sy, 0., 0.)

    @classmethod
    def rotate(cls, radians, x=0., y=0.):
        s, c = sin(radians), cos(radians)
        return cls(c, s, -s, c, x-c*x+s*y, y-s*x-c*y)


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
        self.offset = 0., 0. # translate # XXX TODO USE Matrix XXX TODO

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
        #self.log("get_current_point()")
        return (0., 0.) # seems to work ...
        #return self.pos

    def has_current_point(self):
        return False # seems to work ...
        #return self.pos is not None

    def translate(self, dx, dy):
        x, y = self.offset
        self.offset = (x+dx, y+dy)

    def move_to(self, x, y):
        dx, dy = self.offset
        x, y = (x+dx, y+dy)
        self.pos = x, y

    def line_to(self, x, y):
        dx, dy = self.offset
        x += dx
        y += dy
        self.pos = (x, y)

    def curve_to(self, x0, y0, x1, y1, x2, y2):
        dx, dy = self.offset
        x0 += dx
        y0 += dy
        x1 += dx
        y1 += dy
        x2 += dx
        y2 += dy
        self.pos = (x2, y2)

    def close_path(self):
        pass

    def set_source_rgba(self, r, g, b, a):
        pass

    def set_line_width(self, w):
        pass

    def fill_preserve(self):
        pass

    def stroke(self):
        self.pos = None

    def get_font_options(self):
        return self # me again!

    def set_font_options(self, fo): # skip this ...
        pass

    def set_hint_style(self, x): # font_options method
        pass

    def set_hint_metrics(self, x): # font_options method
        pass

    def set_miter_limit(self, x): # skip this ...
        pass

    def set_antialias(self, x): # skip this ...
        pass



def test():
    x, y = 1.2, 3.4
    radians = 1.234 * pi
    rotate = Matrix.rotate
    translate = Matrix.translate

    I = Matrix()
    assert translate(x, y) * translate(-x, -y) == I

    lhs = translate(x, y)*rotate(radians)*translate(-x, -y)
    rhs = rotate(radians, x, y)
    assert lhs == rhs, (lhs, rhs)



if __name__ == "__main__":
    test()


