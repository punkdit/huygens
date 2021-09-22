#!/usr/bin/env python3

from copy import deepcopy
from math import pi, sqrt, sin, cos, atan

EPSILON = 1e-6

# simple namespace class
class NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    #def __call__(self, *args, **kw):
    #    return self


SCALE_CM_TO_POINT = 72.0/2.54 # convert cm's to points at 72 dpi.
_defaultlinewidth = 0.02

class Base(object):
    def __str__(self):
        attrs = list(self.__dict__.keys())
        attrs.sort()
        attrs = ', '.join("%s=%s"%(k, self.__dict__[k]) for k in attrs)
        return "%s(%s)"%(self.__class__.__name__, attrs)
    __repr__ = __str__


# ----------------------------------------------------------------------------
# 
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

    def transform_point(self, _x, _y):
        x = self.xx * _x + self.xy * _y + self.x0
        y = self.yx * _x + self.yy * _y + self.y0
        return x, y
    __call__ = transform_point

    def transform_distance(self, _dx, _dy):
        dx = self.xx * _dx + self.xy * _dy
        dy = self.yx * _dx + self.yy * _dy
        return dx, dy

#    def transform_angle(self, angle): # ????
#        x, y = cos(angle), sin(angle) 
#        x, y = self.transform_distance(x, y)
#        angle = atan(y / x)
#        return angle

    def __eq__(self, other):
        return sum(abs(self[i]-other[i]) for i in range(6)) < EPSILON

    def __ne__(self, other):
        return sum(abs(self[i]-other[i]) for i in range(6)) > EPSILON

    def __getitem__(self, idx):
        return [self.xx, self.yx, self.xy, self.yy, self.x0, self.y0][idx]

    def multiply(left, right): # UGH this is twisted multiply: right*left !!
        xx = right.xx*left.xx + right.xy*left.yx
        yx = right.yx*left.xx + right.yy*left.yx
        xy = right.xx*left.xy + right.xy*left.yy
        yy = right.xy*left.yx + right.yy*left.yy
        x0 = right.xx*left.x0 + right.xy*left.y0 + right.x0
        y0 = right.yx*left.x0 + right.yy*left.y0 + right.y0
        return Matrix(xx, yx, xy, yy, x0, y0)
    __mul__ = multiply

    #def __mul__(left, right):
    #    return right.multiply(left) # ???

    def __pow__(self, n):
        if n==0:
            return Matrix()
        A = self
        while n>1:
            A = self*A
            n -= 1
        return A

    def asnumpy(self):
        import numpy
        M = [
            [self.xx, self.xy, self.x0],
            [self.yx, self.yy, self.y0],
            [      0,       0,       1]]
        M = numpy.array(M)
        return M

    def inv(self):
        import numpy
        M = self.asnumpy()
        Mi = numpy.linalg.inv(M)
        assert abs(Mi[2,0]) < EPSILON
        assert abs(Mi[2,1]) < EPSILON
        assert abs(Mi[2,2]-1.) < EPSILON
        m = Matrix(Mi[0,0], Mi[1,0], Mi[0,1], Mi[1,1], Mi[0,2], Mi[1,2])
        return m

    the_identity = None
    def is_identity(self):
        return self==Matrix.the_identity

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

Matrix.the_identity = Matrix()


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


# Internally huygens uses cm units, even though we are exposing a cairo interface:

class Context(object):

    save_attrs = 'pos matrix linewidth'.split()

    def __init__(self):
        self.stack = []
        self.pos = None # current point
        self.linewidth = _defaultlinewidth # *SCALE_CM_TO_POINT
        self.matrix = Matrix()

    def save(self):
        #self.log("save")
        state = {}
        for k in self.save_attrs:
            state[k] = deepcopy(getattr(self, k))
        self.stack.append(state)

    def restore(self):
        #self.log("restore")
        state = self.stack.pop()
        self.__dict__.update(state)

    def __str__(self):
        return "Context(%r)"%(self.name,)
    __repr__ = __str__

    def __getattr__(self, attr):
        assert 0, attr
        return Method(self, attr)

    def log(self, method, *args):
        INDENT = "  "*len(self.stack)
        print("%scontext.%s(%s)"%(INDENT, method, ', '.join(str(a) for a in args)))

    def get_current_point(self):
        #self.log("get_current_point()")
        # XXX apply inverse of current matrix XXX
        return (0., 0.) # seems to work ...
        #return self.pos

    def has_current_point(self):
        return False # seems to work ...
        #return self.pos is not None

    def scale(self, sx, sy):
        matrix = Matrix.scale(sx, sy) 
        self.matrix = matrix * self.matrix

    def translate(self, dx, dy):
        dx /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy /= SCALE_CM_TO_POINT # internally huygens uses cm units
        #dx, dy = self.matrix.transform_distance(dx, dy)
        matrix = Matrix.translate(dx, dy) 
        self.matrix = matrix * self.matrix

    def rotate(self, angle):
        matrix = Matrix.rotate(angle)
        self.matrix = matrix * self.matrix

    def transform(self, matrix):
        matrix = tuple(matrix)
        xx, yx, xy, yy, x0, y0 = matrix
        x0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        matrix = Matrix(xx, yx, xy, yy, x0, y0)
        self.matrix = matrix * self.matrix

    def new_sub_path(self):
        self.pos = None

    def move_to(self, x, y):
        x /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x, y = self.matrix(x, y)
        self.pos = x, y

    def line_to(self, x, y):
        x /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x, y = self.matrix(x, y)
        self.pos = (x, y)

    def curve_to(self, x0, y0, x1, y1, x2, y2):
        x0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x1 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y1 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x2 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y2 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x0, y0 = self.matrix(x0, y0)
        x1, y1 = self.matrix(x1, y1)
        x2, y2 = self.matrix(x2, y2)
        self.pos = (x2, y2)

    def rel_move_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        dx /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        self.pos = x+dx, y+dy

    def rel_line_to(self, dx, dy):
        assert self.pos is not None, "no current point"
        dx /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x, y = self.pos
        dx, dy = self.matrix.transform_distance(dx, dy)
        self.pos = x+dx, y+dy

    def rel_curve_to(self, dx0, dy0, dx1, dy1, dx2, dy2):
        assert self.pos is not None, "no current point"
        dx0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy0 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dx1 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy1 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dx2 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        dy2 /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x, y = self.pos
        dx0, dy0 = self.matrix.transform_distance(dx0, dy0)
        dx1, dy1 = self.matrix.transform_distance(dx1, dy1)
        dx2, dy2 = self.matrix.transform_distance(dx2, dy2)
        self.pos = (x+dx2, y+dy2)

    def arc(self, x, y, radius, angle1, angle2):
        x /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x1, y1 = x+radius*cos(angle2), y+radius*sin(angle2)
        self.move_to(x1, y1)

    def arc_negative(self, x, y, radius, angle1, angle2):
        x /= SCALE_CM_TO_POINT # internally huygens uses cm units
        y /= SCALE_CM_TO_POINT # internally huygens uses cm units
        x1, y1 = x+radius*cos(angle2), y+radius*sin(angle2)
        self.move_to(x1, y1)

    def close_path(self):
        pass # self.pos = ?

    def set_source_rgba(self, r, g, b, a):
        pass

    def set_line_width(self, w):
        pass

    def set_line_join(self, w):
        pass

    def set_tolerance(self, x):
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

    def set_line_cap(self, x):
        pass

    def set_fill_rule(self, arg):
        pass



def test():
    x, y = 1.2, 3.4
    radians = 1.234 * pi
    rotate = Matrix.rotate
    translate = Matrix.translate
    scale = Matrix.scale

    I = Matrix()
    assert translate(x, y) * translate(-x, -y) == I

    lhs = translate(-x, -y)*rotate(radians)*translate(x, y)
    rhs = rotate(radians, x, y)
    assert lhs == rhs, (lhs, rhs)

    
    lhs = scale(0.5, 0.5)
    rhs = translate(-2, 0) * scale(0.5, 0.5) * translate(1, 0)

    #angle = rhs.transform_angle(0.)
    #print(angle, radians)
    #assert abs(angle - radians) == 0.


if __name__ == "__main__":
    test()

