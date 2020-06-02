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



