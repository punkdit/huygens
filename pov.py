#!/usr/bin/env python3

"""
Duplicate OpenGL coordinate system...

See:
https://gamedev.stackexchange.com/questions/153078/what-can-i-do-with-the-4th-component-of-gl-position
"""

import sys
from math import sin, cos, pi

import numpy

scalar = numpy.float64
EPSILON = 1e-6

class Mat(object):
    def __init__(self, cs):
        A = numpy.array(cs, dtype=scalar)
        if len(A.shape)==1:
            m = len(A)
            A.shape = (m, 1) # col vector
        assert len(A.shape)==2, A.shape
        self.A = A
        self.shape = A.shape

    def __str__(self):
        A = self.A
        rows = [', '.join(["%.6f"%x for x in row]) for row in A]
        rows = ["[%s]"%row for row in rows]
        rows = "[%s]"%("\n".join(rows),)
        rows = rows.replace(".000000", ".      ")
        return rows
    __repr__ = __str__

    def __eq__(self, other):
        other = Mat.promote(other)
        assert self.shape == other.shape
        err = numpy.abs(self.A - other.A).sum()
        return err < EPSILON

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def promote(cls, item):
        if isinstance(item, Mat):
            return item
        m = Mat(item)
        return m

    @classmethod
    def identity(cls, n):
        A = numpy.identity(n)
        return cls(A)

    def __add__(self, other):
        other = Mat.promote(other)
        assert self.shape == other.shape
        A = self.A + other.A
        return Mat(A)

    def __sub__(self, other):
        other = Mat.promote(other)
        assert self.shape == other.shape
        A = self.A - other.A
        return Mat(A)

    def __neg__(self):
        A = -self.A
        return Mat(A)

    def __mul__(self, other):
        other = Mat.promote(other)
        assert self.shape[1] == other.shape[0]
        A = numpy.dot(self.A, other.A)
        return Mat(A)

    def __rmul__(self, r):
        A = r*self.A
        return Mat(A)

    def __getitem__(self, idx):
        if type(idx) is tuple:
            return self.A[idx]
        if self.shape[1] == 1:
            idx = (idx, 0)
        return self.A[idx]

    def __setitem__(self, idx, value):
        if type(idx) is tuple:
            pass
        elif self.shape[1] == 1:
            idx = (idx, 0)
        self.A[idx] = value

    @classmethod
    def frustum(cls, left, right, bottom, top, nearval, farval):
        # mesa/src/mesa/math/m_matrix.c
        """
       GLfloat x, y, a, b, c, d;
       GLfloat m[16];
    
       x = (2.0F*nearval) / (right-left);
       y = (2.0F*nearval) / (top-bottom);
       a = (right+left) / (right-left);
       b = (top+bottom) / (top-bottom);
       c = -(farval+nearval) / ( farval-nearval);
       d = -(2.0F*farval*nearval) / (farval-nearval);  /* error? */
    
    #define M(row,col)  m[col*4+row]
       M(0,0) = x;     M(0,1) = 0.0F;  M(0,2) = a;      M(0,3) = 0.0F;
       M(1,0) = 0.0F;  M(1,1) = y;     M(1,2) = b;      M(1,3) = 0.0F;
       M(2,0) = 0.0F;  M(2,1) = 0.0F;  M(2,2) = c;      M(2,3) = d;
       M(3,0) = 0.0F;  M(3,1) = 0.0F;  M(3,2) = -1.0F;  M(3,3) = 0.0F;
    #undef M
    
       matrix_multf( mat, m, MAT_FLAG_PERSPECTIVE );
        """
        pass # TODO


    @classmethod
    def translate(cls, *args):
        "modelled after glTranslate"
        n = len(args)+1
        A = numpy.identity(n)
        for i,val in enumerate(args):
            A[i, n-1] = val
        M = cls(A)
        return M

    @classmethod
    def perspective(cls, fovy, aspect, z_near, z_far):
        "modelled after gluPerspective"
        radians = fovy / 2 * pi / 180
    
        delta_z = z_far - z_near
        sine = sin(radians)
        if (delta_z == 0) or (sine == 0) or (aspect == 0):
            return
        cotangent = cos(radians) / sine
    
        A = numpy.identity(4)
        A[0,0] = cotangent / aspect
        A[1,1] = cotangent
        A[2,2] = -(z_far + z_near) / delta_z
        #A[2,3] = -1
        A[3,2] = -1
        #A[3,2] = -2 * z_near * z_far / delta_z
        A[2,3] = -2 * z_near * z_far / delta_z
        A[3,3] = 0

        M = Mat(A)
        return M

    def normalized(self):
        A = self.A
        r = (A*A).sum()**0.5
        return Mat(A/r)

    def cross(self, other):
        assert self.shape == (3, 1)
        assert other.shape == (3, 1)
        cs = [
            self[1]*other[2] - self[2]*other[1],
            self[2]*other[0] - self[0]*other[2],
            self[0]*other[1] - self[1]*other[0]]
        return Mat(cs)

    @classmethod
    #def lookat(cls, eyex, eyey, eyez, centerx, centery, centerz, upx, upy, upz):
    def lookat(cls, eye, center, up):
        "modelled after gluLookAt"
        eye = cls.promote(eye)
        center = cls.promote(center)
        up = cls.promote(up)

        forward = center - eye
        forward = forward.normalized()
    
        side = forward.cross(up)
        side = side.normalized()
    
        up = side.cross(forward)
    
        #M = cls.identity(4)
        M = numpy.identity(4)
        M[0,0] = side[0]
        M[1,0] = side[1]
        M[2,0] = side[2]
    
        M[0,1] = up[0]
        M[1,1] = up[1]
        M[2,1] = up[2]
    
        M[0,2] = -forward[0]
        M[1,2] = -forward[1]
        M[2,2] = -forward[2]
        M = M.transpose()
        M = Mat(M)
    
        M1 = cls.translate(-eye[0], -eye[1], -eye[2])
        M = M*M1
        return M


# ----------------------------------------------------------------------

def test_perspective():
    width, height = 640, 480
    proj = Mat.identity(4)
    
    M = Mat.perspective(45., width/height, 0.1, 100.)
    proj = M * proj
    
    assert proj == Mat([
        [ 1.8106601,  0.,         0.,         0.,       ],
        [ 0.,         2.4142137,  0.,         0.,       ],
        [ 0.,         0.,        -1.002002,  -0.2002002,  ],
        [ 0.,         0.,        -1., 0.,       ]])

test_perspective()


# ----------------------------------------------------------------------


from bruhat.render.base import SCALE_CM_TO_POINT

class GItem(object):
    pass


class Flat(GItem):
    def __init__(self, pts, fill=None, stroke=None):
        self.pts = pts
        self.fill = fill
        self.stroke = stroke




class View(object):
    def __init__(self, _width=640, _height=480):
        #global width, height, viewport, proj, model
        scale = 0.05 # XXX 1./SCALE_CM_TO_POINT
        width, height = scale*_width, scale*_height
        self.viewport = (0., 0., width, height)
        self.proj = Mat.identity(4) # Projection matrix
        self.model = Mat.identity(4) # ModelView matrix
        self.gitems = []
    
    def perspective(self):
        #global proj
        width, height = self.viewport[2:]
        M = Mat.perspective(45., width/height, 0.1, 100.)
        self.proj = M * self.proj
    
    def translate(self, x, y, z): # XXX use Mat
        #global model
        M = Mat.translate(x, y, z)
        self.model = self.model*M
    
    def lookat(self, eye, center, up):
        #global model
        M = Mat.lookat(eye, center, up)
        self.model = self.model*M

    # ------------------------------------------------
    
#    def XXtransform(self, x, y, z): # XXX use Mat
#        v = [x, y, z, 1.]
#        v = self.model * v
#        v = self.proj * v
#        x, y, z, w = v
#        return x, y, z

    def trafo_view(self, point):
        assert point.shape == (3, 1)
        x, y, z = point
        v = [x, y, z, 1.]
        v = self.model * v
        return v

    def trafo_camera(self, point):
        assert point.shape == (4, 1)
        v = self.proj * point
        x, y, z, w = v
        return x, y, z

    def trafo_canvas(self, point):
        v = self.proj * point
        x, y, z, w = v
        x, y = x/w, y/w
    
        x0, y0, width, height = self.viewport
        w2, h2 = width/2, height/2
        x = x0 + w2 + x*w2
        y = y0 + h2 + y*h2
        return x, y
    
#    def XXget(self, x, y, z): # XXX use Mat
#        v = [x, y, z, 1.]
#        v = self.model * v
#        v = self.proj * v
#        x, y, z, w = v
#        x, y = x/w, y/w
#    
#        x0, y0, width, height = self.viewport
#        w2, h2 = width/2, height/2
#        x = x0 + w2 + x*w2
#        y = y0 + h2 + y*h2
#        return x, y

    def depth(self, gitem):
        pts = gitem.pts
        v = pts[0]
        #print(v)
        for v1 in pts[1:]:
            v = v + v1
        v = (1./len(pts))*v
        x, y, z = self.trafo_camera(v)
        return -z

    # -----------------------------------------
    # class Scene ?

    def append(self, face):
        self.gitems.append(face)

    def make_flat(self, pts, fill, stroke):
        pts = [self.trafo_view(p) for p in pts]
        gitem = Flat(pts, fill, stroke)
        self.append(gitem)

    def prepare_canvas(self, clip=True):
        cvs = canvas.canvas()
        cvs.append(style.linewidth.THick)
    
        x0, y0, width, height = self.viewport
        p = mkpath([(x0, y0), (x0+width, y0), (x0+width, y0+height), (x0, y0+height)])
        cvs.fill(p, [color.rgb.black])
        if clip:
            cvs.clip(p)

        cvs.append(style.linejoin.bevel)
        return cvs

    def render(self):
        cvs = self.prepare_canvas()

        gitems = list(self.gitems)
        gitems.sort(key = self.depth)
        for gitem in gitems:
            pts = [self.trafo_canvas(v) for v in gitem.pts]
            cvs.append(Polygon(pts, gitem.fill, gitem.stroke))
        return cvs



# ----------------------------------------------------------------------

from bruhat.render.front import *

def mkpath(pts, closepath=True):
    pts = [path.moveto(*pts[0])]+[path.lineto(*p) for p in pts[1:]]
    if closepath:
        pts.append(path.closepath())
    p = path.path(*pts)
    return p


def main():

    global cvs

    from bruhat import platonic
    polygon = platonic.make_octahedron()
    polygon = platonic.make_icosahedron()
    polygon = platonic.make_cube()
    polygon = platonic.make_dodecahedron()
    #polygon = [ [(0.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)], ]
    polygon = [[Mat(list(v)) for v in face] for face in polygon]
    #for face in polygon:
    #    for v in face:
    #        print(list(v))
    #    print()

    R = 6.0
    theta = 0.
    for frame in range(600):

        view = View()
        view.perspective()
        theta += 0.004*pi
        x, z = R*sin(theta), R*cos(theta)
        R += 0.01
        view.lookat([x, 1., z], [0., 0, 0], [0, 1, 0]) # eye, center, up

        stroke = (0.4, 0.4, 0.4, 1.)
        fill = (0.9, 0.8, 0., 0.8)

        view.translate(-2, 0, 0)
        for pts in polygon:
            view.make_flat(pts, fill, stroke)
        view.translate(+4, 0, 0)
        fill = (0.2, 0.2, 0.4, 0.8)
        for pts in polygon:
            view.make_flat(pts, fill, stroke)

        cvs = view.render()
        cvs.writePNGfile("frames/%.4d.png"%frame)
        if frame == 0:
            cvs.writePDFfile("frames/%.4d.pdf"%frame)

        print(".", end="", flush=True)


    print("OK")





if __name__ == "__main__":

    main()


