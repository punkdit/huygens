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


def init(_width=640, _height=480):
    global width, height, viewport, proj, model
    scale = 0.05
    width, height = scale*_width, scale*_height
    viewport = (0., 0., width, height)
    proj = Mat.identity(4)
    model = Mat.identity(4)

def perspective():
    global proj
    M = Mat.perspective(45., width/height, 0.1, 100.)
    proj = M * proj

def translate(x, y, z):
    global model
    M = Mat.translate(x, y, z)
    model = model*M

def lookat(eye, center, up):
    global model
    M = Mat.lookat(eye, center, up)
    model = model*M


def transform(x, y, z):
    v = [x, y, z, 1.]
    v = model * v
    v = proj * v
    x, y, z, w = v
    return x, y, z


def get(x, y, z):
    v = [x, y, z, 1.]
    v = model * v
    v = proj * v
    x, y, z, w = v
    x, y = x/w, y/w

    x0, y0, width, height = viewport
    w2, h2 = width/2, height/2
    x = x0 + w2 + x*w2
    y = y0 + h2 + y*h2
    return x, y


# ----------------------------------------------------------------------

from bruhat.render.front import *

def mkpath(pts, closepath=True):
    pts = [path.moveto(*pts[0])]+[path.lineto(*p) for p in pts[1:]]
    if closepath:
        pts.append(path.closepath())
    p = path.path(*pts)
    return p


def mk_poly(pts, deco=[]):
    pts = [get(*p) for p in pts]
    p = mkpath(pts)
    cvs.fill(p, deco)
    cvs.stroke(p)


def test():

    global cvs
    cvs = canvas.canvas()

    p = mkpath([(0., 0.), (width, 0.), (width, height), (0., height)])
    cvs.fill(p, [color.rgb.black])
    cvs.clip(p)

    # eye, center, up
    lookat( [5., 5., 5.], [0., 0, 0], [0, 1, 0])
    #lookat( [0., 0., 0.], [0., 0, -1], [0, 1, 0]) # OK
    #lookat( [0., 0., 0.], [0., 1, -1], [0, 1, 0]) # OK
    #lookat( [0., 0., 1.], [0., 0, 0], [0, 1, 0]) # OK
    #lookat( [0., 0., 0.], [0., 1, -1], [0, 1, 0]) # OK
    #lookat( [1., 1., 1.], [0., 0, 0], [0, 1, 0]) # OK

    translate(-1.5, 0.0, -6.0)

    #glColor(1., 1., 0.)

    mk_poly( [
        (0.0, 1.0, -1.0),   
        (1.0, -1.0, -1.0),     
        (-1.0, -1.0, -1.0)], [color.rgb(1., 1., 0)])

    #glColor(1., 1., 1.)

    c = color.rgb(1., 1., 1.)
    mk_poly( [
        (0.0, 1.0, 0.0),       
        (1.0, -1.0, 0.0),      
        (-1.0, -1.0, 0.0)], [c])

    translate(3.0, 0.0, 0.0)

    mk_poly( [
        (-1.0, 1.0, 0.0),      
        (1.0, 1.0, 0.0),       
        (1.0, -1.0, 0.0),      
        (-1.0, -1.0, 0.0)], [c])

    #cvs.dump()
    cvs.writePDFfile("output.pdf")

    print("OK")
    

frame = 0
def save():
    global frame
    cvs.writePNGfile("frames/%.4d.png"%frame)
    cvs.writePDFfile("frames/%.4d.pdf"%frame)
    frame += 1


def depth(face):
    v = face[0]
    #print(v)
    for v1 in face[1:]:
        v = v + v1
    v = (1./len(face))*v
    x, y, z = transform(*tuple(v))
    return -z


class Polygon(Item):
    def __init__(self, pts, stroke=None, fill=None):
        Item.__init__(self)
        assert len(pts)>1
        self.pts = [(x*SCALE_CM_TO_POINT, y*SCALE_CM_TO_POINT)
            for (x, y) in pts]
        self.stroke = stroke
        self.fill = fill

    def process_cairo(self, cxt):
        pts = self.pts
        cxt.save() # <--------- save
        cxt.set_line_width(0.5)

        fill = self.fill or (0., 0., 0., 1.)
        cxt.set_source_rgba(*fill)
        x, y = pts[0]
        cxt.move_to(x, -y)
        for (x, y) in pts[1:]:
            cxt.line_to(x, -y)
        cxt.close_path()
        cxt.fill()

        stroke = self.stroke or (1., 1., 1., 1.)
        cxt.set_source_rgba(*stroke)
        x, y = pts[0]
        cxt.move_to(x, -y)
        for (x, y) in pts[1:]:
            cxt.line_to(x, -y)
        cxt.close_path()
        cxt.stroke()

        cxt.restore() # <------- restore



def main():

    global cvs

    from bruhat import platonic
    faces = platonic.make_octahedron()
    faces = platonic.make_icosahedron()
    faces = platonic.make_cube()
    faces = platonic.make_dodecahedron()
    #faces = [ [(0.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)], ]
    faces = [[Mat(list(v)) for v in face] for face in faces]
    #for face in faces:
    #    for v in face:
    #        print(list(v))
    #    print()

    stroke = (0.8, 0.4, 0.0, 1.)
    fill = (1.0, 0.7, 0., 0.8)

    R = 5.0
    theta = 0.
    for frame in range(400):

        init()
        perspective()
        theta += 0.01*pi
        x, z = R*sin(theta), R*cos(theta)
        R += 0.1
        lookat([x, 1., z], [0., 0, 0], [0, 1, 0]) # eye, center, up

        cvs = canvas.canvas()
    
        p = mkpath([(0., 0.), (width, 0.), (width, height), (0., height)])
        cvs.fill(p, [color.rgb.black])
        cvs.clip(p)

        #cvs.append(trafo.scale(10., 10.))
        cvs.append(style.linejoin.bevel)
        #cvs.fill(path.rect(-2, -2, 4, 4), [color.rgb.gray])

        #translate(-3, 0, 0)
        for i in range(3):
            fs = list(faces)
            fs.sort(key = depth)
            for face in fs:
                pts = [get(*tuple(v)) for v in face]
                cvs.append(Polygon(pts, stroke, fill))
            translate(3., 0., 1.)
            break

        save()
        print(".", end="", flush=True)


    print("OK")





if __name__ == "__main__":

    main()


