#!/usr/bin/env python3



def test_canvas():
    from math import pi
    from bruhat.render.front import canvas, style, path, color, trafo

    def cross(x, y):
        r = 0.1
        st = [color.rgb.blue, style.linewidth.THick, style.linecap.round]
        cvs.stroke(path.line(x-r, y-r, x+r, y+r), st)
        cvs.stroke(path.line(x-r, y+r, x+r, y-r), st)

    cvs = canvas.canvas()

    cross(0., 0.)

    cvs.text(0., 0., "hey there!")

    yield cvs


    cvs = canvas.canvas()


    p = path.path([
        path.moveto(0., 0.),
        path.arc(0., 0., 1., 0., 0.5*pi),
        path.lineto(-1., 1.),
        path.arc(-1., 0., 1., 0.5*pi, 1.0*pi),
        path.arc(-1.5, 0., 0.5, 1.0*pi, 2.0*pi),
        path.closepath()
    ])

    p = path.path(
    [ 
        path.moveto(0., 0.),
        path.arc(0., 0., 1., 0., 0.5*pi),
        path.lineto(-1., 1.), path.arc(-1., 0., 1., 0.5*pi, 1.0*pi),
        path.arc(-1.5, 0., 0.5, 1.0*pi, 2.0*pi), path.closepath() ])

    cvs.fill(p, [color.rgb.red, trafo.scale(0.8, 0.8)])
    cvs.stroke(p, [color.rgb.black, style.linewidth.THick])

    cross(0., 0.)
    cross(-1.2, 1.2)

    yield cvs

    if 0:
        x, y, r, angle1, angle2 = 0., 0., 1., 0., 0.5*pi
        p = arc_to_bezier(x, y, r, angle1, angle2, danglemax=pi/2.)
        cvs.stroke(p, [color.rgb.white])
    
        x, y, r, angle1, angle2 = 0., 0., 1., -0.5*pi, 0.
        p = arc_to_bezier(x, y, r, angle1, angle2, danglemax=pi/2.)
        cvs.stroke(p, [color.rgb.red])

    #cvs.writePDFfile("output.pdf")


def test_turtle():

    from bruhat.render.front import canvas, style, path, color, trafo
    from bruhat.render.turtle import Turtle

    cvs = canvas.canvas()
    extra = [style.linewidth.THIck, color.rgb(0.2, 0.6, 0.2, 0.6),
        style.linejoin.bevel]
    turtle = Turtle(cvs=cvs, extra=extra)

    n = 8
    angle = 360. / n
    R = 3.0
    for i in range(n):
        turtle.fwd(1.*R)
        turtle.left((1./3)*angle)
        turtle.back(0.5*R)
        turtle.left((1./3)*angle)
        turtle.back(0.7*R)
        turtle.left((1./3)*angle)
        turtle.stroke(extra)

    yield cvs

    cvs = canvas.canvas()
    turtle = Turtle(cvs=cvs, extra=extra)

    if 0:
        for i in range(24*2):
            turtle.left(320, 0.6*R)
            turtle.left(-60, 0.3*R)
            turtle.right(90, 0.6*R)
            turtle.stroke(extra)

    for i in range(1):
        turtle.fwd(2.)
        turtle.right(300, 1.)
    turtle.arrow(0.4)
    turtle.stroke(extra)

    yield cvs







