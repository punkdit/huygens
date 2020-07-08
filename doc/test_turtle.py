#!/usr/bin/env python3


#st = [color.rgb.blue, style.linewidth.THick, style.linecap.round]
#attrs = [style.linewidth.THIck, color.rgb(0.2, 0.6, 0.2, 0.6), style.linejoin.bevel]


def test_turtle():

    #  
    # [<<< table of contents](index.html)
    #
    #  ---
    #
    # Turtle graphics
    # ===============
    # Use a turtle to keep track of a path as you build it.

    from huygens import canvas, style, color
    from huygens.turtle import Turtle

    cvs = canvas.canvas()
    turtle = Turtle(cvs=cvs)

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
        turtle.stroke()

    cvs.writeSVGfile("output.svg")

    yield cvs

    # We can change the stroke style that the turtle uses
    # to a thick green line.

    attrs = [style.linewidth.THIck, color.rgb(0.2, 0.6, 0.2)]

    cvs = canvas.canvas()
    turtle = Turtle(cvs=cvs, attrs=attrs)

    turtle.fwd(2.)
    turtle.right(300, 1.)
    turtle.fwd(2.)
    turtle.arrow(0.4)
    turtle.stroke()

    yield cvs


    cvs = canvas.canvas()
    turtle = Turtle(cvs=cvs, attrs=attrs)

    R = 1.0
    for i in range(24*2):
        turtle.left(320, 0.6*R)
        turtle.left(-60, 0.3*R)
        turtle.right(90, 0.6*R)
        turtle.stroke(attrs=attrs)

    yield cvs

