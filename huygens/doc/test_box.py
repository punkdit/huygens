#!/usr/bin/env python3


def test_box():


    #  
    # [<<< table of contents](index.html)
    #
    #  ---
    #
    # Layout with Box's
    # =================
    # 

    from random import random, seed
    seed(0)
    from huygens.front import canvas, path
    from huygens.box import (Box, EmptyBox, CanBox, TextBox, 
        HBox, VBox, OBox, TableBox, FillBox, MarginBox, AlignBox)

    # First we set a debug flag so we can see the shape of every box
    # 

    Box.DEBUG = True

    # Every Box has an anchor point, shown with a cross.
    # Then, the distance to the four sides of the box
    # are stored as attributes `top`, `bot`, `left`, and `right`.

    box = EmptyBox(top=0.5, bot=1.0, left=0.2, right=2.)

    # To render a box onto a canvas:
    cvs = canvas.canvas()
    box.render(cvs)

    # We can then call `cvs.writeSVGfile("output.svg")` to save an svg file:
    
    yield box, "empty"

    #--------------------------------------------------

    box = TextBox("Hey there!")
    yield box, "text"

    #--------------------------------------------------

    cvs = canvas.canvas()
    cvs.stroke(path.line(0., 0., 1., 1.))
    cvs.text(0., 0., "hello everyone")
    box = CanBox(cvs)
    yield box, 'canbox'

    #--------------------------------------------------

    # You cannot use the same box more than once in a container Box:

    box = TextBox("hello")
    box = HBox([box, box]) # FAIL 
    #yield box, "hbox-fail" # raises assert error

    #--------------------------------------------------

    box = HBox("geghh xxde xyeey".split())
    yield box, "hbox-text"

    #--------------------------------------------------

    box = VBox("geghh xxde xyeey".split())
    yield box, "vbox-text"

    #--------------------------------------------------

    r = 1.0
    a = EmptyBox(top=r, bot=r)
    b = EmptyBox(top=r, bot=r)
    c = EmptyBox(left=r, right=r)
    #box = StrictVBox([a, c])
    box = VBox([a, c])
    yield box, 'vbox-empty'

    #--------------------------------------------------

    box = OBox([
        EmptyBox(.4, .1, 0., 2.2),
        EmptyBox(.3, 0., .5, 2.5),
        EmptyBox(1., .5, .5, .5),
        FillBox(.2, .2),
    ])
    yield box, "obox"


    #--------------------------------------------------

    box = HBox([
        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
    ])
    yield box, "hbox-vbox"


    #--------------------------------------------------

    box = TableBox([
        [EmptyBox(.4, .1, 0.2, 2.2), EmptyBox(.3, 1.2, .5, 2.5),],
        [EmptyBox(.8, .1, 0.4, 1.2), EmptyBox(.5, 0.4, .5, 1.5),]
    ])
    yield box, "table"


    #--------------------------------------------------

    def rnd(a, b):
        return (b-a)*random() + a

    a, b = 0.2, 1.0
    rows = []
    for row in range(3):
        row = []
        for col in range(3):
            box = EmptyBox(rnd(a,b), rnd(a,b), rnd(a,b), rnd(a,b))
            row.append(box)
        rows.append(row)

    box = TableBox(rows)
    yield box, "table-2"


    #--------------------------------------------------

    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            box = TextBox(("xbcgef"[i+j])*(i+1)*(j+1))
            box = MarginBox(box, 0.1)
            box = AlignBox(box, "north")
            row.append(box)
        row.append(EmptyBox(bot=1.))
        rows.append(row)
    box = TableBox(rows)
    yield box, "table-3"


    #--------------------------------------------------

    a, b = 0.2, 2.0
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
        boxs.append(TextBox("Hig%d !"%row))
        box = HBox(boxs)
        rows.append(box)
    box = VBox(rows)
    yield box, "table-4"




