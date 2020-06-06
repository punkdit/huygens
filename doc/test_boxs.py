#!/usr/bin/env python3

from bruhat.render.boxs import *

"hi there"

def test_build():

    def rnd(a, b):
        return (b-a)*random() + a


    box = EmptyBox(1., 1., 1., 1.)
    yield box, "empty"

    box = TextBox("Hey there!")
    #box = TextBox(".")
    yield box, "text"

    box = HBox("geghh xxde xyeey".split())
    yield box, "hbox-text"

    box = VBox("geghh xxde xyeey".split())
    yield box, "vbox-text"


    box = OBox([
        EmptyBox(.4, .1, 0., 2.2),
        EmptyBox(.3, 0., .5, 2.5),
        EmptyBox(1., .5, .5, .5),
        FillBox(.2, .2),
    ])
    yield box, "obox"


    box = HBox([
        VBox([TextBox(text) for text in "xxx1 ggg2 xxx3 xx4".split()]),
        VBox([TextBox(text) for text in "123 xfdl sdal".split()]),
    ])
    yield box, "hbox-vbox"


    box = TableBox([
        [EmptyBox(.4, .1, 0.2, 2.2), EmptyBox(.3, 1.2, .5, 2.5),],
        [EmptyBox(.8, .1, 0.4, 1.2), EmptyBox(.5, 0.4, .5, 1.5),]
    ])
    yield box, "table"


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


    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            box = TextBox(choice("xbcgef")*(i+1)*(j+1))
            box = MarginBox(box, 0.1)
            box = AlignBox(box, "north")
            row.append(box)
        row.append(EmptyBox(bot=1.))
        rows.append(row)
    box = TableBox(rows)
    yield box, "table-3"


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
        #boxs.append(TextBox(r"$\to$"))
        box = HBox(boxs)
        rows.append(box)
    box = VBox(rows)
    yield box, "table-4"


def test():

    #Box.DEBUG = argv.get("debug") or argv.get("DEBUG")
    Box.DEBUG = True
    EmptyBox.DEBUG = True

    for (box, name) in test_build():

        try:
            print("rendering", name)
            cvs = Canvas()
            cvs.append(Scale(2.0))
            box.render(cvs)
            #cvs = Canvas([Scale(2.0, 2.0)]+cvs.items)
            cvs.writePDFfile("pic-%s.pdf" % name)
            cvs.writeSVGfile("pic-%s.svg" % name)
            print()
        except:
            print("render failed for %r"%name)
            raise



if __name__ == "__main__":
    test()


