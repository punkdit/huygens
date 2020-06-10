#!/usr/bin/env python3


def test_snake():

    #  
    # [<<< table of contents](index.html)
    #
    #  ---
    #
    # Diagrams
    # ========
    # 
    #  
    # 

    from bruhat.render import canvas
    from bruhat.render.box import HBox
    from bruhat.render.diagram import HDia, VDia, VWire, Cap, Cup, SIZE

    top = HDia([VWire(), Cap()])
    #mid = HDia([VWire(), VWire(), VWire()])
    bot = HDia([Cup(), VWire()])
    lsnake = VDia([top, bot])

    top = HDia([Cap(), VWire()])
    bot = HDia([VWire(), Cup()])
    rsnake = VDia([top, bot])

    boxs = [lsnake, "$=$", VWire(min_height=SIZE), "$=$", rsnake]
    dia = HBox(boxs, align="center")

    cvs = canvas.canvas()
    dia.render(cvs)
    cvs.writeSVGfile("output.svg")

    yield cvs


def test_spider():

    # Instead of explicitly using `VDia` and `HDia`
    # we can use the operators `*` and `@`, respectively.
    # The `*` operator composes top-down, like `VBox`.
    # The `@` operator composes left to right, like `HBox`.

    from bruhat.render.box import Box
    from bruhat.render.diagram import Spider, VWire, Cap, Cup
    Box.DEBUG = True

    box = Spider(2, 2) @ VWire()
    #box = box * (VWire() @ Spider(2, 2))
    box = box * (Spider(3, 1))
    box = box @ VWire()
    #box = box * Cup()
    box = box * Spider(2, 0, weight=0.9)
    #box = (Cap() @ Cap()) * box
    box = (Cap() @ Spider(0, 1) @ Spider(0, 1)) * box

    yield box


def test_relation():

    from bruhat.render.box import Box
    from bruhat.render.diagram import Spider, VWire, Cap, Cup, Relation
    Box.DEBUG = True
    box = Relation(3, 4, topbot=[(0, 0), (1, 0), (2, 1), (1, 2), (2, 3)])
    box = box * (Cup() @ Cup())
    box = (VWire() @ Spider(0, 1) @ VWire()) * box
    box = Cap() * box

    yield box


def test_yang_baxter():

    from bruhat.render.box import Box, HBox
    from bruhat.render.diagram import VWire, Braid
    Box.DEBUG = False

    Id = VWire

    scale = 2.0
    w = 1.5*scale
    h = 0.5*scale
    s12 = lambda : Braid(min_width=w, min_height=h) @ Id(min_height=h, min_width=h)
    s23 = lambda : Id(min_height=h, min_width=h) @ Braid(min_width=w, min_height=h)

    lhs = s12() * s23() * s12()
    #lhs = lhs @ Braid()
    rhs = s23() * s12() * s23()
    box = HBox([lhs, "$=$", rhs], align="center")

    yield box


def XXXtest_braid():
    from random import shuffle, seed, randint
    from operator import matmul
    from functools import reduce

    seed(1)

    scale = 0.5
    w = 1.4*scale
    h = 1.8*scale
    Id = lambda : VWire(min_height=0.5, min_width=0.5)
    Swap = lambda inverse : Braid(inverse=inverse, min_width=w, min_height=h, space=0.7)

    box = None
    m, n = 4, 4
    k = 2*m+n
    for count in range(6):
        items = [Swap(randint(0,1)) for k in range(m)] + [Id() for k in range(n)]
        shuffle(items)
        row = reduce(matmul, items)
        if box is None:
            box = row
        else:
            box = box * row

    #box = Id() @ box
    lhs = reduce(matmul, [Id() for i in range(k)])
    box = lhs @ box
    rels = [(i, 2*k-i-1) for i in range(k)]
    #rels = [(i, i+k) for i in range(k)]
    rel = Relation(0, 2*k, botbot=rels, weight=200.0)
    box = rel * box
    rels = [(i, 2*k-i-1) for i in range(k)]
    rel = Relation(2*k, 0, toptop=rels, weight=200.0)
    box = box * rel

    #rect = RectBox(box, bg=color.rgb(0.9, 0.9, 0.3, 0.6))

    Box.DEBUG = False

    cvs = canvas.canvas()
    #cvs.append(trafo.rotate(pi/2))

    system = box.layout(cvs)

    def rectbox(box):
        x = system[box.llx]
        y = system[box.lly]
        width = system[box.width]
        height = system[box.height]
        return x, y, width, height

    def fillbox(box, st):
        rect = rectbox(box)
        p = path.rect(*rect)
        cvs.fill(p, st)

    #fillbox(box, [color.rgb(0.9, 0.8, 0.5)])

    sub = box[0][1][1]
    x = conv(system[box.llx], system[box.urx])
    y = system[sub.lly]
    width = system[box.urx] - x
    height = system[sub.ury] - y
    p = path.rect(x, y, width, height)
    cvs.fill(p, [color.rgb(0.9, 0.9, 0.6)])
    cvs.stroke(p, [style.linewidth.thick])

    cvs.append(style.linewidth.THICk)
    #cvs.append(color.rgb(0.2,0.5,0.2))
    box.render(cvs)

    cvs.append(style.linewidth.thick)
    cvs.append(color.rgb(0.9,0.9,0.9))
    box.render(cvs)

    #cvs.writePDFfile("test_diagram.pdf")

    yield cvs



if __name__ == "__main__":

    from bruhat.argv import argv
    if argv.profile:
        import cProfile as profile
        profile.run("test()")

    else:

        test_snake()
        test()

    print("OK\n")


