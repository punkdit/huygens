#!/usr/bin/env python3


def test_relation():

    #  
    # [<<< table of contents](index.html)
    #
    #  ---
    #
    # Diagrams
    # ========
    # 
    # Diagrams are built from `Box`'s with some extra variables
    # that describe "input-output ports" on each of the four
    # edges: top, bot, left and right.

    from huygens.box import Box
    #from huygens.diagram import Spider, VWire, Cap, Cup, Relation
    from huygens.diagram import Relation
    Box.DEBUG = True
    box = Relation(2, 4, topbot=[(0, 2), (1, 3)], botbot=[(0, 1)])

    yield box

    # This diagram has two ports on the top edge, and three
    # along the bot edge.
    #
    # Just like `Box`'s, we can compose diagrams vertically
    # and horizontally, as long as the number of ports agree
    # along the connecting edge.

    from huygens.diagram import VDia, HDia

    a_box = Relation(0, 2, botbot=[(0, 1)])
    b_box = Relation(2, 4, topbot=[(0, 2), (1, 3)], botbot=[(0, 1)])
    c_box = Relation(4, 0, toptop=[(0, 3), (1, 2)])

    box = HDia([a_box, b_box, c_box])

    yield box

    box = VDia([a_box, b_box, c_box])

    yield box

    # Each shared edge between two diagrams produces new 
    # constraints specifying that the position of the ports agree.


def test_snake():

    # Composing diagrams and box's
    # -----------------------------
    #
    # The base class for a diagram is `Dia`. Diagrams are
    # also `Box`'s so we can stick them anywhere we can use a `Box`.

    from huygens import config, canvas
    from huygens.box import Box, HBox
    from huygens.diagram import HDia, VDia, VWire, Cap, Cup, get_size
    Box.DEBUG = False

    config(text="pdftex")

    top = HDia([VWire(), Cap()])
    #mid = HDia([VWire(), VWire(), VWire()])
    bot = HDia([Cup(), VWire()])
    lsnake = VDia([top, bot])

    top = HDia([Cap(), VWire()])
    bot = HDia([VWire(), Cup()])
    rsnake = VDia([top, bot])

    boxs = [lsnake, "$=$", VWire(min_height=get_size()), "$=$", rsnake]
    dia = HBox(boxs, align="center")

    yield dia

    # If we do this again with DEBUG you can see how
    # the underlying `Box`s are put together.

    Box.DEBUG = True

    # Currently, the anchor inside each `Dia` is not constrained and
    # so is free to wander around inside the `Box`.
    # Maybe this will change in the future.

    yield dia
    
    # If we use a `StrictHBox` it will stretch the `VWire()`
    # but then we need to put the text in a `SlackBox`.

    from huygens.box import SlackBox, StrictHBox
    boxs = [lsnake, SlackBox("$=$"), VWire(), SlackBox("$=$"), rsnake]
    dia = StrictHBox(boxs, align="center")

    yield dia, "hbox-slack-dia"


def test_spider():

    # Instead of explicitly using `VDia` and `HDia`
    # we can use the operators `*` and `@`, respectively.
    # The `*` operator composes top-down, like `VBox`.
    # The `@` operator composes left to right, like `HBox`.

    from huygens.box import Box
    from huygens.diagram import Spider, VWire, Cap, Cup
    Box.DEBUG = False

    box = Spider(2, 2) @ Spider(1, 1)
    #box = box * (VWire() @ Spider(2, 2))
    box = box * (Spider(3, 1))
    box = box @ Spider(1, 1)
    #box = box * Cup()
    #box = (Cap() @ Cap()) * box
    box = (Spider(0, 2) @ Spider(0, 1) @ Spider(0, 1)) * box
    box = box * Spider(2, 0, weight=0.9)

    yield box

    # Notice we used `weight=0.9` for the spider at the bottom,
    # which is less than the default weight of 1.0.
    # Otherwise this spider starts to push the rest of the
    # diagram around:

    box = Spider(2, 2) @ Spider(1, 1)
    box = box * (Spider(3, 1))
    box = box @ Spider(1, 1)
    box = (Spider(0, 2) @ Spider(0, 1) @ Spider(0, 1)) * box
    box = box * Spider(2, 0)

    yield box

    # The reason becomes clearer whith DEBUG switched on.

    Box.DEBUG = True
    yield box
   
    # The pushy spider is just trying to space its legs evenly
    # inside its large box.


def test_yang_baxter():

    from huygens.box import Box, HBox
    from huygens.diagram import VWire, Braid
    Box.DEBUG = False

    Id = VWire

    scale = 2.0
    w = 1.5*scale
    h = 1.0*scale
    s12 = lambda : Braid(min_width=w, min_height=h) @ Id(min_height=h, min_width=h)
    s23 = lambda : Id(min_height=h, min_width=h) @ Braid(min_width=w, min_height=h)

    lhs = s12() * s23() * s12()
    #lhs = lhs @ Braid()
    rhs = s23() * s12() * s23()
    box = HBox([lhs, "$=$", rhs], align="center")

    yield box, "yang-baxter"


def test_braid_3():
    from huygens.box import Box, HBox
    from huygens.diagram import VWire, Braid
    Box.DEBUG = False

    Id = VWire

    scale = 2.0
    w = 0.8*scale
    h = 0.5*scale
    s1 = lambda : Braid(min_width=w, min_height=h) @ Id(min_height=h, min_width=h)
    s2 = lambda : Id(min_height=h, min_width=h) @ Braid(min_width=w, min_height=h)

    from operator import mul
    from functools import reduce

    word = [[s1, s2][i%2]() for i in range(6)]
    box = reduce(mul, word)

    box = HBox(["$Z =$", box], align="center")

    yield box, "braid-Z"



def test_braid():

    from random import shuffle, seed, randint
    from operator import matmul
    from functools import reduce

    from huygens import canvas, color, style, path
    from huygens.box import Box, HBox
    from huygens.diagram import VWire, Braid, Relation

    seed(1)

    scale = 0.7
    w = 1.0*scale
    h = 1.8*scale
    Id = lambda : VWire(min_height=scale, min_width=scale)
    Swap = lambda inverse : Braid(inverse=inverse, min_width=w, min_height=h, space=0.5)

    box = None
    m, n = 3, 3
    k = 2*m+n
    for count in range(3):
        items = [Swap(randint(0,1)) for k in range(m)] + [Id() for k in range(n)]
        shuffle(items)
        row = reduce(matmul, items)
        if box is None:
            box = row
        else:
            box = box * row

    # This is what the `box` looks like now:

    yield box, "braid-strands"

    # Now we take the closure of this braid:

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

    yield box

    # Draw this now with some more fancy tricks.

    cvs = canvas.canvas()

    system = box.layout(cvs)

    sub = box[0][1][1]
    x = 0.5*(box.llx + box.urx)
    y = sub.lly
    width = box.urx - x
    height = sub.ury - y
    p = path.rect(x, y, width, height)
    cvs.fill(p, [color.rgb(0.9, 0.9, 0.6)])
    cvs.stroke(p, [style.linewidth.thick])

    system.refresh() # refresh for a new render

    cvs.append(style.linewidth.THICk)
    #cvs.append(color.rgb(0.2,0.5,0.2))
    box.render(cvs)

    cvs.append(style.linewidth.thick)
    cvs.append(color.rgb(0.9,0.0,0.0))
    box.render(cvs)

    #cvs.writePDFfile("test_diagram.pdf")

    yield cvs




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
    x = 0.5*(system[box.llx] + system[box.urx])
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

    from huygens import argv
    if argv.profile:
        import cProfile as profile
        profile.run("test()")

    else:

        test_snake()
        test()

    print("OK\n")


