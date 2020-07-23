#!/usr/bin/env python3

from huygens import config
#config(text="xetex") # works, but pdf-->svg buggy on my system, lines too thick
config(text="xelatex") # works
#config(text="pdftex") # works
#config(text="pdflatex") # works

from huygens.front import *


north = [text.halign.boxcenter, text.valign.top]
northeast = [text.halign.boxright, text.valign.top]
northwest = [text.halign.boxleft, text.valign.top]
south = [text.halign.boxcenter, text.valign.bottom]
southeast = [text.halign.boxright, text.valign.bottom]
southwest = [text.halign.boxleft, text.valign.bottom]
east = [text.halign.boxright, text.valign.middle]
west = [text.halign.boxleft, text.valign.middle]
center = [text.halign.boxcenter, text.valign.middle]


x, y = 0., 0.
h = 0.7

def show(text, attrs=[]):
    global x, y
    r = 0.1
    st = [color.rgb.red]
    cvs.stroke(path.line(x, y-r, x, y+r), st)
    cvs.stroke(path.line(x-r, y, x+r, y), st)
    cvs.text(x, y, text, attrs)
    y -= h


cvs = canvas.canvas()

show("great!")
show("great!", [text.halign.boxright])
show("great!", [text.halign.boxcenter])

if 1:
    x, y = 2., 0.
    show("great!", [text.valign.top])
    show("great!", [text.valign.middle])
    show("great!", [text.valign.bottom])
    
    x, y = 6., 0.
    show("north", north)
    show("northeast", northeast)
    show("northwest", northwest)
    show("south", south)
    show("southeast", southeast)
    show("southwest", southwest)
    show("east", east)
    show("west", west)
    show("center", center)

if 1:
    x, y = 10., 0
    show("tiny", [text.size.tiny])
    show("script", [text.size.script])
    show("footnote", [text.size.footnote])
    show("small", [text.size.small])
    show("normal", [text.size.normal])
    show("large", [text.size.large])
    show("Large", [text.size.Large])
    show("LARGE", [text.size.LARGE])
    show("huge", [text.size.huge])
    show("Huge", [text.size.Huge])


cvs.writePDFfile("text.pdf")


