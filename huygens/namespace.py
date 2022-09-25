#!/usr/bin/env python3

from huygens.front import *
from huygens.turtle import mkpath


def conv(alpha, a, b):
    return (1.-alpha)*a + alpha*b


red = color.rgb(1.0, 0.0, 0.0)
darkred = color.rgb(0.8, 0.2, 0.2)
green = color.rgb(0.0, 0.6, 0.0)
yellow = color.rgb(1.0, 1.0, 0.0)
blue = color.rgb(0.2, 0.2, 0.7)
white = color.rgb(1., 1., 1.)
grey = color.rgb(0.8, 0.8, 0.8)
darkgrey = color.rgb(0.5, 0.5, 0.5)
black = color.rgb(0, 0, 0)

st_red = [red]
st_darkred = [darkred]
st_green = [green]
st_blue = [blue]
st_white = [white]
st_grey = [grey]
st_darkgrey = [darkgrey]
st_black = [black]

st_round = [style.linecap.round, style.linejoin.round]

st_dashed = [style.linestyle.dashed]
st_dotted = [style.linestyle.dotted]

st_north = [text.halign.boxcenter, text.valign.top]
st_northeast = [text.halign.boxright, text.valign.top]
st_northwest = [text.halign.boxleft, text.valign.top]
st_south = [text.halign.boxcenter, text.valign.bottom]
st_southeast = [text.halign.boxright, text.valign.bottom]
st_southwest = [text.halign.boxleft, text.valign.bottom]
st_east = [text.halign.boxright, text.valign.middle]
st_west = [text.halign.boxleft, text.valign.middle]
st_hcenter = [text.halign.boxcenter]
st_center = [text.halign.boxcenter, text.valign.middle]

st_footnote = [text.size.footnote] # does not work...??
st_small = [text.size.small] # does not work...??
st_large = [text.size.large] # does not work...??

st_arrow = [deco.earrow]

orange = color.rgb(0.8, 0.2, 0.0)
st_arrow = [deco.earrow()]
st_rarrow = [deco.earrow(reverse=True)]

thin = style.linewidth.thin
normal = style.linewidth.normal
st_normal = [normal]
st_THIN = [style.linewidth.THIN]
st_THIn = [style.linewidth.THIn]
st_THin = [style.linewidth.THin]
st_Thin = [style.linewidth.Thin]
st_thin = [style.linewidth.thin]
st_thick = [style.linewidth.thick]
st_Thick = [style.linewidth.Thick]
st_THick = [style.linewidth.THick]
st_THIck = [style.linewidth.THIck]
st_THICk = [style.linewidth.THICk]
st_THICK = [style.linewidth.THICK]


lw = style.linewidth

