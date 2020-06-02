#!/usr/bin/env python3

"""
string diagram's
"""

from bruhat.render.back import canvas, path
from bruhat.render.boxs import Box, HBox, VBox


class Port(object):
    def __init__(self, dia, idx):
        self.dia = dia
        self.idx = idx


#class Dia(Box):
#    def __init__(self, n_in=0, n_out=0):
#        self.ins = [Port(self, i) for i in range(n_in)]
#        self.outs = [Port(self, i) for i in range(n_out)]

SIZE = 1.0

class Dia(Box):

    def __init__(self, n_top=0, n_bot=0, n_left=0, n_right=0, size=SIZE):
        self.n_top = n_top
        self.size = size

    def on_layout(self, cvs, system):
        Box.on_layout(self, cvs, system)
        radius = 0.5*self.size

        # center align:
        system.add(self.top >= self.bot)
        system.add(self.bot >= radius) # slack
        system.add(self.left == self.right)
        system.add(self.right >= radius) # slack

    def on_render(self, cvs, system):
        Box.on_render(self, cvs, system)
        r = 0.1
        x = system[self.x]
        y = system[self.y]
        #cvs.stroke(path.rect(
        cvs.stroke(path.circle(x, y, r))

    def __mul__(self, other):
        return VDia([self, other])

    def __matmul__(self, other):
        return HDia([self, other])


class HDia(Dia):
    def __init__(self, dias):
        for dia in dias:
            assert isinstance(dia, Dia)



def test():

    a = Dia()
    b = Dia()
    
    box = HBox([a, b])

    cvs = canvas.canvas()
    box.render(cvs)
    cvs.writePDFfile("test_diagram.pdf")



if __name__ == "__main__":

    test()


