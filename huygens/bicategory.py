#!/usr/bin/env python

"""
string diagrams in a bicategory.

For the theory:
https://stringdiagram.com/wp-content/uploads/2024/08/graphicaltheoryofmonadsv2.0.pdf

(note to self. previous version: RiemannCodes2022/bicategory.py)

"""

from string import ascii_lowercase, ascii_uppercase
from functools import reduce
import operator 

import warnings
warnings.filterwarnings('ignore')

import sys
from time import time
start_time = time()

from huygens.namespace import *

#from huygens.box import *
#from huygens.diagram import Spider, TBone, VWire, Relation

# wrap this in some classes:
from huygens import box, diagram
from huygens.box import CVSBox, Box

#lightgrey = 0.9*white
#grey = 0.8*white

diagram.config(vflow="up") # argh... don't change this
diagram.config(size=0.7)


class Cell0(object):
    def __init__(self, name="n", st=st_white):
        self.name = name
        self.st = st # fill 
    def __str__(self):
        return self.name
    def __eq__(self, other):
        assert isinstance(other, Cell0)
        return str(self) == str(other)


class Cell1(object):
    def __init__(self, tgt, src, name="X", st=st_black):
        assert isinstance(tgt, Cell0)
        assert isinstance(src, Cell0)
        self.tgt = tgt
        self.src = src
        self.name = name
        self.st = st # stroke

#    def on_construct(self):
#        return diagram.Spider(1, 1, top_attrs=[self.st], bot_attrs=[self.st])
#        return diagram.VWire(attrs=self.st)

    def __str__(self):
        return "(%s<--%s--%s)"%(self.tgt, self.name, self.src)
    def __eq__(self, other):
        assert isinstance(other, Cell1)
        return str(self) == str(other)

    def __lshift__(self, other):
        lhs = self.cells if isinstance(self, HCell1) else [self]
        rhs = other.cells if isinstance(other, HCell1) else [other]
        return HCell1(lhs+rhs)

    def __len__(self):
        return int(self.st is not None)

    def __getitem__(self, idx):
        if self.st is not None:
            return [self][idx]
        return [][idx]


class HCell1(Cell1):
    def __init__(self, cells):
        tgt = cells[0].tgt
        src = cells[-1].src
        name = "<<".join(cell.name for cell in cells)
        Cell1.__init__(self, tgt, src, name, None)
        self.cells = list(cells)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx]


class Lattice(object):
    def __init__(self, o_ports, i_ports, pairs, paths=[]):
        no = len(o_ports)
        ni = len(i_ports)
        s_paths = set(paths)
        for port in o_ports + i_ports:
            assert type(port) is list
            #assert len(port)
            if not len(port):
                print("WARNING: empty port!")
                assert 0
            #print("Lattice.__init__", len(port))
            for p in port:
                assert isinstance(p, Path)
                s_paths.add(p)
        o_lookup = {} # Path --> o_idx
        i_lookup = {} # Path --> i_idx
        for odx,port in enumerate(o_ports):
            for p in port:
                o_lookup[p] = odx
        for idx,port in enumerate(i_ports):
            for p in port:
                i_lookup[p] = idx
        for l,r in pairs: # l < r
            assert isinstance(l, Path)
            assert isinstance(r, Path)
            assert l in s_paths
            assert r in s_paths
            assert l != r
            s_paths.add(l)
            s_paths.add(r)
        for l,r in list(pairs):
            for s,t in list(pairs):
                if s==r and (l,t) in pairs:
                    pairs.remove((l,t))
        assert s_paths
        paths = list(s_paths)
        self.shape = (no, ni)
        self.paths = paths
        self.o_ports = list(o_ports)
        self.i_ports = list(i_ports)
        self.o_lookup = o_lookup
        self.i_lookup = i_lookup
        self.pairs = list(pairs)
        self.s_paths = s_paths
        #print("Lattice.__init__", 
        #    [len(port) for port in o_ports], 
        #    [len(port) for port in i_ports], 
        #)
    @property
    def sup(self):
        paths = set(self.s_paths)
        for (a,b) in self.pairs:
            if a in paths:
                paths.remove(a)
        return paths
    @property
    def inf(self):
        paths = set(self.s_paths)
        for (a,b) in self.pairs:
            if b in paths:
                paths.remove(b)
        return paths
    def __str__(self):
        o_ports = self.o_ports
        i_ports = self.i_ports
        #return "Lattice(%s <-- %s)"%(len(o_ports), len(i_ports))
        return "Lattice(%s <-- %s)"%(
            [len(port) for port in o_ports], 
            [len(port) for port in i_ports],)
    def dump(L):
        print("="*79)
        print("Lattice.dump", L)
        print("paths:", end =' ')
        letters = ascii_lowercase + ascii_uppercase
        paths = L.paths
        names = {paths[i]:letters[i] for i in range(len(paths))}
        print('\t' + ''.join(names[p] for p in paths))
        for (l,r) in L.pairs:
            print("\t%s < %s"%(names[l], names[r]))
        print("sup:", ''.join(names[p] for p in L.sup))
        print("inf:", ''.join(names[p] for p in L.inf))
        print("="*79)
    def __add__(self, other):
        i_ports = self.i_ports + other.i_ports
        o_ports = self.o_ports + other.o_ports
        pairs = self.pairs + other.pairs
        #pairs.append((self.top, other.bot))
        left = self.sup
        right = other.inf
        for p in left:
          for q in right:
            pairs.append((p,q))
        
        return Lattice(o_ports, i_ports, pairs, self.paths+other.paths)
    def __mul__(top, bot):
        o_ports = [[] for port in top.o_ports]
        i_ports = [[] for port in bot.i_ports]
        pairs = []
        n = len(top.i_ports)
        assert n == len(bot.o_ports)
        #print("__mul__", top, bot)
        #print("n =", n)
        compose = {}
        s_paths = set()
        for i in range(n):
            #print("\ti =", i)
            #print("\t", len(bot.o_ports[i]), len(top.i_ports[i]))
            for a in bot.o_ports[i]:
              for b in top.i_ports[i]:
                p = a>>b
                idx = bot.i_lookup.get(a)
                if idx is not None:
                    i_ports[idx].append(p)
                odx = top.o_lookup.get(b)
                if odx is not None:
                    o_ports[odx].append(p)
                compose[b,a] = p
                s_paths.add(p)
            #print("\t\tpaths:", len(s_paths))
        #print("compose:", len(compose))
        #print(len(top.pairs), len(bot.pairs))
        pairs = []
        for p in bot.paths:
            if bot.o_lookup.get(p) is not None:
                continue
            idx = bot.i_lookup.get(p)
            if idx is not None:
                i_ports[idx].append(p)
            s_paths.add(p)
        for p in top.paths:
            if top.i_lookup.get(p) is not None:
                continue
            odx = top.o_lookup.get(p)
            if odx is not None:
                o_ports[odx].append(p)
            s_paths.add(p)
        for (a,b) in top.pairs:
            for c in bot.paths:
                if (a,c) in compose and (b,c) in compose:
                    pairs.append( (compose[a,c], compose[b,c]) )
                elif (a,c) in compose and b in s_paths:
                    pairs.append( (compose[a,c], b) )
                elif (b,c) in compose and a in s_paths:
                    pairs.append( (a, compose[b,c]) )
        for (b,c) in bot.pairs:
            for a in top.paths:
                if (a,b) in compose and (a,c) in compose:
                    pairs.append( (compose[a,b], compose[a,c]) )
                elif (a,b) in compose and c in s_paths:
                    pairs.append( (compose[a,b], c) )
                elif (a,c) in compose and b in s_paths:
                    pairs.append( (b, compose[a,c]) )
        for (a,b) in top.pairs:
          for (c,d) in bot.pairs:
            if (a,c) in compose and (b,d) in compose:
                pairs.append( (compose[a,c], compose[b,d]) )
        #print("pairs:", len(pairs))
        return Lattice(o_ports, i_ports, pairs, s_paths)


twos = lambda items : [(items[i], items[i+1]) for i in range(len(items)-1)]


class Visitor(object):
    def __init__(self, cell, dia):
        self.cell = cell
        self.dia = dia
        self.lookup = {}

    def on_visit(self, dia, **kw):
        #print("Visitor.__call__", type(dia))
        lookup = self.lookup
        if isinstance(dia, diagram.Spider):
            o_paths = [p.backwards() for p in dia.trace["top"]]
            i_paths = [p.backwards() for p in dia.trace["bot"]]
            i_ports = []
            o_ports = []
            if o_paths and i_paths:
                uniq = {(i_p,o_p):(i_p>>o_p) for o_p in o_paths for i_p in i_paths}
                i_ports = [[uniq[i_p,o_p] for o_p in o_paths] for i_p in i_paths]
                o_ports = [[uniq[i_p,o_p] for i_p in i_paths] for o_p in o_paths]
                pairs = reduce(operator.add, [twos(port) for port in i_ports])
                pairs += reduce(operator.add, [twos(port) for port in o_ports])
            elif o_paths:
                o_ports = [[p] for p in o_paths]
                pairs = twos(o_paths)
            elif i_paths:
                i_ports = [[p] for p in i_paths]
                pairs = twos(i_paths)
            else:
                assert 0, "empty spider: %s?"%dia
            #print("Visitor.__call__:", len(o_ports), len(i_ports))
            M = Lattice(o_ports, i_ports, pairs)

        elif isinstance(dia, diagram.VDia):
            M = reduce(operator.mul, [lookup[child] for child in dia])
        elif isinstance(dia, diagram.HDia):
            M = reduce(operator.add, [lookup[child] for child in dia])
        else:
            assert 0, type(dia)
        lookup[dia] = M

    def process(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup
        if 0:
            bg = box.WeakFillBox([0.9*white])
            dia = box.OBox([bg, dia])
            dia.strict = True
        cvs = dia.render(refresh=False)
        dia.visit(self.on_visit)

        llx, lly = dia.llx, dia.lly
        urx, ury = dia.urx, dia.ury
        cvs.stroke(path.rect(dia.llx, dia.lly, dia.width, dia.height), [grey])

        #x0, y0 = dia.llx, dia.lly
        #x1, y1 = dia.urx, dia.ury

        M = lookup[dia]
        #M.dump()
        sup, inf = M.sup, M.inf

        if len(M.o_ports) == 0 or len(M.i_ports) == 0:
            return cvs

        o_port = M.o_ports[0]
        i_port = M.i_ports[0]

        for p0 in i_port:
            if p0 in inf:
                break

        #p0 = path.path([path.moveto(x0, y0), path.lineto(*p0.getat(0))]) >> p0
        x0, y0 = p0.getat(0)
        x1, y1 = p0.getat(1)
        left = mkpath([(x0,y0),(llx,lly),(llx,ury),(x1,y1)])

        p0 = left >> p0.backwards()
        print(cell.src, cell.tgt)
        
        cvs.stroke(p0, [red]+st_arrow)

        if 0:
            # debug lattice structure
            bg = cvs
            cvs = Canvas()
            bb = bg.get_bound_box()
            x = 0
            for p in M.paths:
                fg = Canvas(bg)
                if p in sup:
                    fg.stroke(p, [red]+st_arrow)
                elif p in inf:
                    fg.stroke(p, [blue]+st_arrow)
                else:
                    fg.stroke(p, [grey]+st_Thick+st_arrow)
                cvs.insert(x, 0, fg)
                x += bb.width+0.2

        return cvs


class Cell2(object):
    def __init__(self, tgt, src, name):
        assert isinstance(tgt, Cell1)
        assert isinstance(src, Cell1)
        self.tgt = tgt
        self.src = src
        self.name = name

    def __str__(self):
        return "(%s<==%s==%s)"%(self.tgt, self.name, self.src)
    def __eq__(self, other):
        assert isinstance(other, Cell2)
        return str(self) == str(other)

    def construct(self):
        dia = self.on_construct()
        visitor = Visitor(self, dia)
        cvs = visitor.process()
        return cvs

    def on_construct(self): # override me
        pass

    def __lshift__(self, other):
        tgt = self.tgt << other.tgt
        src = self.src << other.src
        lhs = self.cells if isinstance(self, HCell2) else [self]
        rhs = other.cells if isinstance(other, HCell2) else [other]
        name = "<<".join(cell.name for cell in lhs+rhs)
        return HCell2(tgt, src, name, lhs+rhs)

    def __mul__(self, other):
        assert self.src == other.tgt
        lhs = self.cells if isinstance(self, VCell2) else [self]
        rhs = other.cells if isinstance(other, VCell2) else [other]
        name = "*".join(cell.name for cell in lhs+rhs)
        return VCell2(self.tgt, other.src, name, lhs+rhs)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [self][idx]


class Compound2(Cell2):
    def __init__(self, tgt, src, name, cells):
        Cell2.__init__(self, tgt, src, name)
        for cell in cells:
            assert isinstance(cell, Cell2), cell
        self.cells = list(cells)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx]


class HCell2(Compound2):
    def on_construct(self):
        dias = [a.on_construct() for a in self.cells]
        dia = diagram.HDia(dias)
        dia.name = self.name
        dia.cell = self
        return dia

class VCell2(Compound2):
    def on_construct(self):
        dias = [a.on_construct() for a in self.cells]
        dia = diagram.VDia(dias)
        dia.name = self.name
        dia.cell = self
        return dia


class Spider(Cell2):
    def __init__(self, tgt, src, name="?", st_pip=st_black, pip_radius=0.07, **kw):
        Cell2.__init__(self, tgt, src, name)
        p = path.circle(0, 0, pip_radius)
        if st_pip is not None:
            self.pip = CVSBox(Canvas().fill(p, st_pip).stroke(p, st_black))
        else:
            self.pip = None
        self.kw = kw

    def on_construct(self):
        tgt, src = self.tgt, self.src
        top_attrs = [a.st for a in tgt]
        bot_attrs = [a.st for a in src]
        dia = diagram.Spider(len(src), len(tgt), 
            top_attrs=top_attrs, bot_attrs=bot_attrs, pip=self.pip, **self.kw)
        dia.name = self.name
        dia.cell = self
        return dia


class Swap(Cell2):
    def __init__(self, X, Y):
        #Cell2.__init__(self, tgt, src, name)
        self.cells = [X, Y]
    def on_construct(self):
        X, Y = self.cells
        dia = diagram.Relation(2, 2, topbot=[(1,0), (0,1)], bot_attrs=[X.st, Y.st])
        dia.name = self.name
        dia.cell = self
        return dia

class Space(Cell2):
    def __init__(self, width=1.):
        #Cell2.__init__(self, tgt, src, name)
        self.width = width
    def on_construct(self):
        dia = diagram.Spider(0, 0, min_width=self.width)
        dia.name = self.name
        dia.cell = self
        return dia


def save(name, value=None):
    if value is None:
        value = cvs
    if isinstance(value, Cell2):
        value = value.construct()
    if isinstance(value, Box):
        #bg = box.WeakFillBox([0.9*white])
        #value = box.OBox([bg, value])
        #value.strict = True
        value = value.render()
    print("save(%r)"%(name,))
    #print(value)
    value.writePDFfile(name+".pdf")
    #value.writeSVGfile(name+".svg")


def test():
    n = Cell0("n", [grey])
    
    X = Cell1(n, n, "X", st_black+st_thick)
    XX = X<<X
    XXX = X<<X<<X
    XXXX = X<<X<<X<<X
    I = Cell1(n, n, "I", None)
    
    mul = Spider(X, XX, "mul")
    nul = Spider(X, XX, st_pip=st_white)
    comul = Spider(XX, X, "comul")
    Comul = Spider(XX, X, weight=10.)
    unit = Spider(X, I)
    counit = Spider(I, X)
    swap = Swap(X, X)
    Xi = Spider(X, X, st_pip=None, weight=10.)
    
    space = Space(0.5)
    sspace = Space(1.0)

    #op = ((comul*mul*comul) << (counit*mul)) * (comul << X)

    #op = (comul << mul)

    gap = unit*counit
    bone = counit*unit

    cvs = Canvas()
    x = 0.
    for op in [
        mul * comul,
        comul * mul * comul,
        ((mul*comul) << Xi) * comul,
        #op << Xi,
        #mul * (mul << Xi) * op * comul,
        (mul << Xi) * (Xi << comul),
        #counit << counit << unit,
        counit * unit,
        #counit * mul,
        #comul * unit,
        #mul << (unit * counit),
        mul * (Xi << gap),
        #mul * (Xi << unit),
        mul * (mul << (unit * counit)),
        #op * (Xi << comul),
        #mul * (op << gap) * (Xi << comul),
    ]:
        fg = op.construct()
        cvs.insert(x, 0, fg)
        x += fg.get_bound_box().width + 0.3

    #bubble = counit * mul * comul * unit
    #op = bubble << bubble

    save("spider-test", cvs)



if __name__ == "__main__":
    test()

    print("OK\n\n\n\n")







