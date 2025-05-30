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
    @property
    def i(self):
        return Cell1(self, self, self.name+".i", None)

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

    def __mul__(self, other):
        return self.i * other

    def __lshift__(self, other):
        if isinstance(other, Cell2):
            return self.i << other
        if isinstance(other, Cell0):
            other = other.i
        assert isinstance(other, Cell1), other
        assert self.src == other.tgt
        lhs = self.cells if isinstance(self, HCell1) else [self]
        rhs = other.cells if isinstance(other, HCell1) else [other]
        cells = [cell for cell in lhs + rhs if cell.st is not None] # fix for identities
        return HCell1(cells) if cells else self
    __matmul__ = __lshift__

    def __len__(self):
        return int(self.st is not None)

    def __getitem__(self, idx):
        if self.st is not None:
            return [self][idx]
        return [][idx]

    @property
    def i(self):
        Ai = Spider(self, self, self.name+".i", st_pip=None, weight=10.)
        return Ai


class HCell1(Cell1):
    def __init__(self, cells):
        assert cells
        tgt = cells[0].tgt
        src = cells[-1].src
        name = "<<".join(cell.name for cell in cells)
        Cell1.__init__(self, tgt, src, name, None)
        self.cells = list(cells)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx]

    @property
    def i(self):
        cell = reduce(operator.lshift, [cell.i for cell in self.cells])
        return cell


class Lattice(object):
    def __init__(self, tgt, src, o_ports, i_ports, pairs, paths=[]):
        assert isinstance(tgt, Cell0)
        assert isinstance(src, Cell0)
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
                    del pairs[l,t]
        #assert s_paths
        paths = list(s_paths)
        self.shape = (no, ni)
        self.paths = paths
        self.o_ports = list(o_ports)
        self.i_ports = list(i_ports)
        self.o_lookup = o_lookup
        self.i_lookup = i_lookup
        self.pairs = dict(pairs)
        self.s_paths = s_paths
        #print("Lattice.__init__", 
        #    [len(port) for port in o_ports], 
        #    [len(port) for port in i_ports], 
        #)
        self.src = src
        self.tgt = tgt
        for key in self.pairs:
            assert isinstance(key, tuple), key
            assert len(key) == 2

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
        #pairs = self.pairs + other.pairs
        pairs = dict(self.pairs)
        pairs.update(other.pairs)
        left = self.sup
        right = other.inf

        assert self.src == other.tgt
        cell0 = self.src

        for p in left:
          for q in right:
            pairs[p,q] = cell0
        
        return Lattice(self.tgt, other.src, o_ports, i_ports, pairs, self.paths+other.paths)

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
        pairs = {}
        for (a,b) in top.pairs:
            cell0 = top.pairs[a,b]
            for c in bot.paths:
                if (a,c) in compose and (b,c) in compose:
                    pairs[ (compose[a,c], compose[b,c]) ] = cell0
                elif (a,c) in compose and b in s_paths:
                    pairs[ (compose[a,c], b) ] = cell0
                elif (b,c) in compose and a in s_paths:
                    pairs[ (a, compose[b,c]) ] = cell0
        for (b,c) in bot.pairs:
            cell0 = bot.pairs[b,c]
            for a in top.paths:
                if (a,b) in compose and (a,c) in compose:
                    pairs[ (compose[a,b], compose[a,c]) ] = cell0
                elif (a,b) in compose and c in s_paths:
                    pairs[ (compose[a,b], c) ] = cell0
                elif (a,c) in compose and b in s_paths:
                    pairs[ (b, compose[a,c]) ] = cell0
        for (a,b) in top.pairs:
          for (c,d) in bot.pairs:
            if (a,c) in compose and (b,d) in compose:
                cell0 = top.pairs[a,b]
                assert bot.pairs[c,d] == cell0, "umm...???"
                pairs[ (compose[a,c], compose[b,d]) ] = cell0
        return Lattice(top.tgt, top.src, o_ports, i_ports, pairs, s_paths)


twos = lambda items : [(items[i], items[i+1]) for i in range(len(items)-1)]


class MonoidalVisitor(object):
    def __init__(self, cell, dia):
        self.cell = cell
        self.dia = dia
        self.lookup = {}

    def on_visit(self, dia, **kw):
        #print("Visitor.__call__", type(dia))
        lookup = self.lookup
        cell = dia.cell # yes it's an unfortunate series of tubes
        assert isinstance(cell, Cell2)
        #print("on_visit", list(kw.keys()))
        M = cell.build_lattice(lookup, dia, **kw)
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
        dia.visit(self.on_visit, cvs=cvs)


class Visitor(object):
    def __init__(self, cell, dia):
        self.cell = cell
        self.dia = dia
        self.lookup = {}
        self.is_monoidal = True
        self.cell0 = None

    def on_visit(self, dia, **kw):
        #print("Visitor.__call__", type(dia))
        lookup = self.lookup
        cell2 = dia.cell # yes it's an unfortunate series of tubes
        assert isinstance(cell2, Cell2)
        #print("on_visit", list(kw.keys()))
        M = cell2.build_lattice(lookup, dia, **kw)
        lookup[dia] = M

        if not self.is_monoidal:
            return

        cell0 = self.cell0
        if cell0 is None:
            cell0 = cell2.src.src
        for cell1 in list(cell2.src)+list(cell2.tgt):
            if cell1.src != cell0:
                self.is_monoidal = False
            if cell1.tgt != cell0:
                self.is_monoidal = False
        self.cell0 = cell0

    def process(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup

        #if self.is_monoidal:
        #    assert self.cell0 is not None
        #    bg = box.WeakFillBox(self.cell0.st)
        #    dia = box.OBox([bg, dia])
        #    dia.strict = True

        cvs = dia.render(refresh=False)
        dia.visit(self.on_visit, cvs=cvs)

        #cvs.stroke(path.rect(dia.llx, dia.lly, dia.width, dia.height), [grey])

        M = lookup[dia]
        #M.dump()
        o_ports = set(reduce(operator.add, M.o_ports, []))
        i_ports = set(reduce(operator.add, M.i_ports, []))

        if self.is_monoidal:
            assert self.cell0 is not None
            print("is_monoidal")
            bg = Canvas()
            bg.fill(path.rect(dia.llx, dia.lly, dia.width, dia.height), self.cell0.st)

        elif o_ports and i_ports:
            print("bicategory.Visitor.process: FIX ME")
            if o_ports.intersection(i_ports):
                # FIX FIX FIX XXX
                bg = self.process_connected()
            else:
                # HACK THIS ARRGGGH
                assert M.src == M.tgt
                cell0 = M.src
                bg = Canvas()
                bg.fill(path.rect(dia.llx, dia.lly, dia.width, dia.height), cell0.st)

        elif len(M.o_ports):
            bg = self.process_o_ports()
        elif len(M.i_ports):
            bg = self.process_i_ports()
        else:
            bg = self.process_bubble()

        cvs = Canvas([bg, cvs])
        cvs.dia = dia # more tubes
        
        #cvs.stroke(p0, [red]+st_arrow)
        #cvs.stroke(p1, [blue]+st_arrow)

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

    def find_paths(self, s_paths):
        M = self.lookup[self.dia]
        paths = list(s_paths)
        assert len(paths)
        p0 = p1 = paths[0]
        follow = []
        while 1:
            nf = len(follow)
            for pair in M.pairs:
                if pair[0]==p1 and pair[1] in s_paths:
                    p1 = pair[1]
                    follow.append(pair)
                if pair[1]==p0 and pair[0] in s_paths:
                    p0 = pair[0]
                    follow.append(pair)
            if nf == len(follow):
                break
        assert len(follow) or p0==p1
        return p0, p1, follow

    def process_o_ports(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup
        M = lookup[dia]

        s_paths = reduce(operator.add, M.o_ports)
        p0, p1, follow = self.find_paths(s_paths)

        bg = Canvas()

        #bg.stroke(p0, [red]+st_arrow)
        #bg.stroke(p1, [red]+st_arrow)

        assert M.src == M.tgt
        cell0 = M.src
        bg.fill(
            (mkpath([dia.ll, dia.ul, p0.getat(1)])>>p0.backwards()>>p1
             >>mkpath([p1.getat(1), dia.ur, dia.lr, dia.ll])), cell0.st)

        for (p,q) in follow:
            cell0 = M.pairs[p,q]
            #bg.fill(p >> q.backwards(), cell0.st)
            #bg.stroke(p, cell0.st+st_THICK)
            bg.fill(p >> mkpath([p.getat(1), q.getat(1)]) >> q.backwards(), cell0.st)

        #print(len(follow), p0==p1)
        #print(follow)

        return bg

    def process_i_ports(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup
        M = lookup[dia]

        s_paths = reduce(operator.add, M.i_ports)
        p0, p1, follow = self.find_paths(s_paths)

        bg = Canvas()

        #bg.stroke(p0, [red]+st_arrow)
        #bg.stroke(p1, [red]+st_arrow)

        assert M.src == M.tgt
        cell0 = M.src
        bg.fill(
            (p0.backwards()
             >> mkpath([p0.getat(0), dia.ll, dia.ul, dia.ur, dia.lr, p1.getat(0)])
             >> p1),
            cell0.st)

        for (p,q) in follow:
            cell0 = M.pairs[p,q]
            bg.fill(p.backwards() >> mkpath([p.getat(0), q.getat(0)]) >> q, cell0.st)

        return bg

    def process_bubble(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup
        M = lookup[dia]

        bg = Canvas()

        #bg.stroke(p0, [red]+st_arrow)
        #bg.stroke(p1, [red]+st_arrow)

        assert M.src == M.tgt
        cell0 = M.src
        bg.fill(path.rect(dia.llx, dia.lly, dia.width, dia.height), cell0.st)

        if not M.s_paths:
            return bg # <-------- return

        p0, p1, follow = self.find_paths(M.s_paths)
        for (p,q) in follow:
            cell0 = M.pairs[p,q]
            bg.fill(p.backwards() >> q, cell0.st)

        return bg

    def process_connected(self):
        cell = self.cell
        dia = self.dia
        lookup = self.lookup
        M = lookup[dia]
        #M.dump()

        #s_paths = set(reduce(operator.add, M.o_ports)).intersection(reduce(operator.add, M.i_ports))
        s_paths = set(reduce(operator.add, M.o_ports)).union(reduce(operator.add, M.i_ports))
        p0, p1, follow = self.find_paths(s_paths)

        tgt = cell.src.tgt
        src = cell.src.src
        bg = Canvas()

        llx, lly = dia.llx, dia.lly
        urx, ury = dia.urx, dia.ury

        # left fill
        x0, y0 = p0.getat(0)
        x1, y1 = p0.getat(1)
        left = mkpath([(x0,y0),(llx,lly),(llx,ury),(x1,y1)]) >> p0.backwards()
        bg.fill(left, tgt.st)

        for (p,q) in follow:
            cell = M.pairs[p,q]
            #p = pair[0] >> pair[1].backwards()
            p = p >> mkpath([p.getat(1), q.getat(1)]) >> q.backwards() >> mkpath([q.getat(0), p.getat(0)])
            bg.fill(p, cell.st)

        # right fill
        x0, y0 = p1.getat(0)
        x1, y1 = p1.getat(1)
        right = p1 >> mkpath([(x1,y1),(urx,ury),(urx,lly),(x0,y0)])
        bg.fill(right, src.st)
        return bg


class Cell2(object):
    def __init__(self, tgt, src, name):
        assert isinstance(tgt, Cell1)
        assert isinstance(src, Cell1)
        assert tgt.src == src.src, "%s != %s"%(tgt.src, src.src)
        assert tgt.tgt == src.tgt, "%s != %s"%(tgt.tgt, src.tgt)
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
        if isinstance(other, Cell0):
            other = other.i
        if isinstance(other, Cell1):
            other = other.i
        assert isinstance(other, Cell2), other
        tgt = self.tgt << other.tgt
        src = self.src << other.src
        lhs = self.cells if isinstance(self, HCell2) else [self]
        rhs = other.cells if isinstance(other, HCell2) else [other]
        name = "<<".join(cell.name for cell in lhs+rhs)
        return HCell2(tgt, src, name, lhs+rhs)
    __matmul__ = __lshift__

    def __mul__(self, other):
        if isinstance(other, Cell0):
            other = other.i
        if isinstance(other, Cell1):
            other = other.i
        assert isinstance(other, Cell2)
        assert self.src == other.tgt, "(%s) * (%s)"%(self.src, other.tgt)
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

    def build_lattice(self, lookup, dia, cvs):
        M = reduce(operator.add, [lookup[child] for child in dia])
        return M


class VCell2(Compound2):
    def on_construct(self):
        dias = [a.on_construct() for a in self.cells]
        dia = diagram.VDia(dias)
        dia.name = self.name
        dia.cell = self
        return dia

    def build_lattice(self, lookup, dia, cvs):
        M = reduce(operator.mul, [lookup[child] for child in dia])
        return M


class Border(Cell2):
    def __init__(self, child, st=st_thick, margin=0.1):
        if isinstance(child, Cell0):
            child = child.i
        if isinstance(child, Cell1):
            child = child.i
        assert isinstance(child, Cell2)
        name = "[%s]"%(child.name,)
        Cell2.__init__(self, child.tgt, child.src, name)
        self.child = child
        self.st = st
        self.r = margin

    def on_construct(self):
        child = self.child
        dia = child.on_construct()
        #item = box.MarginBox(dia, 0.1)
        dia.cell = self
        return dia

    def build_lattice(self, lookup, dia, cvs):
        x, y = dia.ll
        r = self.r
        cvs.stroke(path.rect(x+r, y+r, dia.width-2*r, dia.height-2*r), self.st)
        M = self.child.build_lattice(lookup, dia, cvs)
        return M


class Spider(Cell2):
    def __init__(self, tgt, src, name="?", st_pip=st_black, pip_radius=0.07, pip_cvs=None, **kw):
        Cell2.__init__(self, tgt, src, name)
        p = path.circle(0, 0, pip_radius)
        if pip_cvs is not None:
            self.pip = CVSBox(pip_cvs)
        elif st_pip is not None:
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

    def build_lattice(cell, lookup, dia, cvs):
        o_legs = [p.backwards() for p in dia.trace["top"]]
        i_legs = [p.backwards() for p in dia.trace["bot"]]
        i_ports = []
        o_ports = []
        pairs = {}
        if o_legs and i_legs:
            uniq = {(i_p,o_p):(i_p>>o_p) for o_p in o_legs for i_p in i_legs}
            i_ports = [[uniq[i_p,o_p] for o_p in o_legs] for i_p in i_legs]
            o_ports = [[uniq[i_p,o_p] for i_p in i_legs] for o_p in o_legs]
            #pairs = reduce(operator.add, [twos(port) for port in i_ports])
            #pairs += reduce(operator.add, [twos(port) for port in o_ports])
            for idx in range(len(i_legs)):
              for odx in range(len(o_legs)):
                if idx+1 < len(i_legs):
                    cell0 = cell.src[idx].src
                    pairs[uniq[i_legs[idx],o_legs[odx]], uniq[i_legs[idx+1],o_legs[odx]]] = cell0
                if odx+1 < len(o_legs):
                    cell0 = cell.tgt[odx].src
                    pairs[uniq[i_legs[idx],o_legs[odx]], uniq[i_legs[idx],o_legs[odx+1]]] = cell0
        elif o_legs:
            o_ports = [[p] for p in o_legs]
            for odx in range(len(o_legs)-1):
                cell0 = cell.tgt[odx].src
                pairs[o_legs[odx], o_legs[odx+1]] = cell0
        elif i_legs:
            i_ports = [[p] for p in i_legs]
            for idx in range(len(i_legs)-1):
                cell0 = cell.src[idx].src
                pairs[i_legs[idx], i_legs[idx+1]] = cell0
        else:
            pairs = {} # ?
        #print("Visitor.__call__:", len(o_ports), len(i_ports))
        M = Lattice(cell.src.tgt, cell.src.src, o_ports, i_ports, pairs)
        return M


#class Swap(Cell2):
#    def __init__(self, X, Y):
#        Cell2.__init__(self, tgt, src, name)
#        self.cells = [X, Y]
#    def on_construct(self):
#        X, Y = self.cells
#        dia = diagram.Relation(2, 2, topbot=[(1,0), (0,1)], bot_attrs=[X.st, Y.st])
#        dia.name = self.name
#        dia.cell = self
#        return dia

#class Space(Cell2):
#    def __init__(self, width=1.):
#        Cell2.__init__(self, tgt, src, name)
#        self.width = width
#    def on_construct(self):
#        dia = diagram.Spider(0, 0, min_width=self.width)
#        dia.name = self.name
#        dia.cell = self
#        return dia


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


def test_bubbles():
    n = Cell0("n", [blue+0.5*white])
    m = Cell0("m", [red+0.5*white])
    
    i = Cell1(n, n, "i", None)
    j = Cell1(m, m, "j", None)

    A  = Cell1(n, n, "A", st_black+st_thick)
    C = Cell1(n, m, "C", st_black+st_thick)
    B  = Cell1(m, m, "B", st_black+st_thick)
    D = Cell1(m, n, "D", st_black+st_thick)
    
    A_ = Spider(A, i)
    _A = Spider(i, A)
    A_AA = Spider(A, A << A, "A_AA")
    #wA_AA = Spider(A, A << A, st_pip=st_white)
    AA_A = Spider(A << A, A, "AA_A")


    B_ = Spider(B, j)
    BB_ = Spider(B<<B, j)
    _B = Spider(j, B)
    B_BB = Spider(B, B << B, "B_BB")
    BB_B = Spider(B << B, B, "BB_B")

    CD_A = Spider(C<<D, A, "CD_A")
    A_CD = Spider(A, C<<D, "A_CD")
    DC_B = Spider(D<<C, B, "DC_B")
    AC_C = Spider(A<<C, C, "AC_C")

    CD_ = Spider(C<<D, i, "CD_")
    _CD = Spider(i, C<<D, "CD_")
    DC_ = Spider(D<<C, j, "DC_")

    #Ai = Spider(A, A, st_pip=None, weight=10.)
    #Bi = Spider(B, B, st_pip=None, weight=10.)
    
    #space = Space(0.5)
    #sspace = Space(1.0)

    b = lambda x : Border(x, st_thick+[red])

    cvs = Canvas()
    x = 0.
    for op in [
        CD_A*A_CD*b(CD_A*A_),
        _CD,
        _CD * CD_,
        _A * b(A_CD * CD_A) * A_,
        #(D.i << C.i) * DC_B,
        #(A_AA << C.i) * (A.i << AC_C)*AC_C,
        #A_AA << AA_A,
        #A_AA * AA_A,
        #(AC_C << C.i), #*AC_C,
#        AA_A * A_AA * AA_A,
#        ((A_AA*AA_A) << A.i) * AA_A,
#        (A_AA << A.i) * (A.i << AA_A),
#        #_A * A_, # TODO
#        A_AA * (A.i << (A_ * _A)),
#        A_AA * (A_AA << (A_ * _A)),
    ]:
        fg = op.construct()
        bb = fg.get_bound_box()
        cvs.insert(x-bb.llx, -bb.lly, fg)
        x += bb.width + 0.3

    save("bubbles-test", cvs)


def test():


    lightgrey = 0.9*white
    n = Cell0("n", [lightgrey])
    I = Cell1(n, n, "I", None)
    
    #A = Cell1(n, n, "A", st_black+st_thick+st_arrow)
    #B = Cell1(n, n, "B", st_black+st_thick+st_arrow)
    
    X = Cell1(n, n, "X", st_black+st_thick)
    
    XX = X<<X
    XXX = X<<X<<X
    XXXX = X<<X<<X<<X
    
    assert I << X == X, I<<X
    
    up = Canvas().stroke(path.line(0,-0.1,0,0.1), st_arrow)
    dn = Canvas().stroke(path.line(0,0.1,0,-0.1), st_arrow)
    Xup = Spider(X, X, pip_cvs=up)
    Xdn = Spider(X, X, pip_cvs=dn)
    cap = Spider(I, XX, st_pip=None)
    cup = Spider(XX, I, st_pip=None)
    
    X_XX = mul = Spider(X, XX)
    XX_X = comul = Spider(XX, X)
    #Comul = Spider(XX, X, weight=10.)
    X_ = unit = Spider(X, I)
    _X = counit = Spider(I, X)


    cvs = Canvas()
    x = 0.
    for op in [
        (cap<<Xup)*(Xup<<cup),
        (cap) * (X << cap << X) * (cup << cup),
        mul * (unit << X),
    ]:
        fg = op.construct()
        bb = fg.get_bound_box()
        cvs.insert(x-bb.llx, -bb.lly, fg)
        x += bb.width + 0.3



    save("spider-test", cvs)

if __name__ == "__main__":
    test()

    print("OK\n\n\n\n")







