#!/usr/bin/env python3

"""
Nice interface to a linear programming solver.

See also:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
"""

from functools import reduce
from operator import add

import numpy
from scipy.optimize import linprog



class Expr(object):

    @classmethod
    def promote(self, item):
        if isinstance(item, Expr):
            return item
        assert isinstance(item, (int, float))
        item = Const(item)
        return item

    def __add__(self, other):
        other = Expr.promote(other)
        return Add([self, other])
    __radd__ = __add__

    def __sub__(self, other):
        other = Expr.promote(other)
        return Add([self, -other])

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            pass
        elif isinstance(other, Expr):
            raise TypeError("non-linear operation not implemented (for good reasons.)")
        else:
            raise TypeError()
        if abs(other) < EPSILON:
            return 0.
        return Scale(self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__rmul__(other)
        if isinstance(other, Expr):
            raise TypeError("non-linear operation not implemented (for good reasons.)")
        else:
            raise TypeError()

    def __neg__(self):
        return self.__rmul__(-1)

    def __truediv__(self, r):
        return self.__rmul__(1./r)

    def __le__(self, other):
        other = Expr.promote(other)
        return Le(self, other)

    def __ge__(self, other):
        other = Expr.promote(other)
        return Le(other, self)

    def __eq__(self, other):
        other = Expr.promote(other)
        return Eq(self, other)

    def __lt__(self, other):
        assert isinstance(other, Expr)
        return str(self) < str(other)

    def __gt__(self, other):
        assert isinstance(other, Expr)
        raise TypeError

    def __ne__(self, other):
        assert isinstance(other, Expr)
        raise TypeError


class Const(Expr):
    def __init__(self, r):
        self.r = r

    def __str__(self):
        return str(self.r)
    __repr__ = __str__

    def get_leaves(self):
        return []

    def visit(self, solver):
        solver.on_const(self.r)

    def evaluate(self, vs):
        return self.r

    def __rmul__(self, val):
        if isinstance(val, Const):
            return Const(self.r * val.r)
        elif isinstance(val, (int, float)):
            return Const(self.r * val)
        return Expr.__rmul__(self, val)

    def get_affine(self):
        return Affine({}, self.r)

    def get_subs(self, subs):
        return self


class Variable(Expr):
    def __init__(self, name, weight=1.0, vmin=None, vmax=None):
        self.name = name # any object
        self.weight = weight # relative importance w.r.t. minimize
        self.vmin = vmin
        self.vmax = vmax

    def __str__(self):
        return str(self.name)
    __repr__ = __str__

    def get_leaves(self):
        yield self

    def visit(self, solver):
        solver.on_variable(self)

    def __hash__(self):
        return id(self)

    def evaluate(self, vs):
        return vs[self]

    def get_affine(self):
        return Affine({self:1.}, 0.)

    def get_subs(self, subs):
        return subs.get(self, self)


class Add(Expr):
    def __init__(self, _items):
        items = []
        for item in _items:
            assert isinstance(item, Expr)
            if isinstance(item, Add):
                items += item.items
            else:
                items.append(item)
        self.items = items

    def __str__(self):
        return '+'.join(str(item) for item in self.items)
    __repr__ = __str__

    def __rmul__(self, val):
        items = [item.__rmul__(val) for item in self.items] # distribute
        return Add(items)

    def get_leaves(self):
        for item in self.items:
            for leaf in item.get_leaves():
                yield leaf

    def visit(self, solver):
        for item in self.items:
            item.visit(solver)

    def evaluate(self, vs):
        r = 0.
        for item in self.items:
            r += item.evaluate(vs)
        return r

    def get_affine(self):
        lin, a = {}, 0.
        for item in self.items:
            affine = item.get_affine()
            a += affine.const
            for k,v in affine.lin.items():
                lin[k] = lin.get(k, 0.) + v
        for (k,v) in list(lin.items()):
            if abs(v)<1e-6:
                del lin[k]
        return Affine(lin, a)

    def get_subs(self, subs):
        items = [item.get_subs(subs) for item in self.items]
        return reduce(add, items, 0.)


class Scale(Expr):
    def __init__(self, item, r):
        assert isinstance(item, Expr)
        assert abs(r) > 1e-6
        self.item = item
        self.r = float(r)
        assert isinstance(item, Variable), self # XXX distribute

    def __str__(self):
        return "%s*(%s)"%(self.r, self.item)
    __repr__ = __str__

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return Scale(self.item, self.r*other)

    def get_leaves(self):
        item = self.item
        for leaf in item.get_leaves():
            yield leaf

    def visit(self, solver):
        solver.on_scale(self.r)
        self.item.visit(solver)
        solver.on_scale(1./self.r) # too whacky ??
        # push / pop scale instead ?

    def evaluate(self, vs):
        r = self.item.evaluate(vs)
        return self.r * r

    def get_affine(self):
        affine = self.item.get_affine()
        lin = dict((k,v*self.r) for (k,v) in affine.lin.items())
        const = affine.const * self.r
        return Affine(lin, const)

    def get_subs(self, subs):
        return self.r * self.item.get_subs(subs)


class Term(object):
    pass

class Relation(Term):

    def __init__(self, lhs, rhs):
        assert isinstance(lhs, Expr)
        assert isinstance(rhs, Expr)
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "%s %s %s" % (self.lhs, self.op, self.rhs)

    def get_leaves(self):
        for leaf in self.lhs.get_leaves():
            yield leaf
        for leaf in self.rhs.get_leaves():
            yield leaf

    def visit(self, solver):
        solver.on_relation(self)
        self.lhs.visit(solver)
        solver.on_scale(-1.0)
        self.rhs.visit(solver)
        solver.on_finish()

    def get_subs(self, subs):
        lhs = self.lhs.get_subs(subs)
        rhs = self.rhs.get_subs(subs)
        return self.__class__(lhs, rhs)



EPSILON = 1e-4 # ???

class Le(Relation):
    op = "<="

    def evaluate(self, vs):
        lhs = self.lhs.evaluate(vs)
        rhs = self.rhs.evaluate(vs)
        return lhs <= rhs + EPSILON


class Eq(Relation):
    op = "=="

    def evaluate(self, vs):
        lhs = self.lhs.evaluate(vs)
        rhs = self.rhs.evaluate(vs)
        return abs(lhs - rhs) < EPSILON

#    def is_ready(self):
#        lhs, rhs = self.lhs, self.rhs
#        if isinstance(lhs, Variable) and isinstance(rhs, Const):
#            return True
#        if isinstance(rhs, Variable) and isinstance(lhs, Const):
#            return True
#        return False

    def get_linear(self):
        lhs, rhs = self.lhs, self.rhs
        print("get_linear", self)
        val = lhs - rhs
        print("\t", val.get_affine())


class Solver(object):
    DEBUG = False

    def __init__(self, items=[]):
        assert len(items)
        self.items = list(items)
        vs = []
        for item in items:
            assert isinstance(item, Term)
            vs += list(item.get_leaves())
        vs = list(set(vs))
        vs.sort(key = lambda v:str(v.name))
        lookup = dict((v, idx) for (idx,v) in enumerate(vs))

        self.vs = vs
        self.n = len(vs)
        self.lookup = lookup

        self.r = 1.0
        self.leqs = []
        self.eqs = []

        self.op = None
        self.lhs = None
        self.rhs = None

        for item in items:
            self.debug("Solver:", item)
            item.visit(self)

    def debug(self, *args, **kw):
        if self.DEBUG:
            print(*args, **kw)

    def on_scale(self, r):
        self.r *= r

    def on_variable(self, v):
        idx = self.lookup[v]
        self.debug("Solver.on_variable", self.r, v)
        self.lhs[idx] += self.r 

    def on_const(self, r):
        self.debug("Solver.on_const", self.r, r)
        self.rhs -= self.r * r

    def on_relation(self, rel):
        self.debug("Solver.on_relation", self.r, rel)
        self.lhs = [0.]*self.n
        self.rhs = 0.
        self.op = rel.op
            
    def on_finish(self):
        self.debug("Solver.on_finish")
        if self.op == '<=':
            self.leqs.append((self.lhs, self.rhs))
        elif self.op == '==':
            self.eqs.append((self.lhs, self.rhs))
        else:
            assert 0, self.op
        self.r = 1.0
        self.op = None
        self.lhs = None
        self.rhs = None

    def solve(self, check=True, verbose=False):
        leqs = self.leqs
        eqs = self.eqs
        n = self.n
        assert n>0
        weights = [self.vs[i].weight for i in range(n)]
        bounds = [(self.vs[i].vmin, self.vs[i].vmax) for i in range(n)]
        c = numpy.array(weights)
        if len(leqs):
            A_ub = numpy.array([lhs for lhs,rhs in leqs])
            b_ub = numpy.array([rhs for lhs,rhs in leqs])
        else:
            A_ub = numpy.zeros((0, n))
            b_ub = numpy.zeros((0,))
        if len(eqs):
            A_eq = numpy.array([lhs for lhs,rhs in eqs])
            b_eq = numpy.array([rhs for lhs,rhs in eqs])
        else:
            A_eq = numpy.zeros((0, n))
            b_eq = numpy.zeros((0,))
        #print(c)
        #print(A_ub)
        result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="interior-point")
        self.debug(result)
        #assert result.success, result
        if not result.success:
            print("WARNING: optimize failed")
            print(result)
            print("--------------------------")
        vs = {}
        x = result.x
        for i, v in enumerate(x):
            vs[self.vs[i]] = v

        if check:
            for item in self.items:
                e = item.evaluate(vs) # was this constraint satisfied ?
                self.debug(item, "?", e)
                if not e:
                    print("WARNING: constraint failed")
                    print("%s: lhs=%s, rhs=%s" %(
                        item, item.lhs.evaluate(vs), item.rhs.evaluate(vs)))
                    print("--------------------------")

        if verbose:
            for item in self.items:
                e = item.evaluate(vs) # was this constraint satisfied ?
                print(e, item, item.lhs.evaluate(vs), item.rhs.evaluate(vs))

        return vs


class Listener(object):

    def __hash__(self):
        return id(self)

    def on_update(self, name, value):
        #print(self.__class__.__name__, "on_update", name, value)
        setattr(self, name, value)

    def on_refresh(self, name):
        #print(self.__class__.__name__, "on_refresh", name)
        if name in self.__dict__:
            delattr(self, name)



class System(object):

    def __init__(self):
        self.stems = {}
        self.items = []
        self.str_items = set()
        self.lookup = None
        #self.listeners = {}
        self.all_vars = set()

    def get_var(self, stem="v", weight=1.0, vmin=None, vmax=None):
        idx = self.stems.get(stem, -1) + 1
        v = Variable('%s_%d'%(stem, idx), weight, vmin=vmin, vmax=vmax)
        v.listeners = [] # hang this here...
        self.stems[stem] = idx
        self.all_vars.add(v)
        return v

    def listen_var(self, listener, name, *args, **kw):
        assert isinstance(listener, Listener)
        assert hash(name) is not None, "need hashable name"
        v = self.get_var(*args, **kw)
        #self.listeners[v] = (listener, name)
        v.listeners.append((listener, name, v))
        return v

    def listen_expr(self, listener, name, expr):
        assert isinstance(listener, Listener)
        assert isinstance(expr, Expr)
        assert hash(name) is not None, "need hashable name"
        for v in expr.get_leaves():
            v.listeners.append((listener, name, expr))
            assert v in self.all_vars # sanity check

    def add(self, item, weight=None):
        "add a constraint term"
        assert isinstance(item, Term), "%r expected Term" % item
        if weight is None:
            # strict
            self.items.append(item)
            s = str(item)
            #assert s not in self.str_items, "redundant constraint"
            self.str_items.add(s)
        elif isinstance(item, Eq):
            # allow equality violation at a cost == weight
            assert weight > 0., "??"
            c = self.get_var("eq_slack", weight)
            a, b = item.lhs, item.rhs
            self.add(c >= b-a) # recurse
            self.add(c >= a-b) # recurse
        elif isinstance(item, Le):
            # allow inequality violation at a cost == weight
            c = self.get_var("le_slack", weight, 0.)
            a, b = item.lhs, item.rhs
            self.add(c >= a-b) # recurse
        else:
            assert 0, "item %r not understood" % (item,)

    def minimize(self, expr, weight=1.0):
        assert isinstance(expr, Expr)
        v = self.get_var("minimize", weight)
        self.add(v == expr)

    def maximize(self, expr, weight=1.0):
        assert isinstance(expr, Expr)
        v = self.get_var("maximize", -weight)
        self.add(v == expr)

    def refresh(self):
        #print("System.refresh")
        for v in self.all_vars:
          for listener, name, expr in v.listeners:
            #value = self[expr]
            #listener.on_update(name, expr)
            listener.on_refresh(name)
        self.__init__()

    def solve(self, simplify=False, verbose=False):
        #print("System.solve")
        assert self.lookup is None, "already called solve!"
        if simplify:
            subs, leqs = gaussian_eliminate(self)
            #for k in subs.keys():
            #  for leq in leqs:
            #    for k1 in leq.get_leaves():
            #        assert k is not k1, "%s %s" % (k, leq)
            #print("items:", len(leqs))
            solver = Solver(leqs)
            lookup = solver.solve(verbose=verbose)
            #print(lookup)
            for (src,tgt) in subs.items():
                #print(src, tgt, tgt.evaluate(lookup))
                assert src.weight==0.
                if src in lookup:
                    print(src, tgt, lookup[src])
                assert src not in lookup
                lookup[src] = tgt.evaluate(lookup)
        else:
            items = self.items
            #print("items:", len(items))
            solver = Solver(items)
            lookup = solver.solve(verbose=verbose)
        self.lookup = lookup

        # notify the listeners
        for v in self.all_vars:
          for listener, name, expr in v.listeners:
            value = self[expr] # if KeyError: did this expr participate in the system ?
            listener.on_update(name, value)

    def __getitem__(self, v):
        if isinstance(v, (int, float)):
            return v
        assert isinstance(v, (Variable, Expr, Term)), repr(v)
        assert self.lookup is not None, "call solve first!"
        #return self.lookup[v]
        value = v.evaluate(self.lookup)
        value = float(value)
        return value


class Affine(object):
    " lin + const == 0 "
    def __init__(self, lin, const):
        self.lin = lin
        self.keys = lin.keys
        self.items = lin.items
        self.const = const

    def __getitem__(self, key):
        return self.lin.get(key, 0.)

    def __str__(self):
        return "Affine(%s, %s)"%(self.lin, self.const)

    def __rmul__(self, r):
        lin = dict((v, r*s) for (v,s) in self.lin.items())
        const = r*self.const
        return Affine(lin, const)

    def __isub__(self, other):
        lin = self.lin
        for (v,r) in other.lin.items():
            lin[v] = lin.get(v, 0) - r
            if abs(lin[v])<EPSILON:
                del lin[v]
        self.const -= other.const
        return self

    def get_subs(self, v):
        #self = (1./self[v])*self
        lin, const = self.lin, self.const
        m = self[v]
        expr = Const(-const/m)
        for _v,_m in lin.items():
            if _v is v:
                continue
            expr -= (_m/m)*_v
        return expr


def gaussian_eliminate(system, verbose=False):
    # here we do Gaussian elimination
    all_vars = system.all_vars

    if verbose:
        print("\n\ngaussian_eliminate", len(system.items), len(all_vars))

    vs = list(all_vars)
    vs.sort()
    lookup = dict((v, i) for (i,v) in enumerate(vs))
    subs = {}
    eqs = []
    for item in system.items:
        if not isinstance(item, Eq):
            continue
        eqs.append(item)

    # solve linear system: Ax = b
    n = len(eqs)
    idxs = list(range(n))
    lookup = dict((v,[]) for v in vs)
    affines = []
    for idx, eq in enumerate(eqs):
        val = eq.lhs - eq.rhs
        affine = val.get_affine()
        affine.idx = idx
        for v,r in affine.items():
            if abs(r)>EPSILON:
                lookup[v].append(affine)
        affines.append(affine)
        #print(eq)
        #print(affine, affine.idx)
        #print()
    #print()

    # seems to help, slightly
    vs.sort( key = lambda v : -len(lookup[v]))

    pivots = {} 
    def show():
      for idx in range(n):
        print("[", end="")
        for v in vs:
          x = affines[idx][v]
          s = str(x).rjust(3) if x else '.  '
          print("%4s"%s, end=" ")
          #print("*" if abs(x)>EPSILON else ".", end="")
        print("]", -affines[idx].const)
      print()

    if verbose:
        show()

    for v in vs:
        for affine in lookup[v]:
            if affine in pivots: # or abs(affine[v])<EPSILON:
                continue
            break
        else:
            continue
        m = affine[v]
        #print("pivot", v, affine)
        assert abs(m)>EPSILON, m
        #print("other:", [other.idx for other in lookup[v]])
        for other in list(lookup[v]):
            if other is affine:
                continue
            r = other[v]
            assert abs(r)>EPSILON
            if abs(r) < EPSILON:
                continue
            #print("other:", other.idx)
            for k in other.keys():
                lookup[k].remove(other)
            other -= (r/m)*affine # mutate Affine inplace
            for k in other.keys():
                lookup[k].append(other)
        assert affine not in pivots
        pivots[affine] = v
        assert lookup[v] == [affine]
        #show()
        #print()

    if verbose:
        show()

    subs = {}
    for affine, v in pivots.items():
        subs[v] = affine.get_subs(v)
        #print(v, v.weight, v.vmin, v.vmax)

    #for src, tgt in subs.items():
    #    if isinstance(tgt, Expr):
    #        for t in tgt.get_leaves():
    #            assert t not in subs

    leqs = []
    for item in system.items:
        if isinstance(item, Eq):
            continue
        item = item.get_subs(subs)
        if isinstance(item.lhs, Const) and isinstance(item.rhs, Const):
            continue
        leqs.append(item)
    return subs, leqs




def test_sat():

    from random import choice, randint, seed
    seed(0)
    
    for trial in range(10):
        system = System()
    
        N = 10
        vs = [system.get_var(weight=0) for i in range(N)]
        items = []
        for v in vs:
            items.append(v<=1000)
            items.append(-1000<=v)
        for i in range(N):
            lhs = 0
            for k in range(randint(1,N)):
                lhs += choice([-1,1,2])*choice(vs)
            rhs = randint(-2, 2)
            if randint(0,1):
                items.append(lhs == rhs)
            #else:
            #    items.append(lhs <= rhs)
        #    print(items[-1])
    
        for item in items:
            print(item)
            system.add(item)
    
        subs, leqs = gaussian_eliminate(system)
        for k,v in subs.items():
            print(k, "-->", v)
        #for leq in leqs:
        #    print(leq)

        if leqs:
            for item in leqs:
                print(item)
            solver = Solver(leqs)
            soln = solver.solve()
            print(soln)

        soln = Solver(items).solve()
        print("soln:", soln)

        break



if __name__ == "__main__":

    test_sat()
    print("OK\n")
    


