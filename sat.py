#!/usr/bin/env python3

"""
Nice interface to a linear programming solver.

See also:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
"""

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

    def __neg__(self):
        return Scale(self, -1.0)

    def __sub__(self, other):
        other = Expr.promote(other)
        return Add([self, -other])

    def __rmul__(self, r):
        assert isinstance(r, (int, float))
        return Scale(self, r)

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
        raise TypeError

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

    def visit(self, system):
        system.on_const(self.r)

    def evaluate(self, vs):
        return self.r


class Variable(Expr):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    __repr__ = __str__

    def get_leaves(self):
        yield self

    def visit(self, system):
        system.on_variable(self)

    def __hash__(self):
        return id(self)

    def evaluate(self, vs):
        return vs[self]


class Add(Expr):
    def __init__(self, items):
        for item in items:
            assert isinstance(item, Expr)
        self.items = list(items)

    def __str__(self):
        return '+'.join(str(item) for item in self.items)
    __repr__ = __str__

    def get_leaves(self):
        for item in self.items:
            for leaf in item.get_leaves():
                yield leaf

    def visit(self, system):
        for item in self.items:
            item.visit(system)

    def evaluate(self, vs):
        r = 0.
        for item in self.items:
            r += item.evaluate(vs)
        return r


class Scale(Expr):
    def __init__(self, item, r):
        assert isinstance(item, Expr)
        self.item = item
        self.r = float(r)

    def __str__(self):
        return "%s*(%s)"%(self.r, self.item)
    __repr__ = __str__

    def get_leaves(self):
        item = self.item
        for leaf in item.get_leaves():
            yield leaf

    def visit(self, system):
        system.on_scale(self.r)
        self.item.visit(system)
        system.on_scale(1./self.r) # too whacky ??
        # push / pop scale instead ?

    def evaluate(self, vs):
        r = self.item.evaluate(vs)
        return self.r * r


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

    def visit(self, system):
        system.on_relation(self)
        self.lhs.visit(system)
        system.on_scale(-1.0)
        self.rhs.visit(system)
        system.on_finish()

EPSILON = 1e-6

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



class System(object):
    DEBUG = False

    def __init__(self, items=[]):
        self.items = list(items)
        vs = []
        for item in items:
            vs += list(item.get_leaves())
        vs = list(set(vs))
        vs.sort(key = lambda v:v.name)
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
            self.debug("System:", item)
            item.visit(self)

    def debug(self, *args, **kw):
        if self.DEBUG:
            print(*args, **kw)

    def on_scale(self, r):
        self.r *= r

    def on_variable(self, v):
        idx = self.lookup[v]
        self.debug("System.on_variable", self.r, v)
        self.lhs[idx] += self.r 

    def on_const(self, r):
        self.debug("System.on_const", self.r, r)
        self.rhs -= self.r * r

    def on_relation(self, rel):
        self.debug("System.on_relation", self.r, rel)
        self.lhs = [0.]*self.n
        self.rhs = 0.
        self.op = rel.op
            
    def on_finish(self):
        self.debug("System.on_finish")
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

    def solve(self, check=True):
        leqs = self.leqs
        eqs = self.eqs
        n = self.n
        c = numpy.array([1.]*n) # XXX could use another Expr for this XXX
        A_ub = numpy.array([lhs for lhs,rhs in self.leqs])
        b_ub = numpy.array([rhs for lhs,rhs in self.leqs])
        A_eq = numpy.array([lhs for lhs,rhs in self.eqs])
        b_eq = numpy.array([rhs for lhs,rhs in self.eqs])
        result = linprog(
            c, A_ub, b_ub, A_eq, b_eq)
        self.debug(result)
        assert result.success, result
        vs = {}
        x = result.x
        for i, v in enumerate(x):
            vs[self.vs[i]] = v

        if check:
            for item in self.items:
                e = item.evaluate(vs)
                self.debug(item, "?", e)
                assert e

        return vs


def main():
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    items = [
        x+y >= 1.,
        x+z == 5,
        y >= 3.
    ]

    system = System(items)

    result = system.solve()
    print(result)


if __name__ == "__main__":

    main()
    


