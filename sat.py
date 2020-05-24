#!/usr/bin/env python3



class Value(object):

    @classmethod
    def promote(self, item):
        if isinstance(item, Value):
            return item
        assert isinstance(item, (int, float))
        item = Const(item)
        return item

    def __add__(self, other):
        other = Value.promote(other)
        return Add([self, other])

    def __neg__(self):
        return Scale(self, -1.0)

    def __sub__(self, other):
        other = Value.promote(other)
        return Add([self, -other])

    def __rmul__(self, r):
        assert isinstance(r, (int, float))
        return Scale(self, r)

    def __le__(self, other):
        other = Value.promote(other)
        return Le(self, other)

    def __ge__(self, other):
        other = Value.promote(other)
        return Le(other, self)

    def __eq__(self, other):
        other = Value.promote(other)
        return Eq(self, other)

    def __lt__(self, other):
        assert isinstance(other, Value)
        raise TypeError

    def __gt__(self, other):
        assert isinstance(other, Value)
        raise TypeError

    def __ne__(self, other):
        assert isinstance(other, Value)
        raise TypeError


class Const(Value):
    def __init__(self, r):
        self.r = r

    def __str__(self):
        return str(self.r)
    __repr__ = __str__


class Variable(Value):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    __repr__ = __str__


class Add(Value):
    def __init__(self, items):
        for item in items:
            assert isinstance(item, Value)
        self.items = list(items)

    def __str__(self):
        return '+'.join(str(item) for item in self.items)
    __repr__ = __str__


class Scale(Value):
    def __init__(self, item, r):
        assert isinstance(item, Value)
        self.item = item
        self.r = float(r)

    def __str__(self):
        return "%s*%s"%(self.r, self.item)
    __repr__ = __str__



class Bool(object):
    pass


class Le(Bool):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, Value)
        assert isinstance(rhs, Value)
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "%s <= %s" % (self.lhs, self.rhs)


class Eq(Bool):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, Value)
        assert isinstance(rhs, Value)
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "%s == %s" % (self.lhs, self.rhs)



class System(object):
    def __init__(self, items=[]):
        self.items = list(items)


def main():
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    print(x+y <= 2)
    print(x+2*y == z)


if __name__ == "__main__":

    main()
    


