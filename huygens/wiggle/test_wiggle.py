#!/usr/bin/env python

from huygens.namespace import *
from huygens.wiggle import Cell0, Cell1, Cell2


def test():

    m, n = Cell0("m"), Cell0("n")
    A = Cell1(m, n)
    B = Cell1(m, n)
    f = Cell2(B, A)

    mn = m @ n
    assert mn.level == 0

    mA = m @ A
    assert mA.level == 1
    assert mA[0].level == 1
    assert mA[1].level == 1

    Am = A @ m
    assert Am.level == 1
    assert Am[0].level == 1
    assert Am[1].level == 1

    


if __name__ == "__main__":

    test()

    print("OK\n")


