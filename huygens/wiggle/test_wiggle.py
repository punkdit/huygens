#!/usr/bin/env python

from huygens.namespace import *
from huygens.wiggle import Cell0, Cell1, Cell2
from huygens.wiggle.strict import l_unit, mul, pull_over, unit, braid_over, pull_over, pull_over_i, assoc

pink = color.rgba(1.0, 0.37, 0.36, 0.5)
cream = color.rgba(1.0, 1.0, 0.92, 0.5)
cyan = color.rgba(0.0, 0.81, 0.80, 0.5)
yellow = color.rgba(1.0, 0.80, 0.3, 0.5)

grey = color.rgba(0.75, 0.75, 0.75, 0.5)
red = color.rgba(1., 0.2, 0.2, 0.5)
green = color.rgba(0.3, 0.7, 0.2, 0.5)
blue = color.rgba(0.2, 0.2, 0.7, 0.5)

scheme = [grey, blue, yellow, red, pink] # colour scheme


def dump(cell):
    cell = cell.translate()
    print(cell)
    for path in cell.get_paths():
        print("\t", ' '.join(str(c) for c in path))
    

def test_compose():
    M = Cell0("M", fill=scheme[0])
    N = Cell0("N", fill=scheme[1])

    mul2 = lambda M,N : (mul(M)@mul(N)) << (M.i @ braid_over(N,M) @ N.i)
    src = mul2(M,N) << (mul2(M,N) @ (M.i@N.i))
    tgt = mul2(M,N) << ((M.i@N.i) @ mul2(M,N))
    top = (mul(M).i @ mul(N).i) << (M.i.i @ pull_over_i(N, mul(M)) @ mul(N).i) << (M.i.i@N.i.i@M.i.i@braid_over(N,M).i@N.i.i)
    mid = ((mul(M)<<(M.i@mul(M))).i @ assoc(N)) << (M.i@M.i@braid_over(N,M)@N.i@N.i).i << (M.i@braid_over(N,M)@braid_over(N,M)@N.i).i
    bot = (assoc(M)@mul(N).i) << (M.i.i@M.i.i @ pull_over(mul(N), M) @ N.i.i) << (M.i.i@braid_over(N,M).i@N.i.i@M.i.i@N.i.i)
    
    # these don't compose because we need to insert identity 1-cells
    cell = top * mid
    cell = mid * bot
    cell.render_cvs()


def test_compose_units():
    M = Cell0("M", fill=scheme[0])
    N = Cell0("N", fill=scheme[1])

    # below fails to compose because of identity 0-cells

    #cell = M @ M @ N
    #dump(cell.i)
    #return

    cell = M @ unit(N)
    print(cell.tgt, "<------", cell.src, [str(c) for c in cell.src])


    cell = (M @ unit(N)) << M
    dump(cell)
    return

    cell = (unit(N) @ M) << M
    dump(cell)
    cell = (M @ unit(N) @ M) << (M @ M)
    dump(cell)
    return

    # ((M<---M)@((N<---N@N)<<((N<---ident)@(N<---N))))
    # (((M<---M)@(N<---N@N))<<((((M<---M)@(N<---ident))<<(M<---M))@(N<---N)))

    bot = (l_unit(M) @ mul(N).i) << (pull_over(unit(N), M) @ N.i.i)
    top = (M.i.i @ l_unit(N))

    lhs = top.src.translate()
    rhs = bot.tgt.translate()

    dump(lhs)
    dump(rhs)

    dump(rhs[1])
    dump(rhs[1][0])

    return

    # compose fail
    cvs = Canvas([top.render_cvs(pos="northeast"), Translate(6,0), bot.render_cvs(pos="northeast")])
    cvs.writePDFfile("compose.pdf")

    cell = top * bot
    cvs = cell.render_cvs()
    
    # compose fail
    #top.d_rev()
    #bot.d_rev()



def test_level():

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

    test_level()
    test_compose_units()

    print("OK\n")


