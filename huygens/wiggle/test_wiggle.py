#!/usr/bin/env python

#from huygens.wiggle import cell_experimental
#assert 0
import huygens
huygens.EXPERIMENTAL = True

from huygens.namespace import *
from huygens.wiggle import Cell0, Cell1, Cell2
from huygens.wiggle.strict import l_unit, mul, comul, pull_over, unit, braid_over, pull_over, pull_over_i, assoc

import warnings
warnings.filterwarnings('ignore')

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


#counter = 0
#def save(cell):
#    global counter

def save(cell, name):
    print("writePDFfile: %s.pdf"%name)
    cell = cell.layout() # ARGHHH
    cell.render_cvs(pos="northeast").writePDFfile(name)
    print()
    print("_"*79)


def test_strict():
    M = Cell0("M", fill=scheme[0])
    N = Cell0("N", fill=scheme[1])

    f = Cell1(N, M)
    m_mm = mul(M)
    n_nn = mul(N)
    m_ = unit(M)
    n_ = unit(N)

    f_mul = Cell2(f << m_mm, n_nn << (f@f), pip_color=blue)
    f_unit = Cell2(f << m_, n_, pip_color=blue)

    mid = f_mul << (unit(M).i @ M.i.i)
    bot = mul(N).i << (f_unit @ f.i) << (M.i.i)

    #bot = (f_unit @ f.i) << (M.i.i)

#    print("\n\ntranslate")
#    bot.translate()

    save(mid, "test_mid")
    save(bot, "test_bot")
    return

    print("mid*bot")
    cell = mid * bot

    print("\n\n")
    save(cell, "test_strict")



def test_compose_units():
    M = Cell0("M", fill=scheme[0])
    N = Cell0("N", fill=scheme[1])

    cell = (M @ unit(N)) << (M.i << M.i)
    save(cell, "test_0")

    cell = M.i
    cell = cell.insert_identity_src(1)

    unitor = Cell2(M.i, mul(M)<<(unit(M) @ M))
    cell = unitor << Cell1(M,M, stroke=blue, pip_color=blue).i
    save(cell, "test_unitor")

    #cell = unitor << mul(M)
    src = mul(M) << comul(M)
    cell = Cell2(M.i, src)
    #cell = cell.v_op()
    counitor = unitor.h_op()
    cell = unitor << counitor
    bot = cell.v_op()
    cell = cell * bot
    save(cell, "test_mul")

    cell = M @ unit(N) @ M
    cell = cell << comul(M)
    cell = cell.translate()
    print(cell.src)
    for path in cell.get_paths():
        print(path)
        #for cell in path:
        #    if isinstance(cell, Cell0) and cell.is_identity():
        #        print(repr(cell))
    assert len(list(cell.get_paths())) == 3
    save(cell, "test_1")

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

    #test_strict()
    #test_level()
    test_compose_units()

    print("OK\n")


