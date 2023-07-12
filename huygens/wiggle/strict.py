#!/usr/bin/env python3

"""
previous version: monoid.py

"""

cache = lambda f : f

from huygens.namespace import st_dotted
from huygens.wiggle import DCell0, Cell0, Cell1, Cell2


ident = Cell0.ident

#m = Cell0("m", stroke=black, fill=grey, st_stroke=st_thin)
#n = Cell0("n", stroke=black, fill=blue, st_stroke=st_thin)
#
#m_m = Cell1(m, m, pip_color=None, stroke=None)
#m_m_m = m_m << m_m
#m_mm = Cell1(m, m@m)
#mm_m = Cell1(m@m, m)
#_m = Cell1(ident, m)
#m_ = Cell1(m, ident)
#
#n_n = Cell1(n, n, pip_color=None, stroke=None)
#n_n_n = n_n << n_n
#n_nn = Cell1(n, n@n)
#nn_n = Cell1(n@n, n)
#_n = Cell1(ident, n)
#n_ = Cell1(n, ident)

def swap(m, n):
    return Cell1(n@m, m@n, pip_color=None, st_stroke=st_dotted)

#m_swap = swap(m, m)
#n_swap = swap(n, n)


def on_constrain(cell, system):
    pip_x, pip_y = cell.pip_x, cell.pip_y
    system.add(pip_x == (1/2)*(cell.tgt[0].pip_x + cell.src[0].pip_x))
    system.add(pip_y == (1/2)*(cell.tgt[0].pip_y + cell.src[0].pip_y))


def vertical_rigid(cell, system):
    pip_x, pip_y = cell.pip_x, cell.pip_y
    #system.add(pip_x == (1/2)*(cell.tgt.pip_x + cell.src.pip_x))
    system.add(pip_y == (1/2)*(cell.tgt.pip_y + cell.src.pip_y))
    system.add(pip_x == cell.tgt.pip_x)
    system.add(pip_x == cell.src.pip_x)

#
#m_assoc = Cell2(
#    m_mm<<(m_m @ m_mm),
#    m_mm<<(m_mm @ m_m),
#    cone=0.7,
#    on_constrain=on_constrain)
#n_assoc = Cell2(
#    n_nn<<(n_n @ n_nn),
#    n_nn<<(n_nn @ n_n),
#    cone=0.7,
#    on_constrain=on_constrain)

def on_constrain(cell, system):
    pip_x, pip_y = cell.pip_x, cell.pip_y
    system.add(pip_x == (1/2)*(cell.tgt[1].pip_x + cell.src[1].pip_x))
    system.add(pip_y == (1/2)*(cell.tgt[1].pip_y + cell.src[1].pip_y))

#n_coassoc = Cell2(
#    (n_n @ nn_n) << nn_n,
#    (nn_n @ n_n) << nn_n,
#    cone=0.7,
#    on_constrain=on_constrain)

conv = lambda a,b : 0.5*(a+b)
    

# These on_constrain's don't work so well...
def on_constrain(cell, system):
    pip_x, pip_y = cell.pip_x, cell.pip_y
    system.add(pip_x == conv(cell.src[0].pip_x, cell.tgt.pip_x))
#m_lunit = Cell2(m_m, m_mm<<(m_ @ m_m), cone=0.7, on_constrain=on_constrain)
#n_lunit = Cell2(n_n, n_nn<<(n_ @ n_n), cone=0.7, on_constrain=on_constrain)

def on_constrain(cell, system):
    pip_x, pip_y = cell.pip_x, cell.pip_y
    #system.add(pip_x == cell.src[0].pip_x)
    system.add(pip_x == conv(cell.src[0].pip_x, cell.tgt.pip_x))
#m_runit = Cell2(m_m, m_mm<<(m_m @ m_), cone=0.7, on_constrain=on_constrain)
#n_runit = Cell2(n_n, n_nn<<(n_n @ n_), cone=0.7, on_constrain=on_constrain)

# ----------------------------------------------

def interchange_i(C, A):
    assert isinstance(C, Cell1)
    assert isinstance(A, Cell1)
    rhs = Cell2(A<<A.src.i, A.tgt.i<<A, pip_color=None)
    lhs = Cell2(C.tgt.i<<C, C<<C.src.i, pip_color=None)
    return lhs @ rhs

def interchange(C, A):
    cell = interchange_i(C, A)
    return cell.v_rev()


# ------------ braided stuff -------------------

from huygens.wiggle import st_braid

@cache
def braid_over(B, A, sym=False):
    assert isinstance(B, Cell0)
    assert isinstance(A, Cell0)
    if isinstance(B, DCell0):
        assert len(B) == 2
        B0, B1 = B
        cell = (braid_over(B0, A, sym)@B1.i) << (B0.i @ braid_over(B1, A, sym)) # recurse
        #assert 0
    else:
        st = st_braid if not sym else []
        A_ = A(st_stroke=st)
        B_ = B(st_stroke=st)
        cell = Cell1( A_@B, B@A_, pip_color=None, st_stroke=st_dotted)
    return cell

@cache
def braid_under(A, B, sym=False):
    st = st_braid if not sym else []
    A_ = A(st_stroke=st)
    B_ = B(st_stroke=st)
    cell = Cell1( B@A_, A_@B, pip_color=None, st_stroke=st_dotted)
    return cell

@cache
def b_pos(A, B):
    return Cell2(B.i @ A.i, braid_under(A,B) << braid_over(B,A), cone=0.5, pip_color=None)

@cache
def b_pos_i(A, B):
    return b_pos(A, B).v_rev()

@cache
def b_neg(A, B):
    return Cell2(A.i@B.i, braid_over(B,A) << braid_under(A,B), cone=0.5, pip_color=None)

@cache
def b_neg_i(A, B):
    return b_neg(A, B).v_rev()

loop_over = lambda A,B : b_pos_i(B,A)
loop_over_i = lambda A,B : b_pos(B,A)
loop_under = b_neg_i
loop_under_i = b_neg
    

@cache
def syl_over(A, B):
    return Cell2(A.i @ B.i, braid_over(B,A) << braid_over(A,B), cone=0.5, pip_color=None)

@cache
def syl_over_i(A, B):
    cell = syl_over(A, B)
    return cell.v_rev()



#structure_pip = black
structure_pip = None # yes?

@cache
def pull_over(lhs, rhs, sym=False):
    #if isinstance(lhs, Cell2):
    #    lhs = lhs.src
    #if isinstance(rhs, Cell2):
    #    rhs = rhs.src
    if isinstance(lhs, Cell2) and isinstance(rhs, Cell0):
        f, A = lhs, rhs
        psrc = braid_over(f.src.tgt, A, sym) << (f.src @ A.i)
        if f.tgt.src is ident:
            ptgt = (A.i @ f.tgt) << A.i
        else:
            ptgt = (A.i @ f.tgt) << braid_over(f.tgt.src, A, sym)
        cell = Cell2(ptgt, psrc, pip_color=structure_pip)
    elif isinstance(lhs, Cell1) and isinstance(rhs, Cell0):
        f, A = lhs, rhs
        psrc = braid_over(f.tgt, A, sym) << (f @ A.i)
        if f.src is ident:
            ptgt = (A.i @ f) << A.i
        else:
            ptgt = (A.i @ f) << braid_over(f.src, A, sym)
        cell = Cell2(ptgt, psrc, pip_color=structure_pip)
    elif isinstance(lhs, Cell0):
        A, f = lhs, rhs
        cell = pull_over(f, A, sym) # <<< recurse
        cell = cell.d_rev()
    else:
        assert 0, "%s %s"%(type(lhs), type(rhs))
    return cell

@cache
def pull_under(lhs, rhs, sym=False):
    #if isinstance(lhs, Cell2):
    #    lhs = lhs.src
    #if isinstance(rhs, Cell2):
    #    rhs = rhs.src
    if isinstance(lhs, Cell2) and isinstance(rhs, Cell0):
        f, A = lhs, rhs
        psrc = braid_under(f.src.tgt, A, sym) << (f.src @ A.i)
        if f.tgt.src is ident:
            ptgt = (A.i @ f.tgt) << A.i
        else:
            ptgt = (A.i @ f.tgt) << braid_under(f.tgt.src, A, sym)
        cell = Cell2(ptgt, psrc, pip_color=structure_pip)
    elif isinstance(lhs, Cell1) and isinstance(rhs, Cell1):
        f, g = lhs, rhs
        psrc = braid_under(f.tgt, g.tgt, sym) << (f @ g)
        ptgt = (g @ f) << braid_under(f.src, g.src, sym)
        cell = Cell2(ptgt, psrc, pip_color=structure_pip)
    elif isinstance(lhs, Cell1) and isinstance(rhs, Cell0):
        f, A = lhs, rhs
        psrc = braid_under(f.tgt, A, sym) << (f @ A.i)
        if f.src is ident:
            ptgt = (A.i @ f) << A.i
        else:
            ptgt = (A.i @ f) << braid_under(f.src, A, sym)
        cell = Cell2(ptgt, psrc, pip_color=structure_pip)
    elif isinstance(lhs, Cell0):
        A, f = lhs, rhs
        cell = pull_under(f, A, sym) # <<< recurse
        cell = cell.d_rev()
    else:
        assert 0, "%s %s"%(type(lhs), type(rhs))
    return cell

def pull_over_i(lhs, rhs):
    cell = pull_over(lhs, rhs)
    cell = cell.v_rev()
    return cell

def pull_under_i(lhs, rhs):
    cell = pull_under(lhs, rhs)
    cell = cell.v_rev()
    return cell


@cache
def swap(A, B):
    cell = Cell1(B@A, A@B, pip_color=None, st_stroke=st_dotted)
    return cell

@cache
def swap_insert(A, B):
    src = A.i@B.i
    tgt = swap(B,A) << swap(A,B)
    cell = Cell2(tgt, src, pip_color=None)
    return cell
    
def swap_insert_i(A, B):
    cell = swap_insert(A, B)
    return cell.v_rev()


# ----------------------------------------------


@cache
def mul(A):
    cell = Cell1(A, A@A)
    return cell

def comul(A):
    return mul(A).h_rev()

def unit(A):
    return Cell1(A, ident)

def counit(A):
    return unit(A).h_rev()

def assoc(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == (1/2)*(cell.tgt[0].pip_x + cell.src[0].pip_x))
        system.add(pip_y == (1/2)*(cell.tgt[0].pip_y + cell.src[0].pip_y))
    src = mul(A) << (mul(A) @ A.i)
    tgt = mul(A) << (A.i @ mul(A))
    return Cell2(tgt, src, on_constrain=on_constrain)

def assoc_i(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == (1/2)*(cell.tgt[0].pip_x + cell.src[0].pip_x))
        system.add(pip_y == (1/2)*(cell.tgt[0].pip_y + cell.src[0].pip_y))
    src = mul(A) << (A.i @ mul(A))
    tgt = mul(A) << (mul(A) @ A.i)
    return Cell2(tgt, src, on_constrain=on_constrain)

def coassoc(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == (1/2)*(cell.tgt[1].pip_x + cell.src[1].pip_x))
        system.add(pip_y == (1/2)*(cell.tgt[1].pip_y + cell.src[1].pip_y))
    cell = assoc(A).h_rev()
    cell = cell(on_constrain=on_constrain)
    return cell

def l_unit(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == conv(cell.src[0].pip_x, cell.tgt.pip_x))
    src = mul(A) << (unit(A) @ A.i)
    tgt = A.i
    return Cell2(tgt, src, on_constrain=on_constrain)


def r_unit(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == conv(cell.src[0].pip_x, cell.tgt.pip_x))
    src = mul(A) << (A.i @ unit(A))
    tgt = A.i
    return Cell2(tgt, src, on_constrain=on_constrain)
    
l_counit = lambda A : l_unit(A).h_rev()
r_counit = lambda A : r_unit(A).h_rev()


def l_unit_i(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == conv(cell.tgt[0].pip_x, cell.src.pip_x))
    tgt = mul(A) << (unit(A) @ A.i)
    src = A.i
    return Cell2(tgt, src, on_constrain=on_constrain)


def r_unit_i(A):
    def on_constrain(cell, system):
        pip_x, pip_y = cell.pip_x, cell.pip_y
        system.add(pip_x == conv(cell.tgt[0].pip_x, cell.src.pip_x))
    tgt = mul(A) << (A.i @ unit(A))
    src = A.i
    return Cell2(tgt, src, on_constrain=on_constrain)
    
l_counit_i = lambda A : l_unit_i(A).h_rev()
r_counit_i = lambda A : r_unit_i(A).h_rev()

def comm_over(M, sym=False):
    op = swap if sym else braid_over
    return Cell2(mul(M), mul(M)<<op(M,M))
def comm_under(M, sym=False):
    op = swap if sym else braid_under
    return Cell2(mul(M), mul(M)<<op(M,M))

comm_over_i = lambda M, sym=False : comm_over(M, sym).v_rev()
comm_under_i = lambda M, sym=False : comm_under(M, sym).v_rev()

cocomm_over = lambda M, sym=False : comm_over(M, sym).h_rev()
cocomm_under = lambda M, sym=False : comm_under(M, sym).h_rev()
cocomm_over_i = lambda M, sym=False : cocomm_over(M, sym).v_rev()
cocomm_under_i = lambda M, sym=False : cocomm_under(M, sym).v_rev()

comm = lambda M : comm_over(M, True)
comm_i = lambda M : comm_over_i(M, True)
cocomm = lambda M : cocomm_over(M, True)
cocomm_i = lambda M : cocomm_over_i(M, True)

# ----------------------------------------------


def l_dist(f, M):
    src = f << mul(M)
    tgt = mul(M) << (f @ f)
    return Cell2(tgt, src, pip_color=f.pip_color)

def r_dist(M, f):
    src = comul(M) << f
    tgt = (f @ f) << comul(f)
    return Cell2(tgt, src, pip_color=f.pip_color)

l_dist_i = lambda f, M: l_dist(f, M).v_rev()
r_dist_i = lambda f, M: r_dist(f, M).v_rev()





