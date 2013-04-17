
from computations.core import (Computation, unique, CompositeComputation,
        Identity)

a,b,c,d,e,f,g,h = 'abcdefgh'

def test_Computation():
    C = Computation((a, b, c), (d, e, f))
    assert C.inputs == (a, b, c)
    assert C.outputs == (d, e, f)
    assert set(C.variables) == set((a, b, c, d, e, f))


A = Computation((a, b, c), (d,))
B = Computation((d, e), (f,))
C = CompositeComputation(A, B)

def test_composite():
    assert tuple(C.inputs)  == (a, b, c, e)
    assert tuple(C.outputs) == (f,)
    assert set(C.edges()) == set(((a, A), (b, A), (c, A), (A, d),
                                 (d, B), (e, B), (B, f)))
    assert set(C.variables) == set((a, b, c, d, e, f))

def test_composite_dict():
    assert C.dict_io() == {A: set([B]), B: set()}
    assert C.dict_oi() == {B: set([A]), A: set()}

def test_toposort():
    assert tuple(C.toposort()) == (A, B)

def test_multi_out():
    X = Computation((a, b), (d, e))
    Y = Computation((d,), (f,))
    Z = Computation((a, f), (g, h))
    C = CompositeComputation(X, Y, Z)
    print set(C.inputs)
    print set(C.outputs)
    assert set(C.inputs) == set((a, b))
    assert set(C.outputs) == set((e, g, h))
    assert tuple(C.toposort()) == (X, Y, Z)

def test_add():
    assert A + B == CompositeComputation(A, B)


def test_hash():
    assert hash(A)
    assert hash(C)

def test_canonicalize():
    assert CompositeComputation(A) == A
    assert A + B == B + A

    I = Identity(b)
    assert I + A == A

def test_rm_identity():
    A = Computation((d,), (f,))
    I2 = Identity(f, e)
    I3 = Identity(e)
    assert I2 + A == I3 + A

def test_composite_with_identity_inputs():
    A =  Computation((d,), (f,))
    I = Identity(c)
    C = A+I
    assert set(C.inputs) == set((d, c))
