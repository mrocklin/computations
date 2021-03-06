from computations.core import Computation
import computations.logpy.core

from logpy import run, eq, membero, variables
from logpy.assoccomm import eq_assoccomm as eqac

def test_computation():
    c = Computation((1, 2), (3,))
    d = Computation((1, 0), (3,))

    assert run(0, 0, eq(c, d)) == ()

    with variables(0):
        assert run(0, 0, eq(c, d)) == (2,)

def test_composite_computation():
    c = Computation((1, 2), (3,)) + Computation((3, 4), (5, 6))
    d = Computation((1, 2), (0,)) + Computation((0, 4), (5, 6))
    e = Computation((0, 4), (5, 6)) + Computation((1, 2), (0,))

    assert run(0, 0, eq(c, d)) == ()

    with variables(0):
        assert run(0, 0, eq(c, d)) == (3,)

def test_composite_commutativity():
    c = Computation((1, 2), (3,)) + Computation((3, 4), (5, 6))
    e = Computation((0, 4), (5, 6)) + Computation((1, 2), (0,))

    with variables(0):
        assert run(0, 0, eqac(c, e)) == (3,)

def test_reification_of_computations():
    d = Computation((1, 2), (0,)) + Computation((0, 4), (5, 6))

    with variables(0):
        assert set(run(0, d, membero(0, (10, 11)))) == set((
                Computation((1, 2), (10,)) + Computation((10, 4), (5, 6)),
                Computation((1, 2), (11,)) + Computation((11, 4), (5, 6))))
