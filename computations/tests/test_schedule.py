from computations.schedule import data_dependence, schedule, key_to_cmp
from computations.core import Computation, CompositeComputation

a,b,c,d,e,f,g,h = 'abcdefgh'
A = Computation((a, b, c), (d,))
B = Computation((d, e), (f,))
C = CompositeComputation(A, B)
D = Computation((g,), (h,))

def test_data_dependence():
    a, b = C.toposort()
    assert data_dependence(a, b) < 0

def test_schedule():
    assert schedule([B, A]) == [A, B]
    assert schedule([B, A, D], key_to_cmp(str)) == [A, B, D]

def test_key_to_cmp():
    cmp = key_to_cmp(str)
    assert cmp('a', 'b') == -1
    assert cmp('b', 'b') ==  0
    assert cmp('b', 'a') == +1
