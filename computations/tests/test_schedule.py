from computations.schedule import data_dependence, schedule
from computations.core import Computation, CompositeComputation

a,b,c,d,e,f,g,h = 'abcdefgh'
A = Computation((a, b, c), (d,))
B = Computation((d, e), (f,))
C = CompositeComputation(A, B)

def test_data_dependence():
    a, b = C.toposort()
    assert data_dependence(a, b) < 0
