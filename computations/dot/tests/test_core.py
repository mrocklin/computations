from computations.dot.core import generate_dot
from computations.core import Computation
a,b,c,d,e,f,g,h = 'abcdefgh'

def test_dot():
    X = Computation((a, b), (d, e))
    Y = Computation((d,), (f,))
    Z = Computation((a, f), (g, h))
    C = X + Y + Z

    assert isinstance(generate_dot(X), str)
    assert isinstance(generate_dot(C), str)
