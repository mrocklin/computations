from sympy import MatrixSymbol, Symbol, Q, assuming
from computations.inplace import inplace_compile
from sympy.matrices.expressions.hadamard import HadamardProduct
from computations.matrices.elemental import ElemProd

n = Symbol('n')
x = MatrixSymbol('X', n, 1)
y = MatrixSymbol('Y', n, 1)
c = ElemProd(x,y)

def test_ElemProd():
    assert HadamardProduct(x,y) in c.outputs
    assert x in c.inputs
    assert y in c.inputs

def test_ElemProd_code():
    from computations.matrices.fortran.core import generate
    ic = inplace_compile(c)
    with assuming(Q.real_elements(x), Q.real_elements(y)):
        s = generate(ic, [x,y], [HadamardProduct(x,y)])
    with open('elem_test.f90','w') as f:
        f.write(s)
    assert "= X * Y" in s

