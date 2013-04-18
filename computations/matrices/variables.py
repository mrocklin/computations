
from sympy.matrices.expressions import MatrixSymbol
from sympy import Symbol

# Pattern variables
alpha = Symbol('_alpha')
beta  = Symbol('_beta')
n,m,k = map(Symbol, ['_n', '_m', '_k'])
A = MatrixSymbol('_A', n, m)
B = MatrixSymbol('_B', m, k)
C = MatrixSymbol('_C', n, k)
D = MatrixSymbol('_D', n, n)
X = MatrixSymbol('_X', n, m)
Y = MatrixSymbol('_Y', n, m)
Z = MatrixSymbol('_Z', n, n)
S = MatrixSymbol('_S', n, n)
x = MatrixSymbol('_x', n, 1)
a = MatrixSymbol('_a', m, 1)
b = MatrixSymbol('_b', k, 1)

