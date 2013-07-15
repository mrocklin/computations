from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy import Symbol, Q

n, k    = 1000, 500 # Symbol('n'), Symbol('k')
mu      = MatrixSymbol('mu', n, 1)
Sigma   = MatrixSymbol('Sigma', n, n)
H       = MatrixSymbol('H', k, n)
R       = MatrixSymbol('R', k, k)
data    = MatrixSymbol('data', k, 1)

newmu   = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma= Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

newSigma =  -1.0*Sigma*H.T*(H*Transpose(H*Sigma) + R).I*H*Sigma + Sigma
newmu =  Sigma*H.T*(H*Transpose(H*Sigma) + R).I*(-1.0*data + H*mu) + mu



assumptions = (Q.positive_definite(Sigma), Q.symmetric(Sigma),
               Q.positive_definite(R), Q.symmetric(R), Q.fullrank(H))

inputs = mu, Sigma, H, R, data
outputs = newmu, newSigma
