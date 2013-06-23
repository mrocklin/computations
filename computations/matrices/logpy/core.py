from computations.matrices.core import MatrixCall
import computations.logpy

MatrixCall._as_logpy = lambda self: (type(self), self.args, self.typecode)
MatrixCall._from_logpy = staticmethod(lambda (t, args, tc): t(*(args+(tc,))))

from computations.matrices.blocks import Join, Slice
from logpy import logify

for cls in [Join, Slice]:
    logify(cls)
