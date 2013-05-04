from computations.matrices.core import MatrixCall
import computations.logpy

MatrixCall._as_tuple = lambda self: (type(self), self.args, self.typecode)
MatrixCall._from_tuple = staticmethod(lambda (t, args, tc): t(*(args+(tc,))))
