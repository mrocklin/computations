from computations.core import Computation, CompositeComputation, Identity
from computations.matrices.core import MatrixCall

Computation._as_tuple = lambda self: (type(self), self.inputs, self.outputs)
Identity._as_tuple = lambda self: (type(self), self.inputs)
CompositeComputation._as_tuple = lambda self: (type(self),) + self.computations
MatrixCall._as_tuple = lambda self: (type(self), self.args, self.typecode)

Computation._from_tuple = staticmethod(lambda (t, ins, outs): t(ins, outs))
Identity._from_tuple = staticmethod(lambda (t, ins): t(ins))
CompositeComputation.from_tuple = staticmethod(lambda tup: tup[0](*tup[1:]))
MatrixCall._from_tuple = staticmethod(lambda (t, args, tc): t(*(args+(tc,))))

from logpy import fact
from logpy.assoccomm import commutative, associative

fact(commutative, CompositeComputation)
fact(associative, CompositeComputation)
