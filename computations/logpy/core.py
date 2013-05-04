from computations.core import Computation, CompositeComputation, Identity

Computation._as_tuple = lambda self: (type(self), self.inputs, self.outputs)
Identity._as_tuple = lambda self: (type(self), self.inputs)
CompositeComputation._as_tuple = lambda self: (type(self),) + self.computations

Computation._from_tuple = staticmethod(lambda (t, ins, outs): t(ins, outs))
Identity._from_tuple = staticmethod(lambda (t, ins): t(ins))
CompositeComputation.from_tuple = staticmethod(lambda tup: tup[0](*tup[1:]))

from logpy import fact
from logpy.assoccomm import commutative, associative

fact(commutative, CompositeComputation)
fact(associative, CompositeComputation)
