from computations.core import Computation, CompositeComputation, Identity

Computation._as_logpy = lambda self: (type(self), self.inputs, self.outputs)
Identity._as_logpy = lambda self: (type(self), self.inputs)
CompositeComputation._as_logpy = lambda self: (type(self), self.computations)

Computation._from_logpy = staticmethod(lambda (t, ins, outs): t(ins, outs))
Identity._from_logpy = staticmethod(lambda (t, ins): t(ins))
CompositeComputation._from_logpy = staticmethod(lambda (op, args): op(*args))

from logpy import fact
from logpy.assoccomm import commutative, associative

fact(commutative, CompositeComputation)
fact(associative, CompositeComputation)
