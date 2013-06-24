from computations.core import Computation, CompositeComputation, Identity
from term import termify
termify(Computation)
termify(Identity)
termify(CompositeComputation)

CompositeComputation._term_args = lambda self: self.computations
CompositeComputation._term_new = staticmethod(
        lambda args: CompositeComputation(*args))

from logpy import fact
from logpy.assoccomm import commutative, associative

fact(commutative, CompositeComputation)
fact(associative, CompositeComputation)
