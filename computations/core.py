from util import unique, intersect, identity, remove, toposort
from itertools import chain
from functools import partial
from strategies import exhaust


class Computation(object):
    """ An interface for a Computation

    Computations have inputs and outputs
    """
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs

    def edges(self):
        """ A sequence of edges """
        inedges  = ((i, self) for i in self.inputs)
        outedges = ((self, o) for o in self.outputs)
        return chain(inedges, outedges)

    @property
    def variables(self):
        return tuple(unique(chain(self.inputs, self.outputs)))

    def __add__(self, other):
        return CompositeComputation(self, other)

    def __str__(self):
        ins  = "["+', '.join(map(str, self.inputs)) +"]"
        outs = "["+', '.join(map(str, self.outputs))+"]"
        return "%s -> %s -> %s"%(ins, str(self.__class__.__name__), outs)

    def __repr__(self):
        return str(self)

    def toposort(self):
        """ Order computations in an executable order """
        return [self]

    def _info(self):
        return type(self), self.inputs, self.outputs
    def __hash__(self):
        return hash(self._info())
    def __eq__(self, other):
        return type(self) == type(other) and self._info() == other._info()

    libs = []
    includes = []

class CompositeComputation(Computation):
    """ A computation composed of other computations """

    def __new__(cls, *computations):
        computations = tuple(unique(computations))
        computations = exhaust(rm_identity)(computations)
        computations = exhaust(flatten)(computations)
        if len(computations) == 1:
            return computations[0]
        else:
            obj = object.__new__(cls)
            obj.computations = tuple(computations)
            return obj
    def __init__(self, *args):
        pass

    def _input_outputs(self, canonicalize=identity):
        """ Find the inputs and outputs of the complete computation """
        allin = map(canonicalize, unique(chain(*[c.inputs
                                                for c in self.computations])))
        allout = map(canonicalize, unique(chain(*[c.outputs
                                                for c in self.computations])))

        inputs  = remove(allout.__contains__, allin)
        outputs = remove(allin.__contains__, allout)
        ident_inputs  = [i for c in self.computations if isinstance(c, Identity)
                           for i in c.inputs]
        ident_outputs = [o for c in self.computations if isinstance(c, Identity)
                           for o in c.outputs]
        return tuple(inputs + ident_inputs), tuple(outputs + ident_outputs)

    @property
    def inputs(self):
        return self._input_outputs()[0]

    @property
    def outputs(self):
        return self._input_outputs()[1]

    @property
    def variables(self):
        return tuple(unique(chain(*[c.variables for c in self.computations])))

    def __str__(self):
        return "[[" + "\n  ".join(map(str, self.toposort())) + "]]"

    def edges(self):
        return chain(*[c.edges() for c in self.computations])

    def dict_io(self):
        """ Return a dict of computations from inputs to outputs

        returns {A: {Bs}} such that A must occur before each of the Bs
        """
        return {A: set([B for B in self.computations
                          if intersect(A.outputs, B.inputs)])
                    for A in self.computations}

    def dict_oi(self):
        """ Return a dag of computations from outputs to inputs

        returns {A: {Bs}} such that A requires each of the Bs before it runs
        """
        return {A: set([B for B in self.computations
                          if intersect(A.inputs, B.outputs)])
                    for A in self.computations}

    def toposort(self):
        """ Order computations in an executable order """
        return toposort(self.dict_io())

    def _info(self):
        return type(self), frozenset(self.computations)

    @property
    def libs(self):
        return list(unique(sum([c.libs for c in self.computations], [])))
    @property
    def includes(self):
        return list(unique(sum([c.includes for c in self.computations], [])))


def rm_identity(computations):
    """ Remove or reduce one identity """
    for c in computations:
        if isinstance(c, Identity):
            others = remove(c.__eq__, computations)
            other_vars = reduce(set.union, [o.variables for o in others], set())
            vars = remove(other_vars.__contains__, c.outputs)
            if not vars:
                return others
            if tuple(vars) != c.outputs:
                newident = Identity(*vars)
                return (newident,) + tuple(others)
    return computations

def flatten(computations):
    for c in computations:
        if isinstance(c, CompositeComputation):
            return tuple(remove(c.__eq__, computations)) + tuple(c.computations)
    return computations


class Identity(Computation):
    """ An Identity computation """
    def __init__(self, *inputs):
        self.inputs = inputs

    outputs = property(lambda self: self.inputs)
