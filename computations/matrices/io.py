from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall

n, m = map(Symbol, 'nm')
A = MatrixSymbol('A', n, m)

class Send(MatrixCall):
    _inputs = (A,)
    _outputs = ()
    condition = True
    inputs = property(lambda self: (self.args[0],))
    target = property(lambda self: self.args[1])
    tag    = property(lambda self: self.args[2])

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s-->%s"]' % (
                str(self), str(self.__class__.__name__), str(self.target))

class Recv(MatrixCall):
    _inputs = ()
    _outputs = (A,)
    condition = True
    inputs = ()
    outputs = property(lambda self: (self.args[0],))
    source = property(lambda self: self.args[1])
    tag    = property(lambda self: self.args[2])

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s<--%s"]' % (
                str(self), str(self.__class__.__name__), str(self.source))

tagdb = dict()
def gettag(a, b, expr):
    """ MPI Tag associated to transfer of expr from machine a to machine b """
    if (a, b, expr) not in tagdb:
        tagdb[(a, b, expr)] = gettag._tag
        gettag._tag += 1
    return tagdb[(a, b, expr)]
gettag._tag = 0

def send(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Send(v, to_machine, tag)

def recv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Recv(v, from_machine, tag)
