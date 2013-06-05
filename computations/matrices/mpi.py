from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall
from computations import Computation

n, m = map(Symbol, 'nm')
A = MatrixSymbol('A', n, m)

class MatrixIntegerSymbol(MatrixSymbol):
    def fortran_type(self):
        return 'integer'

def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i)
new_status.i = 0

def new_ierr():
    new_ierr.i += 1
    return Symbol('ierr_%d' % new_ierr.i, integer=True)
new_ierr.i = 0
def new_tag():
    new_tag.i += 1
    return Symbol('tag_%d' % new_tag.i, integer=True)
new_tag.i = 0
def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i,
                               Symbol('MPI_STATUS_SIZE', integer=True), 1)
new_status.i = 0


mpi_type = {'integer': 'MPI_INTEGER',
            'real(kind=8)': 'MPI_DOUBLE_PRECISION',
            'complex(kind=8)': 'MPI_COMPLEX'}

class Send(Computation):
    """ MPI Synchronous Send Operation """
    def __init__(self, data, dest, tag=None, ierr=None):
        self.ierr = ierr or new_ierr()
        self.tag = tag or new_tag()
        self.dest = dest
        self.data = data

        self.inputs = (data,)
        self.outputs = (self.ierr,)

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s-->%s"]' % (
                str(self), str(self.__class__.__name__), str(self.dest))

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        data, = input_names
        ierr, = output_names
        numel = self.data.rows * self.data.cols
        dtype = mpi_type[dtype_of(self.data)]
        dest  = self.dest
        tag   = self.tag
        d = locals()
        return ['call MPI_SEND( %(data)s, %(numel)s, %(dtype)s, %(dest)s, '
                '%(tag)s, MPI_COMM_WORLD, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_SEND Failed'"%d]


class Recv(Computation):
    """ MPI Synchronous Recv Operation """
    def __init__(self, data, source, tag=None, status=None, ierr=None):
        self.ierr = ierr or new_ierr()
        self.status = status or new_status()
        self.tag = tag or new_tag()
        self.source = source
        self.data = data

        self.inputs = ()
        self.outputs = (data, self.status, self.ierr)

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s<--%s"]' % (
                str(self), str(self.__class__.__name__), str(self.source))


    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        data, status, ierr = output_names
        numel  = self.data.rows * self.data.cols
        dtype  = mpi_type[dtype_of(self.data)]
        source = self.source
        tag    = self.tag
        d = locals()
        return ['call MPI_RECV( %(data)s, %(numel)s, %(dtype)s, %(source)s, '
                '%(tag)s, MPI_COMM_WORLD, %(status)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_RECV Failed'"%d]


tagdb = dict()
def gettag(a, b, expr):
    """ MPI Tag associated to transfer of expr from machine a to machine b """
    if (a, b, expr) not in tagdb:
        tagdb[(a, b, expr)] = gettag._tag
        gettag._tag += 1
    return tagdb[(a, b, expr)]
gettag._tag = 1


def send(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Send(v, to_machine, tag=tag)

def recv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Recv(v, from_machine, tag=tag)
