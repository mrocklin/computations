from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall
from computations.core import Computation, CompositeComputation
import itertools as it

n, m = map(Symbol, 'nm')
A = MatrixSymbol('A', n, m)

class MatrixIntegerSymbol(MatrixSymbol):
    def fortran_type(self):
        return 'integer'

def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i)
new_status.i = 0

def new_request():
    new_request.i += 1
    return Symbol('request_%d' % new_request.i, integer=True)
new_request.i = 0
def new_ierr():
    new_ierr.i += 1
    return Symbol('ierr_%d' % new_ierr.i, integer=True)
new_ierr.i = 0
tags = it.count(1)
new_tag = tags.next
def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i,
                               Symbol('MPI_STATUS_SIZE', integer=True), 1)
new_status.i = 0


mpi_type = {'integer': 'MPI_INTEGER',
            'real(kind=8)': 'MPI_DOUBLE_PRECISION',
            'complex(kind=8)': 'MPI_COMPLEX'}

class MPI(Computation):
    libs = ['mpi']
    includes = ['mpif']

class Send(MPI):
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

class iSend(Send):
    """ MPI Synchronous Send Operation """
    def __init__(self, data, dest, tag=None, request=None, ierr=None):
        super(iSend, self).__init__(data, dest, tag=tag, ierr=ierr)
        self.request = request or new_request()
        self.outputs = (self.request,) + self.outputs

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        data, = input_names
        request, ierr = output_names
        numel = self.data.rows * self.data.cols
        dtype = mpi_type[dtype_of(self.data)]
        dest  = self.dest
        tag   = self.tag
        d = locals()
        return ['call MPI_iSEND( %(data)s, %(numel)s, %(dtype)s, %(dest)s, '
                '%(tag)s, MPI_COMM_WORLD, %(request)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_iSEND Failed'"%d]

class Recv(MPI):
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


from sympy import MatAdd, Basic
class Inaccessible(MatAdd):
    def __new__(cls, arg):
        return Basic.__new__(cls, arg)


class iRecv(Recv):
    def __init__(self, data, source, tag=None, request=None, ierr=None):
        self.data = data
        self.source = source
        self.tag = tag or new_tag()
        self.request = request or new_request()
        self.ierr = ierr or new_ierr()

        self.inputs = ()
        self.outputs = (Inaccessible(data), self.request, self.ierr)


    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        request, = input_names
        data, status, ierr = output_names
        numel  = self.data.rows * self.data.cols
        dtype  = mpi_type[dtype_of(self.data)]
        source = self.source
        tag    = self.tag
        d = locals()
        return ['call MPI_RECV( %(data)s, %(numel)s, %(dtype)s, %(source)s, '
                '%(tag)s, MPI_COMM_WORLD, %(status)s, %(request)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_iRECV Failed'"%d]


class iRecvWait(MPI):
    def __init__(self, data, request, status=None, ierr=None):
        self.request = request
        self.data = data
        self.ierr = ierr or new_ierr()
        self.status = status or new_status()

        self.inputs = (Inaccessible(data), self.request,)
        self.outputs = (data, self.status, self.ierr)

    inplace = {0: 0}

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s on %s"]' % (
                str(self), str(self.__class__.__name__), str(self.data))

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        _, request, = input_names
        _, status, ierr = output_names
        d = locals()
        return ['call MPI_WAIT( %(request)s, %(status)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'"%d]

class iSendWait(MPI):
    def __init__(self, request, status=None, ierr=None):
        self.request = request
        self.status = status or new_status()
        self.ierr = ierr or new_ierr()

        self.inputs = (self.request,)
        self.outputs = (self.status, self.ierr)

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s on %s"]' % (
                str(self), str(self.__class__.__name__), str(self.request))

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        request, = input_names
        status, ierr = output_names
        d = locals()
        return ['call MPI_WAIT( %(request)s, %(status)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'"%d]


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
    if not sharedvars:
        raise ValueError('No Shared Variables')
    sends = [Send(v, to_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    return CompositeComputation(*sends)

def isend(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    sends = [iSend(v, to_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    waits = [iSendWait(s.request) for s in sends]
    return CompositeComputation(*(sends + waits))

def recv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    recvs = [Recv(v, from_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    return CompositeComputation(*recvs)

def irecv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    recvs = [iRecv(v, from_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    waits = [iRecvWait(r.data, r.request) for r in recvs]
    return CompositeComputation(*(recvs + waits))
