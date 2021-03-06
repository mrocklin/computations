from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall
from computations.core import Computation, CompositeComputation
from computations.util import memoize
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
tags = it.count(1000)
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

    def pseudocode_call(self, input_names, output_names):
        data, = input_names
        dest  = self.dest
        return ['Send %(data)s to %(dest)s' % locals()]


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
        return ['call MPI_ISend( %(data)s, %(numel)s, %(dtype)s, %(dest)s, '
                '%(tag)s, MPI_COMM_WORLD, %(request)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_ISend Failed'"%d]

    def pseudocode_call(self, input_names, output_names):
        data, = input_names
        dest  = self.dest
        return ['Send %(data)s to %(dest)s asynchronously' % locals()]


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

    def pseudocode_call(self, input_names, output_names):
        data, status, ierr = output_names
        source = self.source
        return ['Receive %(data)s from %(source)s' % locals()]

from sympy import MatAdd, Basic
class Inaccessible(MatAdd):
    def __new__(cls, arg):
        return Basic.__new__(cls, arg)
    def _sympystr(self, printer):
        return "Inaccessible(%s)"%printer._print(self.args[0])


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
        data, request, ierr = output_names
        numel  = self.data.rows * self.data.cols
        dtype  = mpi_type[dtype_of(self.data)]
        source = self.source
        tag    = self.tag
        d = locals()
        return ['call MPI_IRecv( %(data)s, %(numel)s, %(dtype)s, %(source)s, '
                '%(tag)s, MPI_COMM_WORLD, %(request)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_IRecv Failed'"%d]

    def pseudocode_call(self, input_names, output_names):
        data, status, ierr = output_names
        source = self.source
        return ['Receive %(data)s from %(source)s asynchronously' % locals()]

class Wait(MPI):
    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s on %s"]' % (
                str(self), str(self.__class__.__name__), str(self.request))


class iRecvWait(Wait):
    def __init__(self, data, request, status=None, ierr=None):
        self.request = request
        self.data = data
        self.ierr = ierr or new_ierr()
        self.status = status or new_status()

        self.inputs = (Inaccessible(data), self.request,)
        self.outputs = (data, self.status, self.ierr)

    inplace = {0: 0}

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        _, request, = input_names
        _, status, ierr = output_names
        d = locals()
        return ['call MPI_WAIT( %(request)s, %(status)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'"%d]


    def pseudocode_call(self, input_names, output_names):
        data, status, ierr = output_names
        request = self.request
        return ['Wait on transfer of %(data)s from %(request)s to complete' % locals()]


class iSendWait(Wait):
    def __init__(self, request, status=None, ierr=None):
        self.request = request
        self.status = status or new_status()
        self.ierr = ierr or new_ierr()

        self.inputs = (self.request,)
        self.outputs = (self.status, self.ierr)

    def fortran_call(self, input_names, output_names):
        from computations.matrices.fortran.core import dtype_of
        request, = input_names
        status, ierr = output_names
        d = locals()
        return ['call MPI_WAIT( %(request)s, %(status)s, %(ierr)s)'%d,
                "if (%(ierr)s .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'"%d]


    def pseudocode_call(self, input_names, output_names):
        request, = input_names
        request = self.request
        return ['Wait on %(request)s to complete' % locals()]


tagdb = dict()
def gettag(a, b, expr):
    """ MPI Tag associated to transfer of expr from machine a to machine b """
    if (a, b, expr) not in tagdb:
        tagdb[(a, b, expr)] = gettag._tag
        gettag._tag += 1
    return tagdb[(a, b, expr)]
gettag._tag = 1000


def maybe_expr(v):
    from computations.inplace import ExprToken
    return v.expr if isinstance(v, ExprToken) else v

# TODO: make send, recv, etc... return TokenComputations if they receive them
#       Use variable tokens from inputs.  How to get new tokens? need tokenier?

from computations.inplace import tokenize, ExprToken, TokenComputation

def isend_expr(v, from_machine, to_machine):
    send = iSend(v, to_machine, tag=gettag(from_machine, to_machine, v))
    wait = iSendWait(send.request)
    return CompositeComputation(send, wait)

def isend_exprtoken(et, from_machine, to_machine, tokenizer):
    tok = lambda expr: et.token if expr==et.expr else tokenizer(expr)
    return tokenize(isend_expr(et.expr, from_machine, to_machine), tokenizer=tok)

def irecv_expr(v, from_machine, to_machine):
    recv = iRecv(v, from_machine, tag=gettag(from_machine, to_machine, v))
    wait = iRecvWait(recv.data, recv.request)
    return CompositeComputation(recv, wait)

def irecv_exprtoken(et, from_machine, to_machine, tokenizer):
    tok = lambda expr: et.token if expr==et.expr else tokenizer(expr)
    return tokenize(irecv_expr(et.expr, from_machine, to_machine), tokenizer=tok)

@memoize
def send(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    sends = [Send(v, to_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    return CompositeComputation(*sends)

@memoize
def isend(from_machine, to_machine, from_job, to_job, tokenizer=None):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    if isinstance(from_job, TokenComputation):
        return CompositeComputation(*[
            isend_exprtoken(v, from_machine, to_machine, tokenizer)
                                         for v in sharedvars])
    else:
        return CompositeComputation(*[isend_expr(v, from_machine, to_machine)
                                         for v in sharedvars])


@memoize
def recv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    recvs = [Recv(v, from_machine, tag=gettag(from_machine, to_machine, v))
                    for v in sharedvars]
    return CompositeComputation(*recvs)

@memoize
def irecv(from_machine, to_machine, from_job, to_job, tokenizer=None):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    if not sharedvars:
        raise ValueError('No Shared Variables')
    if isinstance(from_job, TokenComputation):
        return CompositeComputation(*[
            irecv_exprtoken(v, from_machine, to_machine, tokenizer)
                                         for v in sharedvars])
    else:
        return CompositeComputation(*[irecv_expr(v, from_machine, to_machine)
                                         for v in sharedvars])


def mpi_key(c):
    if isinstance(c, Wait):         return +1
    elif isinstance(c, MPI):        return -1
    else:                           return  0

def mpi_tag_key(c):
    if isinstance(c, MPI) and hasattr(c, 'tag'):            return c.tag
    else:                                                   return 0

from computations.schedule import key_to_cmp
mpi_cmps = map(key_to_cmp, (mpi_key, mpi_tag_key))
