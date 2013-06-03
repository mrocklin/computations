from itertools import chain
from sympy import Symbol
from computations.core import Computation

class Ticks(Symbol):
    def fortran_type(self):
        return 'integer'

class Time(Symbol):
    def fortran_type(self):
        return 'real*8'

class ClockVar(Ticks): pass

def new_time():
    new_time.i += 1
    return Time('time_%d' % new_time.i)
new_time.i = 0


Rate = ClockVar('clock_rate')
Max  = ClockVar('clock_max')

class Profile(Computation):
    """ A Computation to profile another computation """
    def __init__(self, comp, duration=None):
        self.comp = comp
        self.duration = duration or new_time()
        self.ticks = Ticks('start_time'), Ticks('end_time')

    inputs  = property(lambda self: self.comp.inputs)
    time_vars = property(lambda self: (self.duration, Rate, Max) + self.ticks)
    outputs = property(lambda self: self.time_vars + self.comp.outputs)

    def arguments(self, inputs, outputs):
        subinputs = inputs
        n = len(self.time_vars)
        time_vars, suboutputs = outputs[:n], outputs[:n]
        subarguments = self.comp.arguments(subinputs, suboutputs)

        return tuple(time_vars) + tuple(subarguments)

    def fortran_call(self, input_names, output_names):
        duration, rate, max, start, end = output_names[:5]
        comp_output_names = output_names[5:]
        d = locals()
        return (['call system_clock ( %(start)s, %(rate)s, %(max)s )' % d] +
                self.comp.fortran_call(input_names, comp_output_names) +
                ['call system_clock ( %(end)s,   %(rate)s, %(max)s )' % d,
                 '%(duration)s = real(%(end)s - %(start)s) / real(%(rate)s)' % d])

    @property
    def inplace(self):
        return dict((k+len(self.time_vars), v)
                    for k, v in self.comp.inplace.items())


    @property
    def libs(self):
        return self.comp.libs

    @property
    def includes(self):
        return self.comp.includes

class ProfileMPI(Profile):
    def __init__(self, comp, duration=None):
        self.comp = comp
        self.duration = duration or new_time()
        self._time_vars = self.duration, new_time(), new_time()

    inputs  = property(lambda self: self.comp.inputs)
    time_vars = property(lambda self: self._time_vars)
    outputs = property(lambda self: self.time_vars + self.comp.outputs)

    @property
    def libs(self):
        return self.comp.libs + ['mpi']

    @property
    def includes(self):
        return self.comp.includes + ['mpif']

    def fortran_call(self, input_names, output_names):
        duration, start, end = output_names[:3]
        comp_output_names = output_names[3:]
        d = locals()
        return (['%(start)s = MPI_Wtime()' % d] +
                self.comp.fortran_call(input_names, comp_output_names) +
                ['%(end)s = MPI_Wtime()' % d,
                 '%(duration)s = %(end)s - %(start)s' % d])
