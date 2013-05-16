from itertools import chain
from sympy import Symbol
from computations.core import Computation

class Ticks(Symbol):
    def fortran_type(self):
        return 'integer'

class Duration(Symbol):
    def fortran_type(self):
        return 'real*8'

class ClockVar(Ticks): pass

def new_duration():
    new_duration.i += 1
    return Duration('time_%d' % new_duration.i)
new_duration.i = 0


Rate = ClockVar('clock_rate')
Max  = ClockVar('clock_max')

class Profile(Computation):
    """ A Computation to profile another computation """
    def __init__(self, comp, duration=None):
        self.comp = comp
        self.duration = duration or new_duration()
        self.ticks = Ticks('start'), Ticks('end')

    inputs  = property(lambda self: self.comp.inputs)
    time_vars = property(lambda self: (self.duration, Rate, Max) + self.ticks)
    outputs = property(lambda self: self.time_vars + self.comp.outputs)

    def arguments(self, inputs, outputs):
        subinputs = inputs
        duration, rate, max, start, end = outputs[:5]
        suboutputs = outputs[5:]
        subarguments = self.comp.arguments(subinputs, suboutputs)

        return (duration, start, end, rate, max) + tuple(subarguments)

    def fortran_call(self, input_names, output_names):
        duration, rate, max, start, end = output_names[:5]
        comp_output_names = output_names[5:]
        template = ('call system_clock ( %(start)s, %(rate)s, %(max)s )\n  ' +
                self.comp.fortran_call(input_names, comp_output_names) +
                '\n  call system_clock ( %(end)s,   %(rate)s, %(max)s )\n'
                '  %(duration)s = real(%(end)s - %(start)s) / real(%(rate)s)')
        return template % locals()

    @property
    def inplace(self):
        return dict((k+len(self.time_vars), v)
                    for k, v in self.comp.inplace.items())
