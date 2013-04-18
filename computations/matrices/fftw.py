from computations.core import Computation
from sympy.matrices.expressions.fourier import DFT
from sympy import Basic

'''
class FFTWPlan(Computation):
  @property
  def outputs(self):
    return (Plan(),)

plan = fftw_plan_r2r_1d(N,
'''

class Plan(Basic):
    name = 'plan'

    def fortran_destroy(self, token):
        assert isinstance(token, str)
        return '! destroy ' + token


class FFTW(Computation):
    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (DFT(self.inputs[0].shape[0]) * self.inputs[0], Plan() )

    def fortran_include_statements(self):
        return ["include 'fftw3.f03'"]

    def fortran_use_statements(self):
       return ["use, intrinsic :: iso_c_binding"]

    def fortran_call(self, input_names, output_names):
        d = {'plan_name': output_names[1],
                'in_name': input_names[0],
                'out_name': output_names[0],
                'n': self.inputs[0].shape[0]}
        return ('%(plan_name)s = fftw_plan_dft_1d(%(n)s, %(in_name)s, %(out_name)s, FFTW_FORWARD,FFTW_ESTIMATE) \n'
                'call fftw_execute_dft(%(plan_name)s, %(in_name)s, %(out_name)s) \n'
                'call fftw_destroy_plan(%(plan_name)s)') % d
