from computations.core import Computation
from computations.matrices.core import MatrixCall
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy import Basic

class ElemProd(MatrixCall):
    @property
    def inputs(self):
        return (self.args[0],self.args[1])

    @property
    def outputs(self):
        return (HadamardProduct(self.inputs[0], self.inputs[1]), )

    def fortran_call(self, input_names, output_names):
        d = {'out_name'  : output_names[0],
             'in_name_1' : input_names[0],
             'in_name_2' : input_names[1] }
        return ('%(out_name)s = %(in_name_1)s * %(in_name_2)s ') % d 

