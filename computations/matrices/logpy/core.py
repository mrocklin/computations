from computations.matrices.core import MatrixCall
import computations.logpy

from computations.matrices.blocks import Join, Slice
from term import termify

for cls in [Join, Slice, MatrixCall]:
    termify(cls)
