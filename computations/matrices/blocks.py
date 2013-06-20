from computations import Computation

class Slice(Computation):
    """ Take a slice from a Matrix

    inputs:
        A SymPy MatrixSlice object
    """
    def __init__(self, matrixslice):
        self.inputs = (matrixslice.parent,)
        self.outputs = (matrixslice,)

    def fortran_call(self, input_names, output_names):
        arg     = self.outputs[0]
        parent  = input_names[0]
        child   = output_names[0]
        # Fortran Indexing starts at 1, ends at stated element
        rowslice = arg.rowslice[0]+1, arg.rowslice[1], arg.rowslice[2]
        colslice = arg.colslice[0]+1, arg.colslice[1], arg.colslice[2]

        s = "%s = %s(%s:%s:%s, %s:%s:%s)"%((child, parent) + rowslice+colslice)
        s = s.replace(':1,', ',').replace(':1)', ')')  # remove :1 strides
        return s

class Join(Computation):
    """ Join Matrices into a BlockMatrix

    inputs:
        A SymPy BlockMatrix
    """
    def __init__(self, blockmatrix):
        self.inputs = tuple(blockmatrix.blocks)
        self.outputs = (blockmatrix,)

    def fortran_call(self, input_names, output_names):
        parent = output_names[0]
        blocks = iter(input_names)

        bm = self.outputs[0]
        rv = []
        rowstart = 1
        for row in bm.blocks.tolist():
            colstart = 1
            for block in row:
                rowend = rowstart + block.rows - 1
                colend = colstart + block.cols - 1
                blockname = next(blocks)
                rv.append("%(parent)s(%(rowstart)s:%(rowend)s, %(colstart)s:%(colend)s) = %(blockname)s" % locals())
                colstart += block.cols
            rowstart += block.rows
        return rv
