from computations.matrices.fortran.core import generate, comp_dir
from computations.util import chunked
from computations.matrices.mpi import mpi_cmps


with open(comp_dir + 'matrices/fortran/mpi-program-template.f90') as f:
    mpi_test_template = f.read()

def mpi_test_program(subroutine_name):
    return mpi_test_template % locals()

with open(comp_dir + 'matrices/fortran/rank-switch.f90') as f:
    rank_switch_template = f.read()

def rank_switch(d):
    """ Fortran code to switch subroutine based on MPI rank """
    max_rank = max(d.keys()) + 1
    switch = '\n    '.join('if (rank .eq. %(rank)d)  call %(fn)s()'%{'rank': k, 'fn': v}
            for k, v in d.items())
    return rank_switch_template % locals()

def generate_mpi(*args, **kwargs):
    """

    inputs: comp, inputs, outputs, name, comp, inputs, outputs, name ...
    """
    cmps = kwargs.pop('cmps', [])
    if set(mpi_cmps).issubset(set(cmps)):
        cmps = cmps + type(cmps)(*mpi_cmps)
    comps = args[0::4]
    names = args[3::4]
    codes = [generate(comp, inputs, outputs, name=name, cmps=cmps, **kwargs)
            for comp, inputs, outputs, name in chunked(args, 4)]
    return _generate_mpi(comps, names, codes, **kwargs)


def _generate_mpi(comps, names, codes):
    ranks = dict(zip(range(len(comps)), names))
    return '\n\n'.join([
        mpi_test_program('rank_switch'),
        rank_switch(ranks)] +
        codes)
