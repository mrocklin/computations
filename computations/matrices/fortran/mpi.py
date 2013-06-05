from computations.matrices.fortran.core import generate, comp_dir


with open(comp_dir + 'matrices/fortran/mpi-program-template.f90') as f:
    mpi_test_template = f.read()

def mpi_test_program(subroutine_name):
    return mpi_test_template % locals()

with open(comp_dir + 'matrices/fortran/rank-switch.f90') as f:
    rank_switch_template = f.read()

def rank_switch(d):
    """ Fortran code to switch subroutine based on MPI rank """
    max_rank = max(d.keys()) + 1
    switch = '\n    '.join('if (rank .eq. %(rank)d)  %(fn)s()'%{'rank': k, 'fn': v}
            for k, v in d.items())
    return rank_switch_template % locals()
