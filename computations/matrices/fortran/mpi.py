from computations.matrices.fortran.core import generate, comp_dir


with open(comp_dir + 'matrices/fortran/mpi-program-template.f90') as f:
    mpi_test_template = f.read()

def generate_mpi_tester(comp, *args, **kwargs):
    generate_fns = kwargs.get('generate_fns', [generate])
    subroutine_name = kwargs.get('name', 'f')

    subroutine_definitions = '\n\n'.join(g(comp, *args, **kwargs) for g in generate_fns)

    return mpi_test_template % locals()

template = """
subroutine rank_switch()
    implicit none
    include 'mpif.h'

    integer rank, size, ierr

    call MPI_COMM_RANK( MPI_COMM_WORLD, rank, ierr )
    call MPI_COMM_SIZE( MPI_COMM_WORLD, size, ierr )

    if (size .lt. %(max_rank)s)  print *, 'Need %(max_rank)s processes'

    %(switch)s
end subroutine rank_switch"""

def rank_switch(d):
    """ Fortran code to switch subroutine based on MPI rank """
    max_rank = max(d.keys()) + 1
    switch = '\n    '.join('if (rank .eq. %(rank)d)  %(fn)s()'%{'rank': k, 'fn': v}
            for k, v in d.items())
    return template % locals()
