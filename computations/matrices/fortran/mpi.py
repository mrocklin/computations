from computations.matrices.fortran.core import generate, comp_dir


with open(comp_dir + 'matrices/fortran/mpi-program-template.f90') as f:
    mpi_test_template = f.read()

def generate_mpi_tester(comp, *args, **kwargs):
    generate_fns = kwargs.get('generate_fns', [generate])
    subroutine_name = kwargs.get('name', 'f')

    subroutine_definitions = '\n\n'.join(g(comp, *args, **kwargs) for g in generate_fns)

    return mpi_test_template % locals()


