program mpi_test
    implicit none
    include 'mpif.h'
    integer :: ierr
    call MPI_Init(ierr)
    call %(subroutine_name)s()
    call MPI_Finalize(ierr)
    stop
end program mpi_test

%(subroutine_definitions)s
