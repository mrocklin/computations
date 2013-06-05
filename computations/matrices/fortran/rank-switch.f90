subroutine rank_switch()
    implicit none
    include 'mpif.h'

    integer rank, size, ierr

    call MPI_COMM_RANK( MPI_COMM_WORLD, rank, ierr )
    call MPI_COMM_SIZE( MPI_COMM_WORLD, size, ierr )

    if (size .lt. %(max_rank)s)  print *, 'Need %(max_rank)s processes'

    %(switch)s
end subroutine rank_switch
