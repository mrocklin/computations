subroutine %(subroutine_header)s

%(use_statements)s

  implicit none

%(include_statements)s

! ===================== !
! Argument Declarations !
! ===================== !
%(argument_declarations)s

! ===================== !
! Variable Declarations !
! ===================== !
%(variable_declarations)s
!  real(kind=8) :: starttime, endtime
!  integer :: ierr

  interface
%(function_interfaces)s
  end interface

! ======================== !
! Variable Initializations !
! ======================== !
%(variable_initializations)s

%(array_allocations)s

! ========== !
! Statements !
! ========== !

!   call MPI_barrier(MPI_COMM_WORLD, ierr)
!   starttime = MPI_Wtime()
%(statements)s
!  endtime   = MPI_Wtime()
!  print *, endtime-starttime

! ======================= !
! Variable Deconstruction !
! ======================= !
%(variable_destructions)s

%(array_deallocations)s

  return
%(footer)s

%(function_definitions)s

