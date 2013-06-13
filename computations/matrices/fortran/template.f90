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

