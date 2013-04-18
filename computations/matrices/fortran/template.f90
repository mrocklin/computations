%(subroutine_header)s

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
%(statements)s

! ======================= !
! Variable Deconstruction !
! ======================= !
%(variable_destructions)s

%(array_deallocations)s

  return
%(footer)s

%(function_definitions)s

