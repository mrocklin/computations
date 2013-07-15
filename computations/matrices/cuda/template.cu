%(include_statements)s

%(use_statements)s

%(function_header)s
{

/*
 * Copy inputs to the device
 */
%(arg_inits)s

/*
 * Declare locals
 */
%(variable_declarations)s

/*
 * Statements
 */
%(statements)s

/*
 * Copy outputs back to host
 */
%(arg_ends)s

  return EXIT_SUCCESS;
}

