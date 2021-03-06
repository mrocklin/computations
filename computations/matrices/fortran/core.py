from computations.core import Computation, unique, CompositeComputation
from computations.schedule import schedule
from computations.inplace import TokenComputation, ExprToken, inplace_compile
from computations.util import groupby, remove, iterable
from computations.matrices.fortran.util import (join, is_number, constant_arg,
        update_class)
from functools import partial
from sympy import MatrixExpr, Expr, ZeroMatrix, assuming, ask, Q, Dummy, solve
import os

import computations
comp_dir = computations.__file__
comp_dir = comp_dir[:comp_dir.rfind('/')+1]

with open(comp_dir + 'matrices/fortran/template.f90') as f:
    template = f.read()
with open(comp_dir + 'matrices/fortran/f2py-template.f90') as f:
    f2py_template = f.read()
with open(comp_dir + 'matrices/fortran/module-template.f90') as f:
    module_template = f.read()

class FortranPrintableTokenComputation(object):
    def fortran_footer(self, *args):
        return self.comp.fortran_footer(*args)
    def fortran_header(self, *args):
        return self.comp.fortran_header(*args)
    def fortran_use_statements(self):
        return self.comp.fortran_use_statements()
    def fortran_include_statements(self):
        return self.comp.fortran_include_statements()
    def fortran_call(self):
        return self.comp.fortran_call(self.input_tokens, self.output_tokens)
    def pseudocode_call(self):
        return self.comp.pseudocode_call(self.input_tokens, self.output_tokens)


class FortranPrintableComputation(object):

    # DAG Functions
    def fortran_header(self, name, inputs, outputs):
        return '%s(%s)'%(name, ', '.join(list(inputs)+
                                        remove(inputs.__contains__, outputs)))

    def fortran_use_statements(self):
        if isinstance(self, CompositeComputation):
            return sum([c.fortran_use_statements() for c in self.computations], [])
        else:
            return []

    def fortran_include_statements(self):
        if isinstance(self, CompositeComputation):
            return sum([c.fortran_include_statements() for c in self.computations], [])
        else:
            return ["include '%s.h'" % inc for inc in self.includes]

    def fortran_footer(self, name):
        return 'end subroutine %s'%(name)

    # Atomic Computation Functions
    def fortran_call(self, input_names, output_names):
        raise NotImplementedError()
    def pseudocode_call(self, input_names, output_names):
        raise NotImplementedError()

    def fortran_function_interface(self):
        return ''
    def fortran_function_definition(self):
        return ''


update_class(Computation, FortranPrintableComputation)
update_class(TokenComputation, FortranPrintableTokenComputation)

def dtype_of(expr, *assumptions):
    if hasattr(expr, 'fortran_type'):
        return expr.fortran_type()

    with assuming(*assumptions):
        if ask(Q.integer(expr) | Q.integer_elements(expr)) or expr.is_integer:
            result = 'integer'
        elif ask(Q.real(expr) | Q.real_elements(expr)) or expr.is_real:
            result = 'real(kind=8)'
        elif ask(Q.complex(expr) | Q.complex_elements(expr)) or expr.is_complex:
            result = 'complex(kind=8)'
        else:
            raise TypeError('Could not infer type of %s'%str(expr))
    return result


nbytes_dtype = {'integer': 4,
                'real(kind=8)': 8,
                'real*8': 8,
                'complex(kind=8)': 8,
                'complex*8': 8}

def numel(var):
    if isinstance(var, MatrixExpr):
        return var.shape[0] * var.shape[1]
    if isinstance(var, Expr):
        return 1
    raise NotImplementedError()

def nbytes(var, *assumptions):
    dtype = dtype_of(var, *assumptions)
    return nbytes_dtype[dtype] * numel(var)

def inplace_cmp(cmp):
    return lambda a, b: cmp(a.comp, b.comp)

def tokens_of(comp, inputs, outputs, **kwargs):
    from computations.matrices.mpi import mpi_cmps
    cmps = kwargs.get('cmps', [])
    computations = comp.toposort()
    if cmps:
        computations = schedule(computations, *map(inplace_cmp, cmps))
    vars = list(comp.variables)

    input_tokens  = sorted_tokens(unique(comp.inputs), inputs)
    input_vars = [v for v in vars if v.token in input_tokens]
    output_tokens = sorted_tokens(comp.outputs, outputs)
    tokens = list(set(map(gettoken, vars)))
    dimens = remove(constant_arg, dimensions(comp))

    return (computations, vars, input_tokens, input_vars, output_tokens, tokens,
            dimens)

def generate(comp, inputs, outputs, name='f', **kwargs):
    """ Generate Fortran code from a computation

    comp - a tokenized computation from inplace_compile
    inputs  - a list of SymPy (Matrix)Expressions
    outputs - a list of SymPy (Matrix)Expressions
    name    - the name of your subroutine
    """

    (computations, vars, input_tokens, input_vars, output_tokens, tokens,
            dimens) = tokens_of(comp, inputs, outputs, **kwargs)

    function_definitions = join([c.comp.fortran_function_definition()
                                            for c in computations])
    subroutine_header = comp.fortran_header(name, input_tokens, output_tokens)

    use_statements = join(unique(comp.fortran_use_statements()))
    include_statements = join(unique(comp.fortran_include_statements()))

    function_interfaces = join([c.comp.fortran_function_interface()
                                            for c in computations])
    argument_declarations = join([
        declare_variable(token, comp, inputs, outputs)
        for token in unique(input_tokens + output_tokens)])

    assumed_dim_declarations  = map(assumed_dimension_declaration, dimens)

    variable_declarations = join(sorted([
        declare_variable(token, comp, inputs, outputs)
        for token in (set(tokens) - set(input_tokens + output_tokens))])
        + assumed_dim_declarations)

    dimen_inits = map(dimension_initialization,
                      dimens,
                      map(partial(var_that_uses_dimension, vars=input_vars), dimens))
    variable_initializations = join(map(initialize_variable, vars)
                                  + dimen_inits)

    array_allocations = join([allocate_array(v, input_tokens, output_tokens)
                                for v in unique(vars, key=gettoken)])

    def call(c):
        rv = c.fortran_call()
        try:
            rv = ["! " + s for s in c.pseudocode_call()] + rv
            # rv = ['print *, "%s" ' % s for s in c.pseudocode_call()] + rv
        except NotImplementedError:
            pass
        return rv
    statements = join(sum(map(call, computations), []))

    variable_destructions = join(map(destroy_variable, vars))

    array_deallocations = join([deallocate_array(v, input_tokens, output_tokens)
                                for v in unique(vars, key=gettoken)])

    footer = comp.fortran_footer(name)

    return template % locals()


def generate_f2py_header(comp, inputs, outputs, name='f', **kwargs):
    (computations, vars, input_tokens, input_vars, output_tokens, tokens,
            dimens) = tokens_of(comp, inputs, outputs, **kwargs)

    dimen_tokens = map(str, dimens)

    subroutine_header = comp.fortran_header(name,
                                            input_tokens + dimen_tokens,
                                            output_tokens)
    dimension_declarations = join(map(explicit_dimension_declaration, dimens))

    argument_declarations = join([
        declare_variable(token, comp, inputs, outputs,
            shape_str=explicit_shape_str)
        for token in unique(input_tokens + output_tokens)])

    call_statement = "%s(%s)"%(
            name, ','.join(input_tokens
                         + remove(input_tokens.__contains__, output_tokens)))

    return f2py_template % locals()


def generate_module(comp, *args, **kwargs):
    module_name = kwargs.pop('modname', 'mod')
    generate_fns = kwargs.get('generate_fns', [generate, generate_f2py_header])

    includes = join("include '%s.h'" % inc for inc in comp.includes)

    subroutines = '\n\n'.join([g(comp, *args, **kwargs) for g in generate_fns])

    return module_template % locals()


# <KLUDGE>
import os
extra_flags = ["-L/usr/lib",  "-I/home/mrocklin/include"]
mpif90_flags = os.popen('mpif90 -show').read().split()[1:]
# <\KLUDGE>

default_includes = ['/usr/include']
default_flags = ['-ffixed-line-length-0', '-ffree-line-length-0']
def compile(source, filename='tmp.f90', modname='mod',
            libs=[], includes=[]):
    with open(filename, 'w') as f:
        f.write(source)
    compile_file(filename, modname, libs, includes)

def front_flag(flag):
    return flag.lower()[:2] in ('-l', '-i')

def compile_file(filename, modname='mod', libs=[], includes=[]):
    includes = includes + default_includes
    incflags = ['-I'+inc for inc in includes]
    libflags = ['-l'+lib for lib in libs]
    flags = libflags + incflags + default_flags + mpif90_flags + extra_flags
    front_flags = ' '.join(filter(front_flag, flags))
    back_flags  = ' '.join(remove(front_flag, flags))
    command = 'f2py -c '
    command += ('%(filename)s -m %(modname)s %(front_flags)s '
                ' --f90flags="%(back_flags)s"' % locals())
    pipe = os.popen(command)
    text = pipe.read()
    if "Error" in text:
        print command
        print text
        raise ValueError('Did not compile')



def is_token_computation(c):
    return isinstance(list(c.variables)[0], ExprToken)

def build(comp, inputs, outputs, name='f', modname='mod',
            filename='tmp.f90', **kwargs):
    if not iterable(inputs):
        raise TypeError("Inputs not iterable")
    if not iterable(outputs):
        raise TypeError("Outputs not iterable")
    if not is_token_computation(comp):
        from computations.matrices.blas import COPY
        comp = inplace_compile(comp, Copy=COPY)
    source = generate_module(comp, inputs, outputs, name=name,
                               modname=modname, **kwargs)
    compile(source, filename, modname, libs=comp.libs, includes=comp.includes)
    mod = __import__(modname)
    return getattr(getattr(mod, modname), 'py_'+name)

gettoken = lambda x: x.token
def sorted_tokens(source, exprs):
    vars = sorted([v for v in source if v.expr in exprs],
                            key=lambda v: list(exprs).index(v.expr))
    return map(gettoken, vars)


#####################
# Variable Printing #
#####################

def assumed_shape_str(shape):
    """ Fortran string for a shape.  Remove 1's from Python shapes """
    if shape[0] == 1 or shape[1] == 1:
        return "(:)"
    else:
        return "(:,:)"

def explicit_shape_str(shape):
    """ Fortran string for a shape.  Remove 1's from Python shapes """
    if shape[0] == 1:
        return "(%s)"%str(shape[1])
    if shape[1] == 1:
        return "(%s)"%str(shape[0])
    else:
        return "(%s,%s)"%(str(shape[0]), str(shape[1]))

def intent_str(isinput, isinternal, isoutput):
    """ Intent of variable given

    isinput - If it is an input paramter
    isinternal - If it is overwritten internally
    isoutput - If it is an output parameter
    """
    if isinput and isoutput:
        return ', intent(inout)'
    elif isinput and not isoutput and not isinternal:
        return ', intent(in)'
    elif not isinput and isoutput:
        return ', intent(out)'
    else:
        return ''

def declare_variable(token, comp, inputs, outputs, **kwargs):
    internal_vars = set(v for v in comp.variables if v.expr not in inputs)
    isinput  = any(token == v.token for v in comp.inputs if not
            constant_arg(v.expr))
    isoutput = any(token == v.token for v in comp.outputs if not
            constant_arg(v.expr) and v.expr in outputs)
    isinternal = any(token == v.token for v in internal_vars)
    exprs = set(v.expr for v in comp.variables if v.token == token
                                          and not constant_arg(v.expr))
    if not exprs:
        return ''
    expr = exprs.pop()
    typ = dtype_of(expr)
    return declare_variable_string(token, expr, typ, isinput, isinternal, isoutput, **kwargs)


def declare_variable_string(token, expr, typ, is_input, is_internal, is_output,
        shape_str=assumed_shape_str):
    rv = typ
    intent = intent_str(is_input, is_internal, is_output)
    rv += intent
    if isinstance(expr, MatrixExpr) and not is_input and not is_output:
        rv += ", allocatable"
    rv += ' :: ' + token
    if isinstance(expr, MatrixExpr):
        rv += shape_str(expr.shape)
    return rv

def allocate_array(v, input_tokens, output_tokens):
    if (isinstance(v.expr, MatrixExpr) and
        v.token not in input_tokens + output_tokens):
        root = "allocate(%s"%v.token
        if v.expr.shape[0] == 1:
            return root + '(%s))'%v.expr.shape[1]
        if v.expr.shape[1] == 1:
            return root + '(%s))'%v.expr.shape[0]
        return root + "(%s,%s))"%tuple(map(str, v.expr.shape))
    else:
        return ''

def deallocate_array(v, input_tokens, output_tokens):
    if (isinstance(v.expr, MatrixExpr) and
        v.token not in input_tokens + output_tokens):
        return "deallocate(%s)"%v.token
    else:
        return ''

def initialize_variable(v):
    if hasattr(v.expr, 'fortran_initialize'):
        return v.expr.fortran_initialize(v.token)
    return ''

def destroy_variable(v):
    if hasattr(v.expr, 'fortran_destroy'):
        return v.expr.fortran_destroy(v.token)
    return ''

def assumed_dimension_declaration(dimen):
    return "integer :: %s" % str(dimen)

def explicit_dimension_declaration(dimen):
    return "integer, intent(in) :: %s" % str(dimen)

def dimension_initialization(dimen, var):
    name = 'fhsghsf'
    x = Dummy(name)
    shape = var.expr.shape
    if shape[0].has(dimen):
        idx, d = 0, shape[0]
    elif shape[1].has(dimen):
        idx, d = 1, shape[1]
    else:
        raise ValueError()
    expr = solve(d - x, dimen)[0]
    return (str(dimen) + ' = ' +
            str(expr).replace(str(x), 'size(%s, %d)'%(var.token, idx+1)))

def var_that_uses_dimension(dimen, vars):
    return next(v for v in vars if isinstance(v.expr, MatrixExpr)
                               and dimen in v.expr.shape
                                or any(d.has(dimen) for d in v.expr.shape))

def dimensions(comp):
    """ Collect all of the dimensions in a computation

    For example if a computation contains MatrixSymbol('X', n, m) then n and m
    are in the set returned by this function """
    shapes = [v.expr.shape for v in comp.variables
                           if isinstance(v.expr, MatrixExpr)]
    return set([s for shape in shapes for v in shape for s in v.free_symbols])
