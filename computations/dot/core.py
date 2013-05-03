
def interesting_var(x):
    if any(x == y for y in [0, 1]):
        return False
    return True

def interesting_edge((a, b)):
    return all(map(interesting_var, (a, b)))

def declare_var(variable):
    if hasattr(variable, '_write_dot'):
        return variable._write_dot()
    return '"%s" [shape=ellipse]'%nstr(variable)

def declare_comp(comp):
    if hasattr(comp, '_write_dot'):
        return comp._write_dot()
    return '"%s" [shape=box, label="%s"]'%(nstr(comp), nstr(comp.__class__.__name__))

def call(comp):
    return '\n'.join(['"%s" -> "%s"' % tuple(map(nstr, edge))
                        for edge in comp.edges()
                        if interesting_edge(edge)])

def nstr(o):
    """ Normalized str """
    return str(o).replace('\n', '')

template =\
'''
digraph{

%(flags)s

%(variables)s

%(computations)s

%(calls)s
}
'''

defaults={'orientation': 'TD'}

def generate_dot(computation, **kwargs):
    """ Generate DOT string for computation """
    flags = defaults.copy(); flags.update(kwargs)
    flags = '\n'.join(['%s=%s'%item for item in flags.items()])
    variables = '\n'.join(map(declare_var,
                              filter(interesting_var, computation.variables)))
    computations = '\n'.join(map(declare_comp, computation.toposort()))
    calls = '\n'.join(map(call, computation.toposort()))
    return template % locals()


def writedot(computation, filename, **kwargs):
    with open(filename+'.dot', 'w') as f:
        f.write(generate_dot(computation, **kwargs))


def writepdf(computation, filename, extension='pdf', **kwargs):
    writedot(computation, filename, **kwargs)

    import os
    os.system('dot -T%s %s.dot -o %s.%s' % (
                    extension, filename, filename, extension))


def show(computation, filename='comp', extension='pdf', **kwargs):
    import os
    writepdf(computation, filename)
    os.system('evince %s.pdf' % filename)
