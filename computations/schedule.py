

def data_dependence(a, b):
    """ Comparator for data dependence """
    if a.depends_on(b): return +1
    if b.depends_on(a): return -1
    return 0


def schedule(computations, *cmps):
    from posort import posort
    return posort(computations, data_dependence, *cmps)
