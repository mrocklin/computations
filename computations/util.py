import itertools

identity = lambda x: x
def unique(seq, key=identity):
    seen = set()
    for item in seq:
        k = key(item)
        if k not in seen:
            seen.add(k)
            yield item

def intersect(a, b):
    return not not set(a).intersection(set(b))

def remove(predicate, collection):
    return [item for item in collection if not predicate(item)]

def toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)

    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges

    >>> from computations.util import toposort
    >>> toposort({1: {2, 3}, 2: (3, )})
    [1, 2, 3]

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms

    note: This function was originally written for the Theano Project
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = dict((k, set(val)) for k, val in incoming_edges.items())
    S = set((v for v in edges if v not in incoming_edges))
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d):
    """ Reverses direction of dependence dict

    >>> from computations.util import reverse_dict
    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)
    {1: set(['a']), 2: set(['a', 'b']), 3: set(['b'])}
    """
    result = {}
    for key in d:
        for val in d[key]:
            if val not in result:
                result[val] = set()
            result[val].add(key)
    return result

def merge(*dicts):
    """ Merge several dictionaries """
    out = dict()
    for d in dicts:
        out.update(d)
    return out


def groupby(f, coll):
    d = dict()
    for item in coll:
        key = f(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d

def remove(predicate, collection):
    return [item for item in collection if not predicate(item)]

def iterable(x):
    try:
        iter(x)
        return True
    except:
        return False

def chunked(seq, n):
    """ Sequence separated into chunks

    >>> from computations.util import chunked
    >>> list(chunked([1,2,3,4,5,6], 2))
    [[1, 2], [3, 4], [5, 6]]
    """

    for i in range(0, len(seq), n):
        yield seq[i:i+n]


class memoize(object):
    def __init__(self, f):
        self.cache = {}
        self.f = f

    def key(self, *args, **kwargs):
        return (tuple(args), frozenset(kwargs.items()))

    def __call__(self, *args, **kwargs):
        k = self.key(*args, **kwargs)
        if k not in self.cache:
            self.cache[k] = self.f(*args, **kwargs)
        return self.cache[k]
