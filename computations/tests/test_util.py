from computations.util import toposort, unique, reverse_dict, merge


def test_unique():
    assert tuple(unique((1, 3, 1, 2))) == (1, 3, 2)
    assert tuple(unique((1, 3, 1, 2), key=lambda x: x%2)) == (1, 2)

def test_toposort():
    edges = {1: set((4, 6, 7)), 2: set((4, 6, 7)),
             3: set((5, 7)),    4: set((6, 7)), 5: set((7,))}
    order = toposort(edges)
    assert not any(a in edges.get(b, ()) for i, a in enumerate(order)
                                         for b    in order[i:])

def test_reverse_dict():
    d = {'a': (1, 2), 'b': (2, 3), 'c': ()}
    assert reverse_dict(d) == {1: set(['a']), 2: set(['a', 'b']), 3: set(['b'])}


def test_merge():
    assert merge({1: 2}, {2: 3}, {3: 4, 1: 5}) == {1: 5, 2: 3, 3: 4}
