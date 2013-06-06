from computations.util import (toposort, unique, reverse_dict, merge, groupby,
        remove, iterable, chunked)


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

def test_groupby():
    d = groupby(lambda x: x%2, range(10))
    assert set(d.keys()) == set((0, 1))
    assert set(d[0]) == set((0, 2, 4, 6, 8))
    assert set(d[1]) == set((1, 3, 5, 7, 9))

def test_remove():
    assert remove(str.islower, 'AaBb') == ['A', 'B']

def test_iterable():
    assert iterable([1,2,3])
    assert not iterable(3)
    assert iterable((1,2,3))
    assert iterable(set((1,2,3)))

def test_chunked():
    assert list(chunked(range(9), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
