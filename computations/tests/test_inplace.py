from computations.inplace import (make_getname, Copy, inplace,
        purify_one, tokenize_one, ExprToken, tokenize,
        copies_one, purify, inplace_tokenize, remove_single_copies,
        inplace_compile, TokenComputation, valid_name)
from computations.core import CompositeComputation

from computations.example import inc, flipflop

a,b,c,x,y,z = 'abcxyz'

class inci(inc):
    inplace = {0: 0}

class flipflopi(flipflop):
    inplace = {0: 0, 1: 1}

def test_getname():
    getname = make_getname()
    assert getname('x') == 'x'
    assert getname('y') == 'y'
    class A(object):
        def __init__(self, data):
            self.data = data
        def __str__(self):
            return str(self.data)
    assert getname(A('z')) == 'z'
    assert getname(A('z')) != 'z'
    assert getname('a', 'name') == 'name'
    assert getname('b', 'name') != 'name'
    assert getname('c', '0') != '0'         # 0 not a valid name
    assert len(set(map(getname, (1, 2, 2, 2, 3, 3, 4)))) == 4

def test_valid_name():
    assert valid_name('hello')
    assert not valid_name('hello(5)')
    assert not valid_name('123')

def test_inplace():
    assert inplace(inc(3)) == {}
    assert inplace(inci(3)) == {0: 0}

def test_tokenize_one():
    comp = tokenize_one(inc(3), make_getname())
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

def test_tokenize():
    comp = tokenize(inc(3), make_getname())
    assert comp.op == inc
    assert comp.inputs[0].expr == 3
    assert comp.outputs[0].expr == 4

    comp2 = tokenize(inc(3) + inc(4), make_getname())
    assert len(comp2.computations) == 2
    assert comp2.inputs[0].expr == 3
    assert comp2.outputs[0].expr == 5

def test_copies_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert copies_one(comp, tokenizer) == []

    comp = tokenize(inci(3), tokenizer)
    copy = copies_one(comp, tokenizer)[0]
    assert copy.op == Copy
    assert copy.inputs[0].expr == comp.inputs[0].expr
    assert copy.inputs[0].token == comp.inputs[0].token
    assert copy.inputs[0].token != copy.outputs[0].token
    assert copy.outputs[0].token != comp.inputs[0].token

    comp = tokenize(flipflopi(x, y), tokenizer)
    comp = tokenize(flipflopi(x, y), tokenizer)
    assert len(copies_one(comp, tokenizer)) == 2

def test_purify_one():
    tokenizer = make_getname()
    comp = tokenize(inc(3), tokenizer)
    assert purify_one(comp, tokenizer) == comp

    comp = tokenize(inci(3), tokenizer)
    purecomp = purify_one(comp, tokenizer)
    assert len(purecomp.computations) == 2
    a, b = purecomp.computations
    cp, incinpl = (a, b) if a.op==Copy else (b, a)
    assert cp.outputs == incinpl.inputs
    assert cp.inputs == comp.inputs
    assert incinpl.outputs == comp.outputs
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

    comp = tokenize(flipflopi(x, y), tokenizer)
    comp = tokenize(flipflopi(x, y), tokenizer)
    assert len(purify_one(comp, tokenizer).computations) == 3

def test_purify():
    tokenizer = make_getname()
    assert purify(tokenize(inc(3), tokenizer), tokenizer) == \
            tokenize(inc(3), tokenizer)
    assert purify(tokenize(inci(3), tokenizer), tokenizer) == \
            purify_one(tokenize(inci(3), tokenizer), tokenizer)

    tokenizer = make_getname()
    comp = tokenize(inc(3) + inci(4) + inci(5), tokenizer)
    purecomp = purify(comp, tokenizer)

    assert len(purecomp.computations) == 5
    assert purecomp.inputs == comp.inputs
    assert purecomp.outputs == comp.outputs

def test_copy_keyword():
    class Copy2(Copy): pass

    tokenizer = make_getname()
    comp = tokenize(inci(3), tokenizer)
    purecomp = purify_one(comp, tokenizer, Copy=Copy2)
    assert any(c.op == Copy2 for c in purecomp.computations)

def test_inplace_tokenize():
    comp     = TokenComputation(inci(1), [1], [2])
    expected = TokenComputation(inci(1), [1], [1])
    assert inplace_tokenize(comp) == expected

    comp     = TokenComputation(inci(1), [1], [2]) + TokenComputation(inci(2), [2], [3])
    expected = TokenComputation(inci(1), [1], [1]) + TokenComputation(inci(2), [1], [1])
    assert inplace_tokenize(comp) == expected

def test_remove_single_copies():
    comp     = (TokenComputation(inci(1), ['1'], ['2']) +
                TokenComputation(Copy(1), ['0'], ['1']))
    expected =  TokenComputation(inci(1), ['0'], ['2'])
    assert remove_single_copies(comp) == expected
