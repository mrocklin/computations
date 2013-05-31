from computations.core import Computation, CompositeComputation, Identity
from functools import partial
import re


def valid_name(n):
    if not n or not isinstance(n, str):
        return False
    match = re.search('[a-zA-Z]\w*', n)
    if not match or match.group() != n:
        return False
    return True

def make_getname():
    """ Make a new tokenizer

    Tokenizers maintian state for which variables they have already seen.
    This function makes a new function with a new bound cache
    """

    cache = {}
    seen = set(['', None])

    def getname(key, requested=None):
        """ Get the name associated to a key """
        if key in cache:
            return cache[key]

        if requested and valid_name(requested):
            name = requested
        elif valid_name(str(key)):
            name = str(key)
        else:
            name = 'var'
        if name in seen:
            id = 2
            while(name + '_' + str(id) in seen):
                id += 1
            name = name + '_' + str(id)

        assert name not in cache.values()
        assert name not in seen

        cache[key] = name
        seen.add(name)
        assert isinstance(name, str)
        return name
    return getname


def inplace(x):
    """ Get a dict mapping storage location of each output

    {1: 2} means that the output of index 1 is stored in the input with index 2
    """
    try:
        return x.inplace
    except AttributeError:
        try:
            return x.op.inplace
        except AttributeError:
            pass
    return {}


class Copy(Computation):
    """ A Copy computation """
    def __init__(self, *inputs):
        self.inputs = inputs

    outputs = property(lambda self: self.inputs)

def copies_one(comp, getname, **kwargs):
    """ The necessary copies to make an impure computation pure """
    copy = kwargs.get('Copy', Copy)
    def new_comp(inp, out):
        requested = inp.token if '_' not in inp.token else None
        newtoken = getname((inp.expr, out.expr), requested)
        out = ExprToken(inp.expr, newtoken)
        return TokenComputation(copy(inp.expr), [inp.token], [newtoken])

    return [new_comp(comp.inputs[v], comp.outputs[k])
                 for k, v in inplace(comp).items()]

def purify_one(comp, getname, **kwargs):
    """ A pure version of a single impure computation.

    Adds copies and returns a Composite

    See Also
        purify
    """
    copies = copies_one(comp, getname, **kwargs)
    d = dict((cp.input_tokens[0], cp.output_tokens[0]) for cp in copies)
    if not d:
        return comp

    input_tokens = tuple(d[i] if i in d else i for i in comp.input_tokens)

    newcomp = TokenComputation(comp.comp, input_tokens, comp.output_tokens)  #.canonicalize() ??

    return CompositeComputation(newcomp, *copies)

def purify(comp, getname, **kwargs):
    """ Pure version of an impure computation

    Adds copies and returns a Composite

    See Also
        purify_one
    """
    if not isinstance(comp, CompositeComputation):
        return purify_one(comp, getname, **kwargs)
    return CompositeComputation(*[purify_one(c, getname, **kwargs)
                                    for c in comp.computations])

class ExprToken(object):
    """ A pair of mathematical Expr and computational Token

    The expr contains all necessary mathematical information.
    The token contains all variable information. It is a valid variable name.
    """
    def __init__(self, expr, token):
        self.expr = expr
        self.token = token

    def __str__(self):
        return "%s @ %s" %(self.expr, self.token)

    def _info(self):
        return type(self), self.expr, self.token
    def __hash__(self):
        return hash(self._info())
    def __eq__(self, other):
        return type(self) == type(other) and self._info() == other._info()


def tokenize_one(mathcomp, tokenizer):
    """ Transform mathematical computation into a computation of ExprTokens

    This is the switch from pure math to thinking about variables and memory

    Works on only a single computaion (not a composite)

    See Also
        tokenize
    """
    return TokenComputation(mathcomp, map(tokenizer, mathcomp.inputs),
                             map(tokenizer, mathcomp.outputs))

def tokenize(mathcomp, tokenizer=None):
    """ Transform mathematical computation into a computation of ExprTokens

    This is the switch from pure math to thinking about variables and memory

    Works on composites

    See Also
        tokenize_one
    """
    tokenizer = tokenizer or make_getname()
    if not isinstance(mathcomp, CompositeComputation):
        return tokenize_one(mathcomp, tokenizer)
    return CompositeComputation(*[tokenize_one(c, tokenizer)
                                    for c in mathcomp.computations])


def replace_tokens(comp, switch):
    """ Replace tokens in a computation

    switch, a dictionary mapping source to target token """
    if not any(tok in comp.input_tokens+comp.output_tokens for tok in switch):
        return comp
    intoks = [switch.get(t, t) for t in comp.input_tokens]
    outtoks = [switch.get(t, t) for t in comp.output_tokens]
    return TokenComputation(comp.comp, intoks, outtoks)

def inplace_tokenize(comp):
    """ Change tokens to be consistent with inplace dictionaries """
    computations = comp.toposort()
    for i in range(len(computations)):
        c = computations[i]
        if not c.inplace:
            continue
        switch = dict((c.output_tokens[k], c.input_tokens[v])
                            for k,v in c.inplace.items())
        computations[i:] = map(partial(replace_tokens, switch=switch),
                computations[i:])

    return CompositeComputation(*computations)

def remove_single_copies(comp):
    """ Remove unnecessary copies

    The following changes
    In:  a -> Copy -> b -> A -> c
    Out: a -> A -> c

    The following does not change
    In:  a -> Copy -> b -> A -> C
           ->  B   -> c
    """
    users = {}
    computations = comp.toposort()
    for c in computations:
        for inp in c.inputs:
            s = users.get(inp, set())
            s.add(c)
            users[inp] = s

    single_copies = [cp for s in users.values() for cp in s
                        if len(s) == 1 and issubclass(cp.op, Copy)]

    switch = dict((cp.outputs[0].token, cp.inputs[0].token)
                        for cp in single_copies)

    return CompositeComputation(*[replace_tokens(c, switch) for c in computations
                                            if c not in single_copies])

def remove_zero_copies(comp, is_zero=lambda e: not e):
    """ Remove Copies of Zero """
    if not isinstance(comp, CompositeComputation):
        return comp
    def condition(comp):
        return not (issubclass(comp.op, Copy) and is_zero(comp.inputs[0].expr))
    return CompositeComputation(*filter(condition, comp.computations))

def inplace_compile(comp, **kwargs):
    """ Compile a mathematical computation into a nice inplace one

    This is a master function that calls the following in order

    See Also
        tokenize
        purify
        remove_single_copies
        inplace_tokenize
    """
    tokenizer = make_getname()
    stage0 = comp
    stage1 = tokenize(stage0, tokenizer)
    stage2 = purify(stage1, tokenizer, **kwargs)
    stage3 = remove_single_copies(stage2)
    stage4 = inplace_tokenize(stage3)
    stage5 = remove_zero_copies(stage4)
    return stage5

class TokenComputation(Computation):

    def __init__(self, comp, input_tokens, output_tokens):
        self.comp = comp
        self.input_tokens  = tuple(map(str, input_tokens))
        self.output_tokens = tuple(map(str, output_tokens))

    op = property(lambda self: type(self.comp))

    @property
    def inputs(self):
        return tuple(map(ExprToken, self.comp.inputs, self.input_tokens))

    @property
    def outputs(self):
        return tuple(map(ExprToken, self.comp.outputs, self.output_tokens))

    inplace = property(lambda self: inplace(self.comp) or {})

    def _info(self):
        return type(self), self.comp, self.input_tokens, self.output_tokens

    def __str__(self):
        ins  = "["+', '.join(map(str, self.inputs)) +"]"
        outs = "["+', '.join(map(str, self.outputs))+"]"
        return "%s -> %s -> %s"%(ins, str(self.op.__name__), outs)

    @property
    def libs(self):
        return self.comp.libs

    @property
    def includes(self):
        return self.comp.includes

    def _write_dot(self):
        from computations.dot.core import nstr
        return '"%s" [shape=box, label="%s"]'%(nstr(self),
                nstr(self.comp.__class__.__name__))
