from computations.core import Computation

patterns = []

class inc(Computation):
    def __init__(self, arg):
        self.inputs = (arg,)
        self.outputs = (arg + 1,)

class double(Computation):
    def __init__(self, arg):
        self.inputs = (arg,)
        self.outputs = (arg * 2,)


class add(Computation):
    def __init__(self, *args):
        self.inputs = args
        self.outputs = (args[0] + args[1],)

class incdec(Computation):
    def __init__(self, arg):
        self.inputs = (arg,)
        self.outputs = (arg + 1,  arg - 1)

class flipflop(Computation):
    def __init__(self, *args):
        a, b = args
        self.inputs = (a, b)
        self.outputs = ((a, b), (b, a))

