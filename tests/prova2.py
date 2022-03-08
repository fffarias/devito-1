from devito import *
from devito.types import _SymbolCache

grid = Grid(shape=(4, 4, 4))

f = TimeFunction(name='f', grid=grid, space_order=4)
g = Function(name='g', grid=grid)

eq = Eq(f.forward, (f.dx*cos(g)).dy + sin(g))

op = Operator(eq)

print(op)
