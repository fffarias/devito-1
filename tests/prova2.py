from devito import *
from devito.types import _SymbolCache

grid = Grid(shape=(4, 4, 4))
x, y, z = grid.dimensions
t = grid.stepping_dim

f = TimeFunction(name='f', grid=grid, space_order=4)
g = Function(name='g', grid=grid)

eq = Eq(f.forward, (f.dx*cos(g)).dy + sin(g))

op = Operator(eq)

print(op)

from devito.finite_differences.differentiable import IndexSum, Dot
from devito.types import StencilDimension

i = StencilDimension('i', 3)
expr = f.subs(x, x + i)
a = IndexSum(expr, i)

j = StencilDimension('j', 3)
expr0 = expr.subs(y, y + j)
expr1 = expr.subs(z, z + j)
expr2 = f.subs(x, x + j)
b = Dot(Dot(expr0, expr1, i), expr2, j)
from IPython import embed; embed()
