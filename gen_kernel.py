"""
Symbolic derivation of the time-domain stress field of a point moment-tensor source in an
infinite homogeneous elastic medium (Aki & Richards eq. 4.23 Green's function, differentiated
twice).  Output: _kernel_gen.py with a numba-ready function giving, for one (source, receiver)
pair, the 6x10 matrix of coefficients that multiply
    P_n(t - r/alpha)  n=-2..2     and     P_n(t - r/beta)  n=-2..2
where P_n is the n-th time derivative (negative n = integral) of the moment history S(t).
"""
import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

x, y, z, r = sp.symbols('x y z r', positive=True)
al, be, rho, lam, mu = sp.symbols('alpha beta rho lam mu', positive=True)
mxx, myy, mxy = sp.symbols('mxx myy mxy')
N = range(-2, 3)
Pa = {n: sp.Symbol(f'Pa{n}'.replace('-', 'm')) for n in N}
Pb = {n: sp.Symbol(f'Pb{n}'.replace('-', 'm')) for n in N}
X = [x, y, z]

def D(E, k):
    """d/dx_k including chain rule through r and through the retarded-time arguments."""
    out = sp.diff(E, X[k]) + sp.diff(E, r) * X[k] / r
    for n in N:
        for P, c, nm in ((Pa, al, 'a'), (Pb, be, 'b')):
            d = sp.diff(E, P[n])
            if d != 0:
                if n == 2:
                    raise RuntimeError('need P3')
                out += d * (-X[k] / (c * r)) * P[n + 1]
    return out

g = [xi / r for xi in X]
a_, b_ = r / al, r / be
I2 = a_ * Pa[-1] - b_ * Pb[-1] + Pa[-2] - Pb[-2]
delta = lambda i, j: 1 if i == j else 0
G = [[(1 / (4 * sp.pi * rho)) * ((3 * g[i] * g[j] - delta(i, j)) / r**3 * I2
      + g[i] * g[j] / (al**2 * r) * Pa[0]
      + (delta(i, j) - g[i] * g[j]) / (be**2 * r) * Pb[0]) for j in range(3)] for i in range(3)]

m = sp.Matrix([[mxx, mxy, 0], [mxy, myy, 0], [0, 0, 0]])
# u_i = - M_jk dG_ij/dx_k
u = [-sum(m[j, k] * D(G[i][j], k) for j in range(3) for k in range(3) if m[j, k] != 0) for i in range(3)]
H = [[D(u[i], k) for k in range(3)] for i in range(3)]          # H[i][k] = du_i/dx_k
eps = [[(H[i][k] + H[k][i]) / 2 for k in range(3)] for i in range(3)]
tr = eps[0][0] + eps[1][1] + eps[2][2]
comps = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]          # xx yy zz xy yz xz
sig = [lam * delta(i, j) * tr + 2 * mu * eps[i][j] for i, j in comps]

exprs, names = [], []
for ci, s in enumerate(sig):
    s = sp.expand(s)
    for tag, P in (('a', Pa), ('b', Pb)):
        for n in N:
            exprs.append(sp.expand(sp.diff(s, P[n])))
            names.append((ci, tag, n))
uexprs, unames = [], []
for ci in range(3):
    uu = sp.expand(u[ci])
    for tag, P in (('a', Pa), ('b', Pb)):
        for n in N:
            uexprs.append(sp.expand(sp.diff(uu, P[n])))
            unames.append((ci, tag, n))
print('built', len(exprs), 'coefficient expressions; cse...')
repl, red = sp.cse(exprs, optimizations='basic')
pr = PythonCodePrinter()
lines = ['import math', 'import numba as nb', '',
         '@nb.njit(cache=True, fastmath=True)',
         'def coef(x, y, z, r, alpha, beta, rho, lam, mu, mxx, myy, mxy, C):',
         '    """fill C[6,10]: cols 0-4 -> P_{-2..2}(t-r/alpha), cols 5-9 -> P_{-2..2}(t-r/beta)"""']
for sym, ex in repl:
    lines.append(f'    {sym} = {pr.doprint(ex)}')
for (ci, tag, n), ex in zip(names, red):
    col = (n + 2) + (0 if tag == 'a' else 5)
    lines.append(f'    C[{ci},{col}] = {pr.doprint(ex)}')
repl2, red2 = sp.cse(uexprs, optimizations='basic')
lines += ['', '@nb.njit(cache=True, fastmath=True)',
          'def coef_u(x, y, z, r, alpha, beta, rho, lam, mu, mxx, myy, mxy, C):',
          '    """displacement coefficients C[3,10] (same column layout)"""']
for sym, ex in repl2:
    lines.append(f'    {sym} = {pr.doprint(ex)}')
for (ci, tag, n), ex in zip(unames, red2):
    col = (n + 2) + (0 if tag == 'a' else 5)
    lines.append(f'    C[{ci},{col}] = {pr.doprint(ex)}')
open('/home/claude/kinfault/_kernel_gen.py', 'w').write('\n'.join(lines) + '\n')
print('wrote kernel, lines:', len(lines))
