from sympy import symbols, sin, pi, exp, diff
from sympy import factor, simplify

t, x, c = symbols("t x c")  # symbols

# u = sin(pi * x) * (1.0 + t) * exp(-0.5 * t)
u = sin(pi * x) * (1 + t) * exp(c * t)  # function u(t, x)

du_dt = diff(u, t, 1)  # ∂²u/∂t²
d2u_dx2 = diff(u, x, 2)  # ∂²u/∂x²
result = du_dt - d2u_dx2  # ∂²u/∂t² - ∂²u/∂x²

print("u(t,x) =", u)
print("\nd_t u =", du_dt)
print("\nd_x^2 u =", d2u_dx2)
# print("\nd_t u - d_x^2 u =", result)
print("\nd_t u - d_x^2 u (factorized) =", factor(simplify(result)))

result_sub = result.subs(c, -1./2.)
print("\nd_t^2 u - d_x^2 u with c = -1/2:", result_sub)
