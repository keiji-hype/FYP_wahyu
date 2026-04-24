import sympy as sp

# 1. Define your symbolic variables
# We tell Python these are real, positive numbers so it doesn't give us imaginary roots
r, M = sp.symbols('r M', positive=True, real=True)

# 2. Define the lapse function f(r) for Schwarzschild
f_r = 1 - (2*M / r)

# 3. Define the Effective Potential V(r)
# Based on the exact equation you extracted from your PDF
V_r = (1 / r**2) * f_r

# 4. Find the Photon Sphere (r_ph)
# Take the derivative of V(r) with respect to r
dV_dr = sp.diff(V_r, r)

# Set the derivative to 0 and solve for r
# This returns a list of solutions, we take the first valid one
r_ph_solutions = sp.solve(dV_dr, r)
r_ph = r_ph_solutions[0]

print(f"Photon Sphere radius (r_ph): {r_ph}") # Should output 3*M

# 5. Calculate the Critical Impact Parameter (b_crit / Shadow Radius)
# Plug r_ph back into V(r) to find the maximum potential barrier
V_max = V_r.subs(r, r_ph)

# The shadow radius is b_crit = 1 / sqrt(V_max)
b_crit = 1 / sp.sqrt(V_max)

print(f"Critical Impact Parameter (b_crit): {b_crit}") # Should output 3*sqrt(3)*M