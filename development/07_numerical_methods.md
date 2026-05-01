# Numerical Methods Reference

---

## ODE Solver Selection: Why Radau

Rocket engine transient ODEs are stiff. Stiffness comes from:

1. **Pressure wave propagation**: acoustic timescale ~L/a ~ 1ms (fast)
2. **Rotor acceleration**: mechanical timescale ~I/τ ~ 0.1–1s (slow)
3. **Wall heating**: thermal timescale ~m*Cp/h*A ~ 10–100s (very slow)

Stiffness ratio: fastest/slowest ≈ 10⁵. Explicit methods (RK45) require step
sizes ≤ 1ms to maintain stability, even when accuracy only requires 10ms steps.
This is 10× wasted work.

Radau (implicit Runge-Kutta, order 5) is A-stable: stable for any step size.
It solves an implicit system at each step but adapts step size to accuracy,
not stability. For this problem, Radau is ~10–50x faster than RK45.

### Radau Configuration
```python
scipy.integrate.solve_ivp(
    fun=rhs,
    t_span=(t0, tf),
    y0=X0,
    method='Radau',
    rtol=1e-4,
    atol=1e-6,
    jac=jac_fn,          # optional: provide Jacobian for ~5x speedup
    jac_sparsity=S,      # optional: sparse pattern for efficient FD Jacobian
    max_step=1e-3,       # cap at 1ms to resolve fast valve events
    dense_output=True,
)
```

---

## Newton-Raphson for Steady-State

```
Given: F(x) = 0 to solve (residual vector)
1. x_0 = initial guess
2. Scale: x_s = x / x_ref  (diagonal preconditioning, x_ref = |x_0| + 1)
3. For k = 1..max_iter:
   a. Compute F(x_k)
   b. If ||F(x_k)||_∞ < atol: converged ✓
   c. Build Jacobian J_k = ∂F/∂x at x_k  (FD with sparsity exploitation)
   d. Solve: Δx = -J_k⁻¹ × F(x_k)  (sparse LU decomposition)
   e. Line search: find α ∈ (0,1] s.t. ||F(x+α×Δx)||_2 < ||F(x)||_2
   f. x_{k+1} = x_k + α×Δx
4. If not converged: raise ConvergenceError with diagnostics
```

### Variable Scaling
Critical for mixed-unit state vectors (P in 1e7 Pa, h in 1e6 J/kg, ω in 1e3 rad/s).
Without scaling, the Newton step is dominated by the largest-magnitude variable.

```python
x_ref = np.abs(x_initial) + 1.0  # avoid division by zero
```

---

## Inner Newton for DAE Handling

The engine system has the form:
- Differential states X: dX/dt = f(t, X, Z)
- Algebraic constraints: g(t, X, Z) = 0

Inner Newton eliminates Z at each time step:

```python
def inner_newton(t, X, Z_guess, layout, tol=1e-8, max_iter=20):
    Z = Z_guess.copy()
    for i in range(max_iter):
        r = layout.algebraic_residuals(t, X, Z)
        if np.linalg.norm(r) < tol:
            return Z
        J = finite_diff_jacobian(lambda z: layout.algebraic_residuals(t, X, z), Z)
        Z -= np.linalg.solve(J, r)
    raise ConvergenceError(f"Inner Newton failed at t={t:.6f}")
```

Z_guess warm-starts from previous successful step → typically converges in 2-3 iterations.

---

## Jacobian via Finite Differences with Sparsity

```python
# Group non-overlapping columns (graph coloring on sparsity pattern)
# Perturb one group at a time: O(n_colors) evaluations vs O(n_vars)
# For typical engines: n_colors ~ 10-20 vs n_vars ~ 100

h = 1e-6  # finite difference step
for group in column_groups:
    x_plus = x.copy()
    x_plus[group] += h  # perturb all columns in group simultaneously
    F_plus = F(x_plus)
    for j in group:
        J[:, j] = (F_plus - F0) / h
```

---

## Linearization

State-space matrices via complex-step finite differences:

```
A[i,j] = ∂(dX_i/dt) / ∂X_j  |_{trim}  (complex step: h=1e-20j)
B[i,j] = ∂(dX_i/dt) / ∂U_j  |_{trim}
C[i,j] = ∂Y_i / ∂X_j         |_{trim}
D[i,j] = ∂Y_i / ∂U_j         |_{trim}
```

Complex step eliminates cancellation error in finite differences:
```python
f_plus = rhs(t, X + 1j*1e-20*e_k)
A[:,k] = np.imag(f_plus) / 1e-20  # exact to machine precision
```

---

## Stiff System Integration Strategy

For the TTBE engine (122 states, stiffness ratio ~10^5):

1. Start with loose tolerances (rtol=1e-3) during model development
2. Tighten to (rtol=1e-4, atol=1e-6) for validation runs
3. Provide Jacobian sparsity pattern to solve_ivp for 5-10x speedup
4. Use max_step=1e-3 to ensure valve events are resolved
5. Use dense_output=True to interpolate output at fixed recording intervals
   without affecting solver step size selection

Expected performance: 10 seconds of engine time in ~30-60 seconds wall time
(122-state system with Radau and sparse Jacobian).
