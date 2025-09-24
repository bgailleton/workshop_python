# Mathematical expression benchmark: NumPy vs NumExpr vs JAX vs Numba vs f2py
# --------------------------------------------------------------------------
# Expression: result = sin(x) * exp(-y**2) + sqrt(abs(x * z)) - log1p(abs(y))
# Common scientific computing operations including:
# - Trigonometric functions (sin)
# - Exponential functions (exp)
# - Power operations (**2)
# - Element-wise operations (*, +, -)
# - Mathematical functions (sqrt, log1p, abs)

import os, time, math
import numpy as np

# Optional: stabilize CPU timings by pinning threads (comment out if you want full speed)
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numexpr as ne
    NUMEXPR_OK = True
except Exception:
    NUMEXPR_OK = False

try:
    import jax.numpy as jnp
    import jax
    from jax import jit
    JAX_OK = True
except Exception:
    JAX_OK = False

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

try:
    # f2py module will be compiled on first run if needed
    import f2py_math_expr
    F2PY_OK = True
except Exception:
    F2PY_OK = False

# -----------------------
# Problem setup
# -----------------------
np.random.seed(42)
n = 10_000_000                  # 10M elements, adjust if you lack RAM
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
z = np.random.randn(n).astype(np.float32)

print(f"Array size: {n:,} elements ({x.nbytes/1024**2:.1f} MB each)")
print(f"Total memory: {3 * x.nbytes/1024**2:.1f} MB")

# -----------------------
# Pure Python (explicit loop baseline)
# -----------------------
def compute_python(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pure Python loops - slowest but most readable"""
    n = len(x)
    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        result[i] = (math.sin(x[i]) * math.exp(-y[i]**2) +
                    math.sqrt(abs(x[i] * z[i])) -
                    math.log1p(abs(y[i])))
    return result

# -----------------------
# NumPy (vectorized baseline)
# -----------------------
def compute_numpy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.exp(-y**2) + np.sqrt(np.abs(x * z)) - np.log1p(np.abs(y))

# -----------------------
# NumExpr (multi-threaded)
# -----------------------
if NUMEXPR_OK:
    def compute_numexpr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return ne.evaluate("sin(x) * exp(-y**2) + sqrt(abs(x * z)) - log1p(abs(y))")

# -----------------------
# JAX (JIT compiled)
# -----------------------
if JAX_OK:
    def compute_jax_base(x, y, z):
        return jnp.sin(x) * jnp.exp(-y**2) + jnp.sqrt(jnp.abs(x * z)) - jnp.log1p(jnp.abs(y))

    compute_jax = jit(compute_jax_base)

    # Convert arrays to JAX format
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    z_jax = jnp.array(z)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

# -----------------------
# Numba (JIT parallel loops)
# -----------------------
if NUMBA_OK:
    @njit(parallel=True, fastmath=True)
    def compute_numba(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        n = len(x)
        result = np.empty(n, dtype=np.float32)
        for i in prange(n):
            result[i] = (math.sin(x[i]) * math.exp(-y[i]**2) +
                        math.sqrt(abs(x[i] * z[i])) -
                        math.log1p(abs(y[i])))
        return result

# -----------------------
# f2py (Fortran compiled)
# -----------------------
def setup_f2py():
    """Create and compile Fortran module if needed"""
    fortran_code = """
subroutine compute_fortran(x, y, z, result, n)
    implicit none
    integer, intent(in) :: n
    real*4, intent(in) :: x(n), y(n), z(n)
    real*4, intent(out) :: result(n)
    integer :: i

    do i = 1, n
        result(i) = sin(x(i)) * exp(-y(i)**2) + sqrt(abs(x(i) * z(i))) - log(1.0 + abs(y(i)))
    end do
end subroutine compute_fortran
"""

    # Write Fortran file
    with open('f2py_math_expr.f90', 'w') as f:
        f.write(fortran_code)

    # Compile with f2py
    print("Compiling Fortran module with f2py...")
    cmd = "python -m numpy.f2py -c -m f2py_math_expr f2py_math_expr.f90"
    result = os.system(cmd)

    if result == 0:
        print("f2py compilation successful")
        return True
    else:
        print("f2py compilation failed")
        return False

# Try to setup f2py if not available
if not F2PY_OK:
    try:
        if setup_f2py():
            import f2py_math_expr
            F2PY_OK = True
    except Exception as e:
        print(f"f2py setup failed: {e}")
        F2PY_OK = False

if F2PY_OK:
    def compute_f2py(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return f2py_math_expr.compute_fortran(x, y, z, len(x))

# -----------------------
# Benchmark helpers
# -----------------------
def bench(fn, warmups=2, runs=5):
    # warmups
    for _ in range(warmups):
        result = fn()
        # Handle JAX async execution
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        # Handle JAX async execution
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), np.mean(times), np.std(times)

print("\\n=== Computing reference (NumPy) and benchmarking ===")
result_np = compute_numpy(x, y, z)  # reference result

# Pure Python benchmark (on smaller subset for reasonable timing)
n_small = min(100000, n)  # Use smaller array for Python loop
x_small, y_small, z_small = x[:n_small], y[:n_small], z[:n_small]
print(f"Pure Python tested on {n_small:,} elements (would be too slow on full array)")
t_py_med, t_py_mean, t_py_std = bench(lambda: compute_python(x_small, y_small, z_small))
result_py = compute_python(x_small, y_small, z_small)
ok_py = np.allclose(result_np[:n_small], result_py, rtol=1e-5, atol=1e-6)
print(f"Pure Python  : median {t_py_med:.4f}s  mean {t_py_mean:.4f}s ± {t_py_std:.4f}s  | match NumPy: {ok_py}")

# NumPy benchmark
t_np_med, t_np_mean, t_np_std = bench(lambda: compute_numpy(x, y, z))
print(f"NumPy        : median {t_np_med:.4f}s  mean {t_np_mean:.4f}s ± {t_np_std:.4f}s")

# NumExpr benchmark
if NUMEXPR_OK:
    t_ne_med, t_ne_mean, t_ne_std = bench(lambda: compute_numexpr(x, y, z))
    result_ne = compute_numexpr(x, y, z)
    ok_ne = np.allclose(result_np, result_ne, rtol=1e-5, atol=1e-6)
    print(f"NumExpr      : median {t_ne_med:.4f}s  mean {t_ne_mean:.4f}s ± {t_ne_std:.4f}s  | match NumPy: {ok_ne}")
else:
    print("NumExpr not available; skip.")

# JAX benchmark
if JAX_OK:
    # first compile (JIT) + warmup
    _ = compute_jax(x_jax, y_jax, z_jax).block_until_ready()
    t_jax_med, t_jax_mean, t_jax_std = bench(lambda: compute_jax(x_jax, y_jax, z_jax))
    result_jax = np.array(compute_jax(x_jax, y_jax, z_jax))
    ok_jax = np.allclose(result_np, result_jax, rtol=1e-5, atol=1e-6)
    print(f"JAX (jit)    : median {t_jax_med:.4f}s  mean {t_jax_mean:.4f}s ± {t_jax_std:.4f}s  | match NumPy: {ok_jax}")
else:
    print("JAX not available; skip.")

# Numba benchmark
if NUMBA_OK:
    # first compile (JIT) + warmup
    _ = compute_numba(x, y, z)
    t_nb_med, t_nb_mean, t_nb_std = bench(lambda: compute_numba(x, y, z))
    result_nb = compute_numba(x, y, z)
    ok_nb = np.allclose(result_np, result_nb, rtol=1e-5, atol=1e-6)
    print(f"Numba (jit)  : median {t_nb_med:.4f}s  mean {t_nb_mean:.4f}s ± {t_nb_std:.4f}s  | match NumPy: {ok_nb}")
else:
    print("Numba not available; skip.")

# f2py benchmark
if F2PY_OK:
    t_f2py_med, t_f2py_mean, t_f2py_std = bench(lambda: compute_f2py(x, y, z))
    result_f2py = compute_f2py(x, y, z)
    ok_f2py = np.allclose(result_np, result_f2py, rtol=1e-5, atol=1e-6)
    print(f"f2py (fortran): median {t_f2py_med:.4f}s  mean {t_f2py_mean:.4f}s ± {t_f2py_std:.4f}s  | match NumPy: {ok_f2py}")
else:
    print("f2py not available; skip.")

# Quick speedup summary (if all present)
def fmt(x): return f"{x:.1f}x"
print("\\n=== Speedup Summary ===")
# Scale Python time to full array for comparison
t_py_scaled = t_py_med * (n / n_small)
print(f"Pure Python (scaled)   : ~{t_py_scaled:.1f}s (estimated for full array)")
print(f"Speedup NumPy / Python : {fmt(t_py_scaled / t_np_med)}")
if NUMEXPR_OK:
    print(f"Speedup NumExpr / NumPy: {fmt(t_np_med / t_ne_med)}")
if JAX_OK:
    print(f"Speedup JAX / NumPy    : {fmt(t_np_med / t_jax_med)}")
if NUMBA_OK:
    print(f"Speedup Numba / NumPy  : {fmt(t_np_med / t_nb_med)}")
if F2PY_OK:
    print(f"Speedup f2py / NumPy   : {fmt(t_np_med / t_f2py_med)}")