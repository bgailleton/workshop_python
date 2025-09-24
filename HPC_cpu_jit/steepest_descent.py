# Steepest-slope 8-neighbor benchmark: NumPy vs Numba vs Taichi
# -------------------------------------------------------------
# Slope definition:
#   S[i,j] = max_k (Z[i,j] - Z[i+di_k, j+dj_k]) / (dx * dist_k)
# where k enumerates the 8 neighbors, dist_k = 1 for N,E,S,W and sqrt(2) for diagonals.
# Result is computed for interior cells only: shape (n-2, m-2).

import os, time, math
import numpy as np

# Optional: stabilize CPU timings by pinning threads (comment out if you want full speed)
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

try:
    import taichi as ti
    TAI_CHI_OK = True
except Exception:
    TAI_CHI_OK = False

try:
    import f2py_steepest
    F2PY_OK = True
except Exception:
    F2PY_OK = False

# -----------------------
# Problem setup
# -----------------------
np.random.seed(0)
n, m = 4096, 4096           # adjust size if you lack RAM/VRAM
dx = 1.0                    # cell size (same in x and y)
Z = (np.random.rand(n, m).astype(np.float32) * 100.0)  # synthetic DEM

# Neighbors (8-connectivity)
offs_i = np.array([-1,-1,-1, 0,0, 1, 1, 1], dtype=np.int32)
offs_j = np.array([-1, 0, 1,-1,1,-1, 0, 1], dtype=np.int32)
dists  = np.array([math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)], dtype=np.float32)

# -----------------------
# Pure Python (explicit loops)
# -----------------------
def steepest_python(Z: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Pure Python nested loops - slowest but most readable"""
    n, m = Z.shape
    out = np.empty((n-2, m-2), dtype=np.float32)
    for i in range(1, n-1):
        for j in range(1, m-1):
            zc = Z[i, j]
            best = -1e30
            for k in range(8):
                val = (zc - Z[i+offs_i[k], j+offs_j[k]]) / (dx * dists[k])
                if val > best:
                    best = val
            out[i-1, j-1] = best
    return out

# -----------------------
# NumPy (vectorized)
# -----------------------
def steepest_numpy(Z: np.ndarray, dx: float = 1.0) -> np.ndarray:
    Zi = Z[1:-1, 1:-1]
    n2, m2 = Zi.shape
    candidates = []
    for di, dj, d in zip(offs_i, offs_j, dists):
        nb = Z[1+di:n-1+di, 1+dj:m-1+dj]
        candidates.append((Zi - nb) / (dx * d))
    return np.maximum.reduce(candidates)  # shape (n-2, m-2)

# -----------------------
# Numba (parallel loops)
# -----------------------
if NUMBA_OK:
    @njit(parallel=True, fastmath=True)
    def steepest_numba(Z: np.ndarray, dx: float) -> np.ndarray:
        n, m = Z.shape
        out = np.empty((n-2, m-2), dtype=np.float32)
        # constants for fast access
        oi = np.array([-1,-1,-1, 0,0, 1, 1, 1], dtype=np.int32)
        oj = np.array([-1, 0, 1,-1,1,-1, 0, 1], dtype=np.int32)
        dd = np.array([math.sqrt(2.0),1.0,math.sqrt(2.0),1.0,1.0,math.sqrt(2.0),1.0,math.sqrt(2.0)], dtype=np.float32)

        # Loop manually through all pixels
        for i in prange(1, n-1):
            for j in range(1, m-1):
                zc = Z[i, j]
                best = -1e30
                for k in range(8):
                    val = (zc - Z[i+oi[k], j+oj[k]]) / (dx * dd[k])
                    if val > best:
                        best = val
                out[i-1, j-1] = best
        return out

# -----------------------
# Taichi (GPU/CPU kernel)
# -----------------------
if TAI_CHI_OK:
    # Try GPU, otherwise CPU
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

    Z_ti  = ti.field(dtype=ti.f32, shape=(n, m))
    S_ti  = ti.field(dtype=ti.f32, shape=(n-2, m-2))
    oi_t  = [-1,-1,-1, 0,0, 1, 1, 1]
    oj_t  = [-1, 0, 1,-1,1,-1, 0, 1]
    dd_t  = [math.sqrt(2.0),1.0,math.sqrt(2.0),1.0,1.0,math.sqrt(2.0),1.0,math.sqrt(2.0)]

    @ti.kernel
    def steepest_taichi(dx: ti.f32):
        for I, J in S_ti:  # I in [0, n-2), J in [0, m-2)
            i = I + 1
            j = J + 1
            zc = Z_ti[i, j]
            best = ti.cast(-1e30, ti.f32)
            for k in ti.static(range(8)):
                val = (zc - Z_ti[i + oi_t[k], j + oj_t[k]]) / (dx * dd_t[k])
                if val > best:
                    best = val
            S_ti[I, J] = best

# -----------------------
# f2py (Fortran compiled)
# -----------------------
def setup_f2py():
    """Create and compile Fortran module if needed"""
    fortran_code = """
subroutine steepest_fortran(z, dx, out, n, m)
    implicit none
    integer, intent(in) :: n, m
    real*4, intent(in) :: z(n, m), dx
    real*4, intent(out) :: out(n-2, m-2)
    integer :: i, j, k
    real*4 :: zc, val, best
    integer :: oi(8), oj(8)
    real*4 :: dd(8)

    ! 8-neighbor offsets
    oi = (/ -1, -1, -1,  0,  0,  1,  1,  1 /)
    oj = (/ -1,  0,  1, -1,  1, -1,  0,  1 /)
    dd = (/ 1.4142136, 1.0, 1.4142136, 1.0, 1.0, 1.4142136, 1.0, 1.4142136 /)

    do j = 2, m-1
        do i = 2, n-1
            zc = z(i, j)
            best = -1e30
            do k = 1, 8
                val = (zc - z(i + oi(k), j + oj(k))) / (dx * dd(k))
                if (val > best) then
                    best = val
                end if
            end do
            out(i-1, j-1) = best
        end do
    end do
end subroutine steepest_fortran
"""

    # Write Fortran file
    with open('f2py_steepest.f90', 'w') as f:
        f.write(fortran_code)

    # Compile with f2py
    print("Compiling Fortran module with f2py...")
    cmd = "python -m numpy.f2py -c -m f2py_steepest f2py_steepest.f90"
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
            import f2py_steepest
            F2PY_OK = True
    except Exception as e:
        print(f"f2py setup failed: {e}")
        F2PY_OK = False

if F2PY_OK:
    def steepest_f2py(Z: np.ndarray, dx: float = 1.0) -> np.ndarray:
        return f2py_steepest.steepest_fortran(Z, dx, Z.shape[0], Z.shape[1])

# -----------------------
# Benchmark helpers
# -----------------------
def bench(fn, warmups=2, runs=5):
    # warmups
    for _ in range(warmups):
        _ = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), np.mean(times), np.std(times)

print("=== Computing reference (NumPy) and benchmarking ===")
S_np = steepest_numpy(Z, dx=dx)  # reference result

# Pure Python benchmark (on smaller subset for reasonable timing)
n_small, m_small = min(512, n), min(512, m)  # Use smaller array for Python loops
Z_small = Z[:n_small, :m_small].copy()
print(f"Pure Python tested on {n_small}x{m_small} array (would be too slow on full {n}x{m})")
t_py_med, t_py_mean, t_py_std = bench(lambda: steepest_python(Z_small, dx))
S_py = steepest_python(Z_small, dx)
ok_py = np.allclose(S_np[:n_small-2, :m_small-2], S_py, rtol=1e-5, atol=1e-6)
print(f"Pure Python  : median {t_py_med:.4f}s  mean {t_py_mean:.4f}s ± {t_py_std:.4f}s  | match NumPy: {ok_py}")

# NumPy benchmark
t_np_med, t_np_mean, t_np_std = bench(lambda: steepest_numpy(Z, dx))
print(f"NumPy        : median {t_np_med:.4f}s  mean {t_np_mean:.4f}s ± {t_np_std:.4f}s")

# Numba benchmark
if NUMBA_OK:
    # first compile (JIT) + warmup
    _ = steepest_numba(Z, dx)
    t_nb_med, t_nb_mean, t_nb_std = bench(lambda: steepest_numba(Z, dx))
    S_nb = steepest_numba(Z, dx)
    ok_nb = np.allclose(S_np, S_nb, rtol=1e-5, atol=1e-6)
    print(f"Numba (jit)  : median {t_nb_med:.4f}s  mean {t_nb_mean:.4f}s ± {t_nb_std:.4f}s  | match NumPy: {ok_nb}")
else:
    print("Numba not available; skip.")

# Taichi benchmark
if TAI_CHI_OK:
    # host->device once
    Z_ti.from_numpy(Z)
    # one compile + warmup
    steepest_taichi(dx)
    ti.sync()
    def run_taichi():
        steepest_taichi(dx)
        ti.sync()  # ensure timing covers kernel execution
        return None
    t_ti_med, t_ti_mean, t_ti_std = bench(run_taichi)
    S_ti_np = S_ti.to_numpy()
    ok_ti = np.allclose(S_np, S_ti_np, rtol=1e-5, atol=1e-6)
    print(f"Taichi kernel: median {t_ti_med:.4f}s  mean {t_ti_mean:.4f}s ± {t_ti_std:.4f}s  | match NumPy: {ok_ti}")
else:
    print("Taichi not available; skip.")

# f2py benchmark
if F2PY_OK:
    t_f2py_med, t_f2py_mean, t_f2py_std = bench(lambda: steepest_f2py(Z, dx))
    S_f2py = steepest_f2py(Z, dx)
    ok_f2py = np.allclose(S_np, S_f2py, rtol=1e-5, atol=1e-6)
    print(f"f2py (fortran): median {t_f2py_med:.4f}s  mean {t_f2py_mean:.4f}s ± {t_f2py_std:.4f}s  | match NumPy: {ok_f2py}")
else:
    print("f2py not available; skip.")

# Quick speedup summary (if all present)
def fmt(x): return f"{x:.1f}x"
# Scale Python time to full array for comparison
pixels_small = (n_small - 2) * (m_small - 2)
pixels_full = (n - 2) * (m - 2)
t_py_scaled = t_py_med * (pixels_full / pixels_small)
print(f"Pure Python (scaled)   : ~{t_py_scaled:.1f}s (estimated for full array)")
print(f"Speedup NumPy / Python : {fmt(t_py_scaled / t_np_med)}")
if NUMBA_OK:
    print(f"Speedup Numba / NumPy : {fmt(t_np_med / t_nb_med)}")
if TAI_CHI_OK:
    print(f"Speedup Taichi / NumPy: {fmt(t_np_med / t_ti_med)}")
if F2PY_OK:
    print(f"Speedup f2py / NumPy  : {fmt(t_np_med / t_f2py_med)}")


# @numba.njit(parallel=False, fastmath=True)
# def flow_accumulation(stack, receivers, width, height, nodata_value):

#     # Most of Numpy is accepted into numba
#     Accumulation = np.zeros_like(stack)

#     # iterating through all the pixels (nodes) ordered from upstream to downstream
#     for node in stack:

#         # Checking if the pixel is active
#         if node == nodata_value:
#             continue

#         # Local source 
#         Accumulation[node] += 1

#         # Checking the downstream node of node
#         this_receiver = receivers[node]

#         # Propagating
#         Accumulation[this_receiver] += Accumulation[node]

#     # Return 2D array
#     return Accumulation.reshape(width,height)