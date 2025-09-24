# HPC_cpu_jit

CPU acceleration grab-bag to contrast JIT, vectorisation, and compiled extensions.

## Contents
- `basic_examples_speedup.ipynb` – Baseline timing comparisons for pure Python vs. Numba/NumExpr.
- `jax_demo.ipynb` – JAX autotdiff + JIT on CPU for array workloads.
- `numexpr_demo.ipynb` – Broadcasting-heavy expressions optimised with NumExpr.
- `steepest_descent.py` & `basic_examples_speedup.py` – Script versions ready for command-line timing.
- `f2py_*.so` – Pre-built extension modules used by the notebooks.

## Run Tips
Stay in the repo root when launching notebooks so shared `.so` files resolve. Enable `NUMBA_NUM_THREADS` to experiment with parallelism.
