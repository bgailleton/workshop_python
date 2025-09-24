# numba

Deep dive into Numba: from basics to geomorph modelling.

## Notebooks
- `numba_cheatsheet.ipynb` – Decorator overview, templated kernels, typed containers, and diagnostics.
- `numba_speedups.ipynb` – Timing harness comparing pure Python vs. Numba on multiple workloads.
- `numba_stream_power_model.ipynb` – Implicit stream-power solver with D8 routing and uplift.
- `numba_advanced_features.ipynb` – `jitclass`, typed `List`/`Dict`, callbacks, RNG, and solver patterns.

## Run Tips
Most notebooks auto-install dependencies. For the stream-power model, ensure `numpy>=1.23` and give it a few seconds for the initial JIT compile.
