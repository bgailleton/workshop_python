# numba

Deep dive into Numba: from basics to geomorph modelling.

## Notebooks
- `numba_cheatsheet.ipynb` – Decorator overview, templated kernels, typed containers, and diagnostics. A one-stop refresher that pairs narrative guidance with executable cells so you can copy/paste patterns directly into production code.
- `numba_speedups.ipynb` – Timing harness comparing pure Python vs. Numba on multiple workloads. Each section explains why the optimisation works (loop fusion, prange, ufuncs) before charting the real-world speedup.
- `numba_stream_power_model.ipynb` – Implicit stream-power solver with D8 routing and uplift. Walks through receivers, drainage accumulation, Newton updates, and then visualises relief change and drainage structure.
- `numba_advanced_features.ipynb` – `jitclass`, typed `List`/`Dict`, callbacks, RNG, and solver patterns. Shows how to structure larger projects with Numba-friendly classes and interoperability tricks.

## Run Tips
Most notebooks auto-install dependencies. For the stream-power model, ensure `numpy>=1.23` and give it a few seconds for the initial JIT compile.
