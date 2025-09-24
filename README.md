# Python Workshop Playbook

Hands-on notebooks that showcase practical pathways for moving from MATLAB-style workflows to modern Python tooling. Each folder is paced for a short workshop segment: punchy intros, focused demos, and persuasive visuals to help teams adopt the stack quickly.

## Folder Index
- **basic/** – Concept primers for scientific Python, including motivation to switch from MATLAB and how to layer abstraction cleanly.
- **data_heavy/** – High-volume data handling with HDF5, NetCDF, Zarr, and Dask for scalable IO and lazy computation.
- **geomorpho/** – Geomorphology workflows; currently a flood-routing example bridging PyTopoToolbox concepts.
- **geospatial/** – Raster/vector GIS essentials with Rasterio and GeoPandas-style tooling.
- **HPC_cpu_jit/** – CPU acceleration tour featuring Numba, NumExpr, JAX, F2PY, and profiling-ready scripts.
- **Machine_Learning/** – Compact ML labs: titanic with CatBoost, digit recognition, anomaly detection, and number theory puzzles.
- **numba/** – Fresh notebooks on Numba fundamentals, performance wins, advanced features, and a full implicit stream-power model.
- **pandas_polars/** – Tabular manipulation contrasts using pandas and Polars, backed by a ready-made synthetic dataset.
- **taichi_lang/** – GPU-ready numerical demos with Taichi, including heat diffusion and hydraulic erosion sims.
- **visualisation/** – Plotting gallery covering Matplotlib, Seaborn, and Datashader + HoloViews.

## Getting Started
1. Create or activate a conda/mamba environment with Python 3.11+.
2. Install notebook dependencies as you go (most notebooks include a `!pip install ...` cell for convenience).
3. Launch JupyterLab or VS Code notebooks in this repository root so relative paths resolve.

## Suggested Flow
- Start with `basic/` and `pandas_polars/` to get comfortable with the ecosystem.
- Pick domain-specific tracks: geospatial, geomorphology, visualisation.
- Finish with acceleration stories in `numba/` and `taichi_lang/` to drive home performance.

## Contributing
Keep notebooks concise and interactive. When adding content, include a short README entry (see folder READMEs for the tone) and prefer self-contained installs so newcomers can run cells top-to-bottom without extra setup.
