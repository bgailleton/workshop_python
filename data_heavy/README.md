# data_heavy

High-volume IO techniques for multidimensional earth-science data.

## Notebooks
- `hdf5_tutorial.ipynb` – H5Py workflows, chunking strategies, and metadata conventions.
- `netcdf_tutorial.ipynb` – NetCDF4/CDM recap plus xarray integration pointers.
- `zarr_tutorial.ipynb` – Cloud-friendly chunked stores with local and remote examples.
- `dask_tutorial.ipynb` – Parallel/lazy array pipelines built on top of the formats above.

## Run Tips
These notebooks assume moderate RAM; Dask examples stay within a single machine but demonstrate how to scale to clusters.
