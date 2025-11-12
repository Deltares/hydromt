.. _custom_driver:

================
Custom Drivers
================

Drivers define **how to read and write data** once a URIResolver has determined which file(s) to read.

While resolvers locate and list datasets, drivers handle the actual I/O, parsing, and optional slicing.
For example:

- The same ``ConventionResolver`` could return file lists for **.nc**, **.zarr**, or **.tif** datasets.
- The same driver could be used with multiple resolvers to read datasets organized differently.

Custom drivers are often necessary when working with non-standard formats, proprietary files, or datasets
that require preprocessing before loading into HydroMT components.

Overview
--------

A **driver** is responsible for:

1. Reading raw data from URIs into Python objects (``xarray.Dataset``, ``GeoDataFrame``, ``numpy.ndarray``, etc.)
2. Optionally slicing, sub-setting, or transforming the data if the underlying python reading (e.g. xarray, pandas) function supports it.
3. Writing processed datasets back to disk or another storage backend

Drivers should **not** determine which files to read — that is the resolver's job — but may rely on additional arguments such
as ``time_range``, ``variables`` or ``mask`` to properly slice the dataset or select specific variables/columns.

Implementing a Driver
---------------------

At minimum, a driver must implement the ``read`` method:

.. code-block:: python

    from typing import List, Any
    from hydromt.data_catalog import DataDriver

    class MyRasterDriver(DataDriver):
        def read(
            self,
            uris: List[str],
            time_range: Optional[tuple[str, str]] = None,
            mask: Optional[gpd.GeoDataFrame] = None,
            variables: Optional[list[str]] = None,
            **kwargs,
        ) -> Any:
            """Read data from one or more URIs and return a standardized dataset."""
            # read and combine files as needed
            ...

Optionally, you can implement ``write`` to save data in the same or a different format. This
is useful for e.g. when using ``DataCatalog.export_data`` functionality.

.. code-block:: python

    def write(
        self,
        path: str,
        data: Any,
        **kwargs,
    ):
        """Write data to the specified path."""
        # save dataset in desired format
        ...

The ``read`` method should generally match the arguments used in your resolver to allow smooth integration with HydroMT's workflow system.

Example
-------

A minimal raster driver that reads a single **.tif** file using ``rasterio``:

.. code-block:: python

    import rasterio
    import xarray as xr

    class SimpleRasterDriver(DataDriver):
        def read(self, uris: List[str], **kwargs) -> xr.DataArray:
            path = uris[0]
            with rasterio.open(path) as src:
                data = src.read(1)
                coords = {"x": src.xy(0, 0)[0], "y": src.xy(0, 0)[1]}
            return xr.DataArray(data, coords=coords, dims=("y", "x"))
