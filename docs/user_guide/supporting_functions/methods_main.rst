.. _methods_processes:

Overview methods and processes
==============================

Methods and workflows are the engine of HydroMT. Methods provide the low-level functionality, only accessible through the Python interface,
to do the required processing of common data types such as grid and vector data. Workflows combine several methods to go from raw input
data to a model component. Examples of workflows include the delineation of hydrological basins (watersheds), conversion of landuse-landcover to model parameter maps, etc.


.. _xarray_accessors:

Xarray Accessors
----------------
Some powerful functionality that HydroMT uses is exposed in the ``gis`` module. In this
module `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ are located. These
allow for powerful new methods on top of xarray ``Dataset`` and ``DataArray`` classes.
There is the :ref:`raster API <raster_api>`, such as functionality to repoject,
resample, transform, interpolate nodata or zonal statistics. There is also the
:ref:`GeoDataset API <geodataset_api>` to work with geodataset data (N-dim
point/line/polygon geometry). For example, reprojecting, transform, update geometry or
convert to geopandas.GeoDataFrame to access further GIS methods.

.. warning::
    Remember to close any open datasets when you are finished working with them.
    Leaving datasets open may cause xarray to lock the files, which can prevent
    access by other processes until your Python session ends. For example, if you
    open a NetCDF file used by a model and do not close it before calling
    ``Model.update()``, HydroMT will raise a ``PermissionError`` when attempting
    to write to the model.

.. _flowpy_wrappers:

FlowPy Wrappers:
----------------

`PyFlwDir <https://deltares.github.io/pyflwdir/latest/index.html>` contains a series of
methods to work with gridded DEM and flow direction datasets. The ``gis.flw`` module
builds on top of this and provides hydrological methods for raster DEM data. For
example, calculate flow direction, flow accumulation, stream network, catchments, or
reproject hydrography.

.. _stats:

Stats:
------

The ``stats`` module has statistical methods including ``skills`` to compute skill
scores of models (e.g. NSE, KGE, bias and many more) and ``extremes`` to analyse extreme
events (extract peaks or compute return values).

.. _processes:

.. currentmodule:: hydromt.model

The ``model`` module has a ``processes`` submodule. This module contains some functions
to work with different kinds of model in- and ouput.

* :ref:`grid <workflows_grid_api>`: generic workflows to prepare regular gridded data.
    Used with the :class:`~grid.GridComponent`. For example to prepare regular grid data
    from constant, from RasterDataset (with or without reclassification) or from GeoDataFrame.
* :ref:`mesh <workflows_mesh_api>`: generic workflows to prepare unstructured mesh
    data. Used with :class:`~mesh.MeshComponent`. For example to create a mesh grid or
    prepare unstructured mesh data from RasterDataset.
* :ref:`basin_mask <workflows_basin_api>`: workflows to prepare a basin mask based on
    different region definitions (bounding box, point coordinates, polygon etc.)
* :ref:`rivers <workflows_rivers_api>`: workflows to prepare river profile data like
    width and depth.
* :ref:`temp <workflows_forcing_api>`: workflows to prepare meteorological forcing
    data. For example to prepare precipitation, temperature, or compute
    evapotranspiration data. Advanced downscaling methods are also available within
    these workflows.

You can find a couple of detailed examples of how to use HydroMT methods and workflows in Python:

* `Working with raster data <../_examples/working_with_raster.ipynb>`_
* `Working with flow direction data <../_examples/working_with_flow_directions.ipynb>`_
* `Define hydrological model regions <../_examples/delineate_basin.ipynb>`_
* `Extreme Value Analysis <../_examples/doing_extreme_value_analysis.ipynb>`_
