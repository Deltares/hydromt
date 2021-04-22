.. _cli:

Command Line Interface
======================
Configure workflow
------------------
The command line interface (CLI) provides a high-level interface to the different models and methods 
in HydroMT. Currently three methods are supported and explained below:

- **build** to prepare a model from scratch
- **update** to update specific component(s) of an existing model
- **clip** to extract a smaller sub-region of an existing model

To use the CLI, you need to open a command prompt (exemple Windows command prompt, Linux prompt, Anaconda prompt, .bat or .cmd files) and first 
activate the HydroMT python environment and then run the command line with the HydroMT method:

.. code-block:: console

    activate hydromt
    hydromt build --help

The choice of components to include when building or updating a model is defined in the :ref:`HydroMT configuration file <ini_options>`.
The different methods and their command line arguments are described below.

.. _cli_build:

build
-----

To build a complete model from scratch using available data the build command can be used. 
The interface is identical for each model, but the configuration file has different 
options (see documentation of the individual models). The mandatory :ref:`region <cli_region>` 
argument describes the region of interest. The build method will start by running the setup_basemaps 
component of the model in order to prepare the model schematization/grid based on the region arguments. 
The configuration file should then start by listing the setup_basemaps components and then all the components 
that the user wants to include during the build. If no specific configuration is provided, only the setup_basemaps 
component will be prepared.

.. include:: ../_generated/cli_build.rst

.. _cli_update:

update
-------

The update method can be used to update one or several model components. The model components are 
identical to the headers in the ini-files of each model. Options for a component 
can be set in the ini-file or provides via command line with the `c` and `opt` options if only one component 
is updated. For several, use the configuration file with the `i` option.

.. include:: ../_generated/cli_update.rst

.. _cli_clip:

clip
----

The clip method allows to clip a subregion from a model, including all static maps,
static geometries and forcing data. This method has only been implemented for the 
wflow model so far. The :ref:`region <cli_region>` argument follows the same syntax as 
in the build method.

.. include:: ../_generated/cli_clip.rst

Region options
--------------

.. _cli_region:


The region of interest can be described in three ways:

- based on **geospatial region** using *bbox*, *geom*:

    - ``{'bbox': [xmin, ymin, xmax, ymax]}``
    - ``{'geom': '/path/to/geometry_file'}``

- Based on a **hydrographic region** using *basin*, *subbasin*, *interbasin*:

  A *basin* is defined by the entire area which drain to the sea or a local depression.
  Users can supply a point location ``[x, y]`` or an area of interest based on a 
  bounding box ``[xmin, ymin, xmax, ymax]`` or geometry ``'/path/to/geometry_file'`` 
  for which the intersecting basin(s) are delineated. 
  Optionally, ``<variable>:<threshold>`` combinations (e.g. 'uparea':30) can be passed to 
  define streams to filter the basin selection. To filter the basins based on their 
  outlet location within the area of interest use `'outlets': true`.  
  
    - ``{'basin': [x, y]}``
    - ``{'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}``
    - ``{'basin': [xmin, ymin, xmax, ymax], 'outlets': true}``

  A *subbasin* is based on a user supplied outlet location and defined by the area
  that drains into that outlet. 
  Users can supply a the outlet point location ``[x, y]``, or an area of interest based
  on a bounding box ``[xmin, ymin, xmax, ymax]`` or geometry ``'/path/to/geometry_file'``
  from which all outlet locations are used and all upstream area is delineated. 
  Optionally, ``<variable>:<threshold>`` combinations (e.g. 'uparea':30) can be passed to define streams to 
  which the outlet point location are snapped or to filter outlets within the area of interest. 
  To speed up the delineation process users can supply an initial bounding box of the subbasin
  (This must be larger than the area of interest!) with ``'bounds': [xmin, ymin, xmax, ymax]`` 
  In case not all upstream cells are included a warning is raised.   
  
    - ``{'subbasin': [x, y], '<variable>': threshold}``
    - ``{'subbasin': [x, y], '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]}``
    - ``{'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]}``

  An *interbasin* is based on a user supplied area of interest and contains the most 
  downstream contiguous area that drains to any outflow of the area of interest. 
  Users can supply an area of interest based on a bounding box ``[xmin, ymin, xmax, ymax]`` 
  or geometry ``'/path/to/geometry_file'`` from which all outlet locations are used. 
  Optionally, ``<variable>:<threshold>`` (e.g. 'uparea':30) combinations can be passed to define streams 
  to filter outlets within the area of interest. To only use basin outlets 
  within the area of interest to derive interbasins use `'outlets': true`.
  
    - ``{'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}``
    - ``{'interbasin': [xmin, ymin, xmax, ymax], '<variable>': 'outets': true}``

- based on the grid of **another model**, e.g:

    - ``{'wflow': '/path/to/wflow/root'}``

For a detailed desription of the options see :py:meth:`~hydromt.workflows.basin_mask.parse_region`
and :py:meth:`~hydromt.workflows.basin_mask.get_basin_geometry`. 

.. image:: ../img/region.png