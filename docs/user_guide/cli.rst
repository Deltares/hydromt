.. _cli:

Interacting with a model
========================
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

.. code-block:: console

    $ hydromt build --help
    Usage: hydromt build [wflow|sfincs|delwaq|wflow_sediment] MODEL_ROOT REGION [OPTIONS]

    Build models from source data.

    Example usage:
    --------------

    To build a wflow model for a subbasin using and point coordinates snapped to cells with stream order >= 4
    hydromt build wflow /path/to/model_root "{'subbasin': [-7.24, 62.09], 'strord': 4}" -i /path/to/wflow_config.ini

    To build a base wflow model based on basin ID
    hydromt build wflow /path/to/model_root "{'basin': 230001006}"

    To build a sfincs model based on a bbox (for Texel)
    hydromt build sfincs /path/to/model_root "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -i /path/to/sfincs_config.ini

    Options:
    -r, --res FLOAT             Model resolution, if not provided the model
                                default is selected. Note that the unit may different
                                depending on the model. {'wflow':
                                0.008333333333333333, 'sfincs': 100}.

    --opt TEXT                  Component specific keyword arguments, see the
                                setup_<component> method of the specific model
                                for more information about the arguments.

    -i, --config PATH           Path to hydroMT configuration file.
    -q, --quiet                 Decrease verbosity.
    -v, --verbose               Increase verbosity.
    --help                      Show this message and exit.

.. _cli_update:

update
-------

The update method can be used to update one or several model components. The model components are 
identical to the headers in the ini-files of each model. Options for a component 
can be set in the ini-file or provides via command line with the `c` and `opt` options if only one component 
is updated. For several, use the configuration file with the `i` option.

.. code-block:: console

    $ hydromt update --help
    Usage: hydromt update [wflow|sfincs|delwaq|wflow_sediment] MODEL_ROOT [OPTIONS]

    Update one or several specific component of a model.  Set an output directory to copy
    the edited model to a new folder, otherwise maps are overwritten.

    Example usage:
    --------------

    Update (overwrite) landuse-landcover maps in a wflow model
    hydromt update wflow /path/to/model_root -c setup_lulcmaps --opt source_name=vito

    Update reservoir maps based on default settings in a wflow model and write to new directory
    hydromt update wflow /path/to/model_root -o /path/to/model_out -c setup_reservoirs -i path/to/wflow_config.ini

    Options:
    -c, --components COMPONENT Model component to update
    -o, --model-out DIRECTORY  Output model folder. Maps in MODEL_ROOT are
                                overwritten if left empty.

    --opt TEXT                 Component specific keyword arguments, see the
                                setup_<component> method of the specific model
                                for more information about the arguments.

    -i, --config PATH          Path to hydroMT configuration file.
    -q, --quiet                Decrease verbosity.
    -v, --verbose              Increase verbosity.
    --help                     Show this message and exit.

.. _cli_clip:

clip
----

The clip method allows to clip a subregion from a model, including all static maps,
static geometries and forcing data. This method has only been implemented for the 
wflow model so far. The :ref:`region <cli_region>` argument follows the same syntax as 
in the build method.

.. code-block:: console

    $ hydromt clip --help
    Usage: hydromt clip [wflow] MODEL_ROOT MODEL_DESTINATION REGION [OPTIONS]

    Create a new model based on clipped region of an existing model. If the
    existing model contains forcing, they will also be clipped to the new
    model.

    For options to build wflow models see:

    Example usage to clip a wflow model for a subbasin derived from point coordinates
    snapped to cells with stream order >= 4
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'subbasin': [-7.24, 62.09], 'wflow_streamorder': 4}"

    Example usage basin based on ID from model_root basins map
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'basin': 1}"

    Example usage basins whose outlets are inside a geometry
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'outlet': 'geometry.geojson'}"

    All available option in the clip_staticmaps function help.

    Options:
    -q, --quiet    Decrease verbosity.
    -v, --verbose  Increase verbosity.
    --help         Show this message and exit.


.. _cli_region:

Region options
--------------

.. image:: ../img/region.png

The region of interest can be described in two ways:

- based on **coordinates** using a combination of five options (*bbox*, *geom*, *outlet*, *basin*, *subbasin*) with four 
  types of values (point coordinates, bounding box, geometry and ID). 
- based on the region of **another model** for models built on top of others
  (example water quality models on top of hydrology or hydraulic).

The bounding box *bbox* and geometry *geom* options in combination are 
used to desribe the exact outline of the region of interest. 

To use the (sub)basin(s) shape as model boundary, several options are available.
A bounding box or geometry can be set to the *basin*, *outlet* or *subbasin* options to 
find all river (sub)basins or outlets/sinks within given area of interest. 
A unique basin ID or point coordinates can be set to *basin* option to lookup the 
outline of a entire basin.
Note that the basin ID is specific to each data source or model (e.g. wflow_subcatch).
Point coordinates can be also be set to the *subbasin* option to describe the outlet 
of a subbasin.

To use the model option to describe the region, the type and path of the existing 
model are provided (eg "wflow path/to/wflow_model").

The complete syntax for the region argument is given below.

For an exact outline of the region:

- ``{'bbox': [xmin, ymin, xmax, ymax]}``
- ``{'geom': '/path/to/geometry_file'}``

For basins/outlets intersecting with the region:

- ``{'basin': [xmin, ymin, xmax, ymax]}``
- ``{'basin': '/path/to/geometry_file'}``
- ``{'outlet': [xmin, ymin, xmax, ymax]}``
- ``{'outlet': '/path/to/geometry_file'}``

For basins with ID or point coordinates:

- ``{'basin': ID}``
- ``{'basin': [ID1, ID2, ..]}``
- ``{'basin': [x, y]}``
- ``{'basin': [[x1, x2, ..], [y1, y2, ..]]}``

For subbasins upstream from a point location or inside a boundig box or geometry. 
An optional minimum thresholdargument can be provided to snap the point locations to 
(larger) streams or filter outlets from the bounding box or geomtry.  
The threshold can be set to any variable in the dataset or model, e.g.: uparea: 30. 
To delineate subbasins based on point coordinates within large basins the process 
is speed up by providing an additional bounding bbox argument. 
A warning will be raised if some upstream cell are located outside the bounding box. 

- ``{'subbasin': [x, y]}``
- ``{'subbasin': [x, y], 'variable': threshold}``
- ``{'subbasin': [x, y], 'variable': threshold, 'bbox': [xmin, ymin, xmax, ymax]}``
- ``{'subbasin': [[x1, x2, ..], [y1, y2, ..]], 'variable': threshold}``
- ``{'subbasin': [xmin, ymin, xmax, ymax], 'variable': threshold}``
- ``{'subbasin': '/path/to/geometry_file', 'variable': threshold}``

For a region based on another existing model supported by HydroMT:

- ``{'MODEL': path/to/model_root}``
