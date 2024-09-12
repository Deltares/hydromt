.. _custom_model_builder:

Custom Model Builders
=====================


Model class and components
--------------------------
The main goal of your plugin is to be able to build and update model instances for your own
software. As a reminder, the figure below illustrates the process of model building in HydroMT:

.. figure:: ../../_static/model_building_process.png
   :align: center

   Model building process in HydroMT.

In this section, we will detail the bottom part of the schematic above ie how to initialise your HydroMT Model class for your plugin, the main properties
and setup methods and how to work with its model components. The top part of the schematic (data preparation) is taken care of by the core of HydroMT.

Initialisation
^^^^^^^^^^^^^^
In the :ref:`create plugin section <plugin_create>`, we already saw which HydroMT ``Model`` class to choose for your plugin. Here we will focus on additional
properties and the initialisation of your new Model subclass.

To fully initialise your new subclass (eg *MyModelModel*), you need to initialise a couple of high level properties and in some cases, you may wish to modify
the default initialisation function of you parent class (the HydroMT core class you choose, ``Model`` or ``GridModel`` etc.).

.. TIP::

  In python, a child class (eg *MyModelModel(Model)*), inherits all methods and properties of the parent class (``Model``). If you wish to completely overwrite one
  of the parent methods (eg ``setup_region``), you can just redefine the function in the child class using the name (``setup_region``). You can also decide to
  re-use the parent function and for example add some extra steps, change the default values or the docstring of the function. For this, you can redefine the
  function in the child class, and within the redefined function, call the parent one use *super()* python class attribute (*super().setup_region*). You will see
  an example below for the initialisation function ``__init__`` .



Additional Model properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can have a look at the :ref:`Model API documentation <model_api>` to find out all other properties of the ``Model`` class and
the other subclasses. Apart from the components here are a couple of useful properties to use when developing your plugin:

- :py:attr:`~Model.root`: path to the root folder of the model instance you are working with.
- :py:attr:`~Model.crs`: the reference coordinate system (pyproj.CRS) of the model.
- :py:attr:`~Model.region`: the region (GeoDataFrame) of your model, stored in :py:attr:`~Model.geoms`.
- :py:attr:`~Model.data_catalog`: the current data catalog you can use to add data to your model.
- :py:attr:`~Model.logger`: the HydroMT logger object for your model instance.
- :py:attr:`~Model._read`: flag if the model is in read mode ('r' or 'r+' when initialising).
- :py:attr:`~Model._write`: flag if the model is in write mode ('w' or 'w+' when initialising).

Some submodel classes can have additional attributes based on their additional components, so check out the :ref:`API reference <model_api>`.

.. _plugin_setup:

Setup basic model objects
^^^^^^^^^^^^^^^^^^^^^^^^^
When building a model from scratch, the first thing you should do is take inventory of what your `setup_*` functions
(e.g. `setup_grid`, `setup_mesh`, etc.) need. They typically (though not always) take the region of interest as an
argument so that is usually a good place to start. This consists of the region in the world your model is located in,
if needed its CRS and its computational unit (grid, mesh, response_unit etc.). This is usually typically set by a first
base or region setup method which typically parses the region argument of the HydroMT CLI. The idea is that after this
function has been called, the user should already be able to have the minimum model properties or files in order to
be able to call HydroMT to ``update`` the model to add additional data (``build`` is not required anymore).

For example, this is what the ``setup_region`` from ``Model`` does by adding ``region`` to ``geoms``, or ``setup_grid``
from ``GridModel`` which generates a regular grid based on the region argument, a CRS and a resolution.
You can re-use the core methods or decide to define your own.

.. NOTE::

  **Order of the setup methods**: Typically, building a model starts with defining the computational units (grid, mesh, vector etc.).
  Afterwards data layers are added to model components and there might be dependencies between the different layers. For example,
  a method to define river dimensions should probably be called after a method which defines the river cells on the grid or mesh itself.
  However, there is no real check on the order in which setup methods are called apart from checks that you can build-in that certain
  layers are already present with clear error messages. Clear documentation will help your user too.
  For Command Line Interface users, the functions in the hydromt configuration  yaml file will be executed in the order they appear in the file.
  Python Interface users can call the setup functions in any order they want from a script.

Setup methods
^^^^^^^^^^^^^

In general, a HydroMT ``setup_<>`` method does 4 things:

  1. read and parse the data using the ``DataCatalog`` and corresponding ``DataAdapter.get_data`` method (
  ``get_rasterdataset`` for RasterDataset, ``get_GeoDataset`` for GeoDataset, ``get_geodataframe`` for GeoDataFrame and
  ``get_dataframe`` for DataFrame).

  2. process that data in some way, optionally by calling an external workflow function.
  3. Optionally, rename or update attributes from HydroMT variable conventions (name, unit) to the specific model conventions.
  4. add the data to the corresponding HydroMT model components.

Below is a simplified example of what a setup function would look like for a hypothetical landuse grid from a raster
input data using an external workflow from hydromt core is:

.. code-block:: python

  def setup_landuse(
        self,
        landuse: Union[str, Path, xr.DataArray],
    ):
        """Add landuse data variable to grid.

        Adds model layers:

        * **landuse_class** grid: data from landuse

        Parameters
        ----------
        landuse: str, Path, xr.DataArray
            Data catalog key, path to raster file or raster xarray data object.
            If a path to a raster file is provided it will be added
            to the data_catalog with its name based on the file basename without
            extension.
        """
        self.logger.info(f"Preparing landuse data from raster source {landuse}")
        # 1. Read landuse raster data
        da_landuse = self.data_catalog.get_rasterdataset(
            landuse,
            geom=self.region,
            buffer=2,
            variables=["landuse"],
        )
        # 2. Do some transformation or processing
        ds_out = hydromt.model.processes.grid.grid_from_rasterdataset(
            grid_like=self.grid,
            ds=da_landuse,
            fill_method="nearest",
            reproject_method="mode",
        )
        # 3. Rename or transform from HydroMT to model conventions
        rmdict = {"landuse": "landuse_class"}
        # Or using a properly initialised _GRIDS
        # rmdict = {k: v for k, v in self._GRIDS.items() if k in ds_out.data_vars}
        ds_out = ds_out.rename(rmdict)
        # 4. Add to grid
        self.set_grid(ds_out)

.. NOTE::

  **Input data type of the setup method**: Typically a setup function tries to go from one type of dataset
  (landuse raster) to a HydroMT model component (landuse map in ``maps``). So it's good to make clear for your user in
  the setup function docstrings which type of input data this function can work with. You could decide to support
  several data types in one setup function but be aware that the GIS processing functions like resampling, reprojection can
  be quite different for a raster or a vector for example. So you could decide to create two setup functions that
  prepare the same data but from different type of input data (eg *setup_landuse_from_raster* and *setup_landuse_from_vector*).



Workflows
^^^^^^^^^
Because the python script defining your plugin Model class can get quite long and it makes unit testing easier,
we encourage you to use or define external data processing functions. In HydroMT these are usually called workflows.
These workflows are usually stored in separate python scripts that you can decide to store in a workflow subfolder.

A couple of tips if you want to define workflows:

- check out the :ref: `workflows available` in HydroMT core
- avoid passing the HydroMT model class to your workflow function, but pass the required arguments directly (eg crs = self.crs, data = self.grid). Ideally the workflows work from common python objects like xarray or geopandas rather than with the ``Model`` class.
- if you want to do some GIS processing on ``RasterDataset`` or ``GeoDataset``, HydroMT defines a lot of useful methods. Check out the :ref: `Raster methods API doc` for RasterDataset and :ref: `GeoDataset methods API doc`. For ``GeoDataFrame``, the `geopandas <https://geopandas.org/en/stable/index.html>`_ library should have most of what you need (and for ``UgridDataset`` or mesh, the `xugrid <https://deltares.github.io/xugrid/>`_ library). For computing or deriving other variables from an input dataset, HydroMT contains also a couple of useful workflows for example ``flwdir`` for flow direction methods, ``basin_mask`` to derive basin shape, or ``stats`` to derive general, efficiency or extreme value statistics from data.



Adding a new property or component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you wish to add additional properties or attributes to your plugin Model subclass, you can either decide to add simple
attributes directly in the initialisation function ``__init__`` or define specific properties. Example for a shortcut to the basin
vector in the geoms object:

.. code-block:: python

  @property
  def basins(self) -> gpd.GeoDataFrame:
    """Returns a basin(s) geometry as a geopandas.GeoDataFrame."""
    # Check in geoms
    if "basins" in self.geoms:
      gdf = self.geoms["basins"]
    else:
        self.logger.warning(
            f"Basin map {self._MAPS['basins']} not found in maps or geoms."
        )
        gdf = gpd.GeoDataFrame()
    return gdf

In most cases, we hope that the components defined in HydroMT `Model` classes (``config``, ``geoms``, ``maps``, ``forcing``, ``states``,
``results``) and its generic subclasses (``grid``, ``mesh``, ``vector``) should allow you to store any data required by your
model in a proper way. If it is not the case, you can always define your own new model components by respecting the following steps
(example if your model has a lot of 2D non-geospatial tabular data that could nicely be stored as pandas.DataFrame objects, *tables*):

1. Initialise your new component placeholder in the ``__init__`` function, if possible with None.

.. code-block:: python

  self._tables = None

2. Define the component itself as a new property, that looks for the placeholder and tries reading if empty in read mode.

.. code-block:: python

  @property
  def tables(self) -> Dict[str, pd.DataFrame]:
      """Returns a dictionary of pandas.DataFrame tabular files."""
      if self._tables is None:
        self._tables = dict()
        if self._read:
          self.read_tables()
      return self._tables

3. Define a reading and a writing method for your new component.

.. code-block:: python

  def read_tables(self, **kwargs):
    """Read table files at <root> and parse to dict of dataframes"""
    if not self._write:
      self._tables = dict()  # start fresh in read-only mode

    self.logger.info("Reading model table files.")
    fns = glob.glob(join(self.root, f"*.csv"))
    if len(fns) > 0:
      for fn in fns:
        name = basename(fn).split(".")[0]
        tbl = pd.read_csv(fn)
        self.set_tables(tbl, name=name)

  def write_tables(self):
    """Write tables at <root>."""
    self._assert_write_mode
    if len(self.tables) > 0:
      self.logger.info("Writing table files.")
      for name in self.tables:
        write_path = join(self.root, f"{name}.csv")
        self.tables[name].to_csv(write_path, sep=",", index=False, header=True)

4. Define a set function in order to add or update data in your component.

.. code-block:: python

  def set_tables(self, df: pd.DataFrame, name:str):
    """Add table <pandas.DataFrame> to model."""
    if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
      raise ValueError("df type not recognized, should be pandas.DataFrame or pandas.Series.")
    if name in self._tables:
      if not self._write:
        raise IOError(f"Cannot overwrite table {name} in read-only mode")
      elif self._read:
        self.logger.warning(f"Overwriting table: {name}")
    self._tables[name] = df
