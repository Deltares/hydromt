.. _model_workflow:

Creating a Model Workflow
=========================

A model workflow (the ``.yaml`` file that tells hydromt what to do and in which orders) consists of three
main sections:

1. ``modeltype``
2. ``global``
3. ``steps``

The ``modeltype`` tells hydromt which model to use. In the case of using hydromt core,
``model`` and ``example_model`` are the only options but if you have plugins installed those will probably
provide other options as well (e.g ``sfincs``, ``wflow_sbm``, etc.). You can discover
which options you have installed with the CLI ``hydromt --models``.

The ``global`` is where you provide any configuration that the model will need at
initialization. This is where you for example, can list which data catalogs to use
(if you do not want to specify it in your CLI or python script), or other options
depending on the plugin you are using (e.g. name of the configuration file for Wflow,
crs for Delft3D etc.).

For users that wish to use the core ``model`` class, this is also where you
define which components the model should have, if they are spatial components and which
components the model should use to define its region. If you are using a plugin,
the plugin will have done this for you and you do not need to define the components.
To know more about defining components, please check the :ref:`model components <model_components>` page.

Generally if you use a plugin, the ``global`` part may look something like this:

.. code-block:: yaml

    modeltype: wflow_sbm
    global:
        data_libs:
            - artifact_data
            - local_data.yml
        config_filename: wflow_sbm.toml
        ...

Finally there is the ``steps`` part of the workflow. This should be a list, where each
list item should be a name of a function you want to run, followed by any arguments you
want to pass to that function. You can use the ``.`` syntax to call functions on
components, or omit this if the function you want to call is defined on the model.

For example, using the core ``example_model`` plugin:

.. code-block:: yaml

    steps:
        - config.update: # update lines in the model config file
            data:
              'starttime': '2000-01-01'
              'endtime': '2010-12-31'
        - grid.create_from_region: # create the model grid from a region
            region:
              subbasin: [12.2051, 45.8331]
              uparea: 50
            res: 1000
            crs: utm
            hydrography_fn: merit_hydro_1k
            basin_index_fn: merit_hydro_index
        - grid.add_data_from_rasterdataset: # add DEM data to the model grid
            raster_fn: merit_hydro_1k
            variables: elevtn
            fill_method: null
            reproject_method: bilinear
            rename:
              elevtn: DEM

In the example above ``config`` and ``grid`` are the name of the components and
``update``, ``create_from_region``, and ``add_data_from_rasterdataset`` are the functions on it that you want to call.
Please check for each specific component which functions are available to call. This should be well
documented in the documentation of the plugin you are using.

In general, at the end of the steps, HydroMT will end with a last hidden step to ``write`` the whole
model. If you only wish to write specific components (e.g. when updating) or change some of the
write options, you can add a final step to call ``component.write`` on the components you wish to write.

.. code-block:: yaml

    steps:
        ...
        - grid.write: # write only the grid component to disk and change the filename
            filename: grid.nc
        - config.write: # write only the config component to disk

Finally, some plugins like Wflow SBM may have defined their methods at the model and not the component
level. In this case, you would not need to specify the component name before the method name.
For example it could be that a specific method actually updates several components at once
(e.g. ``setup_basemaps`` in Wflow SBM updates ``config``, ``staticmaps`` and ``geoms``).

Here is an example of a Wflow SBM workflow:

.. code-block:: yaml

    modeltype: wflow_sbm
    global:
        data_libs:
            - artifact_data
        config_filename: wflow_sbm.toml
    steps:
        - setup_basemaps: # create the model grid, basin mask and flow directions from a region
            region:
              subbasin: [12.2051, 45.8331]
              uparea: 50
            res: 0.008333
            hydrography_fn: merit_hydro_ihu
            basin_index_fn: merit_hydro_index
        - setup_rivers: # add river network
            hydrography_fn: merit_hydro_ihu
            river_geom_fn: hydro_rivers_lin
            river_upa: 30
        - staticmaps.write:
        - geoms.write:

As mentioned in :ref:`building <model_build>` and :ref:`updating <model_update>` a model
pages, if you are using hydromt in Python, you can also use the worfklow file and the ``build``
and ``update`` methods, or just prepare your model step by step by calling the methods directly.

For example:

.. code-block:: python

        from hydromt import ExampleModel

        # Instantiate model
        model = ExampleModel(
            root="./path/to/example_model",
            data_catalog=["artifact_data"],
        )
        # Build model step by step
        # Step 1: populate the config with some values
        model.config.update(
            data = {'starttime': '2000-01-01', 'endtime': '2010-12-31'}
        )
        # Step 2: define the model grid
        model.grid.create_from_region(
            region={"subbasin": [12.2051, 45.8331], "uparea": 50},
            res=1000,
            crs="utm",
            hydrography_fn="merit_hydro_1k",
            basin_index_fn="merit_hydro_index",
        )
        # Step 3: add DEM data to the model grid
        model.grid.add_data_from_rasterdataset(
            raster_fn="merit_hydro_1k",
            variables="elevtn",
            fill_method=None,
            reproject_method="bilinear",
            rename={"elevtn": "DEM"},
        )
        # Write the model to disk
        model.write()
