.. _model_build:

Building a model
================

To build a complete model from scratch using available data the ``build`` method can be used.
The build method is identical for all :ref:`HydroMT model plugins <plugins>`,
but the model methods (i.e. sections and options in the .yaml configuration file) are different for each model.

**Steps in brief:**

1) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
2) Prepare a **model workflow** which describes the complete pipeline to build your model: see :ref:`model workflow <model_workflow>`.
   This workflow file will also contain the **region** definition for your model (e.g. bounding box, point coordinates, polygon etc.):
   see :ref:`region definition <region>`.
3) **Build** you model using the CLI or Python interface


The ``hydromt build`` method can be run from the command line or Python after the right conda environment is activated.
The HydroMT core package contain implementation for generalized model classes. Specific model implementation for softwares have to be built
from associated :ref:`HydroMT plugin <plugins>` that needs to be installed to your Python environment.

If you work from command line, you can check which models are available by running:

.. code-block:: console

    $ hydromt --models

Here is how to build a model from:

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        For the hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog:

        .. code-block:: console

            $ hydromt build example_model "./path/to/example_model" -d "artifact_data" -i "./path/to/build_options.yaml" -v

        For HydroMT Wflow SBM plugin ``wflow_sbm`` with a user-defined data catalog:

        .. code-block:: console

            $ hydromt build wflow_sbm "./path/to/wflow_model" -d "./path/to/data_catalog.yml" -i "./path/to/build_options.yaml" -v

    .. tab-item:: Python API

        For the hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog:

        .. code-block:: python

            from hydromt import ExampleModel
            from hydromt.readers import read_workflow_yaml

            # Instantiate model
            model = ExampleModel(
                root="./path/to/example_model",
                data_catalog=["artifact_data"],
            )
            # Read build options from yaml
            _, _, build_options = read_workflow_yaml(
                "./path/to/build_options.yaml"
            )
            # Build model
            model.build(steps=build_options)

        For HydroMT Wflow SBM plugin ``wflow_sbm`` with a user-defined data catalog:

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel
            from hydromt.readers import read_workflow_yaml

            # Instantiate model
            model = WflowSbmModel(
                root="./path/to/wflow_model",
                data_catalog=["./path/to/data_catalog.yml"],
            )
            # Read build options from yaml
            _, _, build_options = read_workflow_yaml(
                "./path/to/build_options.yaml"
            )
            # Build model
            model.build(steps=build_options)

        Additionally, in Python, you can also build the model step-by step by calling each of the
        model steps as methods instead of using a workflow file. For example:

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
