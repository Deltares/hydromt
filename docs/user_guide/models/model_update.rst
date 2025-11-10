.. _model_update:

Updating a model
================

To add or change one or more components of an existing model the ``update`` method can be used.
The update method works identical for all :ref:`HydroMT model plugins <plugins>`,
but the model methods (i.e. sections and options in the :ref:`.yaml workflow file <model_workflow>`) are different for each model.

**Steps in brief:**

1) You have an **existing model** schematization. This model does not have to be complete.
2) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
3) Prepare a **model workflow** with the methods that you want to use to add or change components of your model: see :ref:`model workflow <model_workflow>`.
4) **Update** your model using the CLI or Python interface

The ``hydromt update`` method can be run from the command line or Python after the right conda
environment is activated. By default, the model is updated in place, overwriting the existing
model schematization. To save a copy of the model provide a new output model root directory (using
the ``-o`` option when working from command line).
All model methods in the .yaml configuration file will be updated.


Here is how to update a model from:

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        For the hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog
        and saving the updated model in place:

        .. code-block:: console

            $ hydromt update example_model "./path/to/example_model_to_update" -d "artifact_data" -i "./path/to/update_options.yaml" -v

        For the hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog
        and saving the updated model to a new location:

        .. code-block:: console

            $ hydromt update example_model "./path/to/example_model_to_update" -o "./path/to/updated_example_model" -d "artifact_data" -i "./path/to/update_options.yaml" -v

        For HydroMT Wflow SBM plugin ``wflow_sbm`` with a user-defined data catalog:

        .. code-block:: console

            $ hydromt update wflow_sbm "./path/to/wflow_model" -d "./path/to/data_catalog.yml" -i "./path/to/update_options.yaml" -v

    .. tab-item:: Python API

        For the hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog
        and saving the updated model in place:

        .. code-block:: python

            from hydromt import ExampleModel
            from hydromt.readers import read_workflow_yaml

            # Instantiate model
            model = ExampleModel(
                root="./path/to/example_model_to_update",
                data_catalog=["artifact_data"],
                mode = "r+", # open model in read and write mode
            )
            # Read update options from yaml
            _, _, update_options = read_workflow_yaml(
                "./path/to/update_options.yaml"
            )
            # Update model
            model.update(steps=update_options)

        For hydromt core ``example_model`` plugin using the pre-defined ``artifact_data`` catalog
        and saving the updated model to a new location:

        .. code-block:: python

            from hydromt import ExampleModel
            from hydromt.readers import read_workflow_yaml

            # Instantiate model
            model = ExampleModel(
                root="./path/to/example_model_to_update",
                data_catalog=["artifact_data"],
                mode = "r+", # open model in read and write mode
            )
            # Read update options from yaml
            _, _, update_options = read_workflow_yaml(
                "./path/to/update_options.yaml"
            )
            # If you want to save the model in a different folder
            model.read()
            model.root.set("./path/to/updated_example_model", mode="w")
            # Update model
            model.update(steps=update_options)

        For HydroMT Wflow SBM plugin ``wflow_sbm`` with a user-defined data catalog:

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel
            from hydromt.readers import read_workflow_yaml

            # Instantiate model
            model = WflowSbmModel(
                root="./path/to/wflow_model",
                data_catalog=["./path/to/data_catalog.yml"],
                mode = "r+", # open model in read and write mode
            )
            # Read update options from yaml
            _, _, update_options = read_workflow_yaml(
                "./path/to/update_options.yaml"
            )
            # Update model
            model.update(steps=update_options)

        Additionally, in Python, you can also update the model step-by step by calling each of the
        model steps as methods instead of using a workflow file. For example:

        .. code-block:: python

            from hydromt import ExampleModel

            # Instantiate model
            model = ExampleModel(
                root="./path/to/example_model_to_update",
                data_catalog=["./path/to/data_catalog.yml"],
                mode = "r+", # open model in read and write mode
            )
            # If you want to save the model in a different folder
            model.read()
            model.root.set("./path/to/updated_example_model", mode="w")
            # Update model step by step
            # Step 1: update the config with new values
            model.config.update(
                data = {'starttime': '2010-01-01', 'endtime': '2020-12-31'}
            )
            # Step 2: add landuse data in the model grid
            model.grid.add_data_from_rasterdataset(
                raster_fn="vito",
                reproject_method="mode",
                rename={"vito": "landuse"},
            )
            # Write the updated model to disk
            model.write()
