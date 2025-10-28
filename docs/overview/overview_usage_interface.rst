HydroMT interface
-----------------

HydroMT provides both a command line interface (CLI) and a Python application
programming interface (API) to build and update models. Here are examples of
how to use both interfaces to build a model from a configuration file.

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        .. code-block:: console

            $ hydromt build wflow_sbm "./path/to/wflow_model" -d "./path/to/data_catalog.yml" -i "./path/to/build_options.yaml" -v

    .. tab-item:: Python API

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel
            from hydromt.io import read_workflow_yaml

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


If you use the :ref:`command line interface <hydromt_cli>`, only a few high-level commands
are available to build and update models or export data from the data catalog. If you use
the :ref:`Python API <hydromt_python>`, you can also access the underlying methods of HydroMT
to read data from a catalog, perform GIS operations or write your own plugin.
