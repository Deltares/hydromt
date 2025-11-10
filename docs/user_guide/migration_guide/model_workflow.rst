.. _model_workflow_migration:

Migrating the model workflow file
=================================

Overview
--------
The HydroMT model configuration format has been redesigned.
The root YAML file now includes three main keys: ``modeltype``, ``global``, and ``steps``.

- ``modeltype`` (optional): Defines which model plugin is being used (e.g. ``model``, ``wflow_sbm``, ``wflow_sfincs`` etc.).
- ``global``: Defines model-wide configuration, including data catalog(s), or model components if using the core ``model`` plugin.
- ``steps``: Replaces the old numbered dictionary format with a sequential list of function calls.

Some of the functions (component specific read and write) are now explicitly mapped to model or component methods using the `<component>.<method>` syntax.
This is for example the case for reading and writing of individual model components (e.g. ``config.read``, ``grid.write`` etc.).

Additionally, to keep a consistent experience for our users we believe it is best to offer a single
format for configuring HydroMT, as well as reducing the maintenance burden on our side.
We have decided that **YAML** suits this use case the best. Therefore we have decided to
deprecate other config formats for configuring HydroMT including **ini** and **toml** formats.

Finally, the command line interface no longer supports a `--region` argument. The ``region`` should be specified
under the appropriate section of the YAML file depending on the model plugin you are using.

See more information about the current format in the :ref:`data catalog documentation <model_workflow>`.

How to upgrade
--------------

There is no automatic way to convert old HydroMT model workflow files to the new format.
This is mainly because the file is highly dependent on the specific model plugin and methods being used.
Some plugins may have changed the names of their methods or some of the arguments.
We therefore advise that you check the documentation of the specific model plugin you are using.
Usually they will provide templates or examples of the new YAML format as well to limit manual effort.

In general, you can follow these steps:

1. If you are using an ``.ini`` or ``.toml`` file to configure HydroMT, convert this to a YAML format.
   You can refer to the :ref:`model workflow documentation <model_workflow>` for examples of how to structure
   the YAML file.
2. Update the structure of the YAML file to include the ``modeltype``, ``global``, and ``steps`` keys.
3. For each step in the old format, convert it to the new list format under the ``steps`` key.
   Use the `<component>.<method>` syntax for component-specific methods. Be careful of indents.
4. Move the ``region`` specification from the command line to the YAML file under the appropriate section. This will depending on the model plugin you are using.
5. Review the function names and arguments to ensure they match the new API.
6. Test the updated workflow file with your HydroMT model to ensure it works as expected using the ``check`` command:

.. code-block:: bash

   hydromt check -i /path/to/your_workflow.yml -v
