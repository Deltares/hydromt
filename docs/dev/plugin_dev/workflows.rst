
.. _kernel_config:

Preparing a model workflow
==========================

A user can define a complete pipeline of methods and their options to :ref:`build a model <model_build>` or :ref:`updating an existing model <model_update>`
in the configuration `.yaml file <https://en.wikipedia.org/wiki/YAML>`_

**Steps in brief:**

1. Start with a template of the :ref:`HydroMT model plugin <plugins>` with which you want to build / update a model. Templates can be found in the online documentation of each model plugin.
2. Edit / add / remove sections (i.e. methods) based on which components you want to build or adapt. The arguments of specific methods can be found in the API chapter in the online documentation of each model plugin.
3. Save the configuration file and use it in combination with the HydroMT :ref:`build <model_build>` and :ref:`update <model_update>` methods.


Workflow (.yaml) file
--------------------------------

The YAML file serves as the configuration and workflow definition for building and updating models in HydroMT. It uses a simple key-value syntax with structured sections to define steps and arguments for each method. Each step corresponds to a model action (method), and arguments specify how that action is applied. The order of execution is determined by the sequence of steps listed in the YAML file.

- **Model Type**: The top level defines the `modeltype`, such as "model".
- **Global Components**: Under the `global` section, you can define reusable components
  that apply throughout the model workflow. Each component is named and has a specific
  type, like `GridComponent` or `ConfigComponent`.
- **Steps**: The main logic of the model is defined under `steps`. Each step is a method call, which typically includes the method name and its required arguments. The order of the steps is critical, as HydroMT executes each step sequentially.

Below is an example of the YAML format used in HydroMT:

.. code-block:: yaml

  ---
  modeltype: model
  global:
    components:
      grid:
        type: GridComponent
      config:
        type: ConfigComponent

  steps:
    - config.update:
        data:
          header.settings: value
          timers.end: '2010-02-15'
          timers.start: '2010-02-05'

    - grid.read:
        filename: grid.nc

    - write:
        components:
          - config
          - grid

### Explanation of Key Methods

- **`config.update`**: Updates configuration settings. In the example, it sets parameters like `header.settings`, and start and end times for the model run.
- **`grid.read`**: Reads the grid data from a specified file (e.g., `grid.nc`).
- **`write`**: Specifies which components of the model (e.g., `config`, `grid`) should be written to disk at the end of the workflow. By default, all files are written unless specified otherwise.

It should be noted that, by default, the HydroMT `build` and `update` commands write all output files at the end of the workflow using the `write` method. This behavior can be customized by explicitly specifying the `write` step in the YAML file, allowing more granular control over which files are written and when.
