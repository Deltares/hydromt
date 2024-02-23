.. _dev_install:

Developer installation guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we describe three ways to install HydroMT for development.
HydroMT makes use of pixi as a task runner with multiple environments that can be activated.

But first, you need to clone the HydroMT ``git`` repo using
`ssh <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_
from `github <https://github.com/Deltares/hydromt.git>`_.
Then, navigate into the the code folder (where the pyproject.toml is located)

.. code-block:: console

    $ git clone git@github.com:Deltares/hydromt.git
    $ cd hydromt


Pixi based installation
---------------------------

We use `pixi <https://prefix.dev/docs/pixi/overview>`_ as our task runner. For more information about installing pixi please refer to the linked webpage.

Once pixi is installed you can create a developer installation by running the following command:

.. code-block:: console

    $ pixi run -e <ENVIRONMENT> install

This will automatically install all dependencies needed to develop HydroMT, and it install HydroMT within the current environment in editable mode.
You can use `open-vscode.bat` to open the current folder in VSCode with the correct environment activated.
This bat-file simply runs `pixi run -e <ENVIRONMENT> code .` to set up the pixi environment before starting VSCode.
The python installation can be found in the `.pixi` folder. No need to switch interpreters.
The `ENVIRONMENT` to specify is of your preference, based on what you are going to do.
The most complete environment would be `full-py311` / `default`.


Fine tuned installation
-----------------------

If you want a more fine tuned installation you can also specify exactly which features you'd like to install.
For instance, you can add a new environment in `pixi.lock` to install _extra_, _io_ and _doc_ dependencies.

.. code-block:: toml

    [environments]
    my_env = ["extra", "io", "doc"]

We have 7 optional dependency groups you can specify (see `pixi.toml` for list of dependencies in each group):

1. `io`: Reading and writing various formats like excel but also cloud file systems
2. `extra`: Couldn't think of a better name for this one, but it has some extra for ET and mesh calculations
3. `dev`: everything you need to develop and publish HydroMT
4. `test` What you need to run the test suite. Test suite should be setup that only tests that use the dependencies that are installed are run, so this should always pass no matter what other dependencies you may or may not have installed.
5. `doc` generate the docs
6. `examples` Run Jupyter notebook examples. Used this for binder support mostly.
7. `deprecated` dependencies that we hope to remove soon, but aren't quite ready to yet.


We also have 3 pre-defined environments. These are more or less just collections of one or more groups designed for common use cases:
1. `min` no optional dependencies. mostly as a base to build your DIY stack on.
2. `slim` Just the operational bits, what most people will probably want if you're using HydroMT, and it's what the cloud will most likely use
3. `full` absolutely everything, useful for developing.

We also have docker images for each of the flavours available on the deltares dockerhub page.
