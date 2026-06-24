.. _installation_guide:

==================
Installation guide
==================

HydroMT is available on PyPI and conda-forge.
The most common usage of HydroMT is through a model plugin, e.g. via hydromt_wflow, hydromt_sfincs etc.
These plugins install HydroMT as a dependency.
However, if you want to use HydroMT directly, e.g. to develop your own model plugin or to use the HydroMT Python API without a plugin,
you can install HydroMT using the following steps.

.. _installation_prerequisites:

Prerequisite: python installation
=================================

You'll need **Python 3.11 or greater** and a package manager such as uv, pixi, or others in order to use HydroMT.
These package managers help you to install (Python) packages and
`manage environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
such that different installations do not conflict.

If you do not yet have such a package manager, we recommend using either:

- `uv <https://docs.astral.sh/uv/>`_: uses `pypi.org <https://pypi.org>`_ for downloading dependencies.
- `pixi <https://pixi.sh>`_: uses `conda forge <https://conda-forge.org/>`_ for downloading dependencies.


.. _installation_hydromt:

Installing HydroMT
==================

HydroMT is available from PyPI and conda-forge, and can be installed using uv, pixi, or others.
Here we will describe the installation using **uv** and the **pixi** package manager.

Basic installation
------------------

We strongly recommend installing HydroMT in a separate environment to avoid conflicts with other packages.
Therefore we use uv or pixi, which installs Python directly in the project folder.

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uv init my_hydromt
      $ cd my_hydromt
      $ uv add hydromt
      $ uv sync

    .. note::
      If you want to develop a model plugin, we recommend to :code:`uv init` with the ``--library`` option, which will create a library project instead of an application project.

  .. tab-item:: pixi
    :sync: pixi

    .. code-block:: console

      $ pixi init my_hydromt
      $ cd my_hydromt
      $ pixi add hydromt

To test whether the installation was successful, run :code:`uvx hydromt --plugins` on uv, or :code:`pixi run hydromt --plugins` on pixi.
The output should look similar to the example below:

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uvx hydromt --plugins
        Model plugins:
          - model (hydromt x.y.z)
          - example_model (hydromt x.y.z)
        Component plugins:
          - ConfigComponent (hydromt x.y.z)
          - DatasetsComponent (hydromt x.y.z)
          - GeomsComponent (hydromt x.y.z)
        Driver plugins:
          - dataset_xarray (hydromt x.y.z)
          - geodataframe_table (hydromt x.y.z)
        Catalog plugins:
          - deltares_data (hydromt x.y.z)
          - artifact_data (hydromt x.y.z)
        Uri_resolver plugins:
          - azure_blob (hydromt x.y.z)
          - convention (hydromt x.y.z)
          - raster_tindex (hydromt x.y.z)

  .. tab-item:: pixi
    :sync: pixi

    .. code-block:: console

      $ pixi run hydromt --plugins
        Model plugins:
          - model (hydromt x.y.z)
          - example_model (hydromt x.y.z)
        Component plugins:
          - ConfigComponent (hydromt x.y.z)
          - DatasetsComponent (hydromt x.y.z)
          - GeomsComponent (hydromt x.y.z)
        Driver plugins:
          - dataset_xarray (hydromt x.y.z)
          - geodataframe_table (hydromt x.y.z)
        Catalog plugins:
          - deltares_data (hydromt x.y.z)
          - artifact_data (hydromt x.y.z)
        Uri_resolver plugins:
          - azure_blob (hydromt x.y.z)
          - convention (hydromt x.y.z)
          - raster_tindex (hydromt x.y.z)



Installing optional dependencies
--------------------------------

HydroMT comes with a minimal set of dependencies.
However, depending on your use case, you might want to install additional optional dependencies.
For example, if you want to work with cloud data from AWS or Google Cloud Storage.

Some of the optional dependencies can be installed using uv/pip and predefined lists of dependencies:

- **io**: for additional input data support (e.g. cloud data, parquet or excel files...).
- **extra**: for additional functionalities. So far includes matplotlib and pyet.
- **examples**: for running jupyter notebooks and HydroMT examples.
- **slim**: installs optional dependencies in io, extra and examples.

To install these optional dependencies, you can use the following uv/pip commands:

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uv pip install "hydromt[io]"
      $ uv pip install "hydromt[extra]"
      $ uv pip install "hydromt[examples]"
      $ uv pip install "hydromt[slim]"

  .. tab-item:: pixi
    :sync: pixi

    Not required, as pixi install via conda-forge, which contains all optional dependencies by default.


Developer's installation
------------------------
If you want to contribute to the HydroMT codebase, or make some local changes, we advise
you to install HydroMT in developer mode. We have some different recommendations for this
available in the :ref:`developer's installation guide <dev_install>`.

.. _installation_examples:

Downloading and running the examples
====================================

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

Several iPython notebook examples have been prepared for **HydroMT** which you can
use as a HydroMT tutorial.

These examples can be run online or on your local machine.
To run these examples online press the **binder** badge above.

To run these examples locally, you need to:

1. Install HydroMT including the **examples** optional dependencies as described above.
2. Download the examples from the HydroMT GitHub repository. You can either
   clone the repository (option 1) or download and unzip the examples manually (option 2).

The examples will depend on which HydroMT version you have installed. You first need to
check which version you have using:

.. warning::

  Depending on your installed version of HydroMT, you will need to download the correct versions of the examples.
  To check the version of HydroMT that you have installed, do:

  .. code-block:: console

    $ hydromt --version

    hydroMT version: 1.4.0

**Option 1 - Clone the HydroMT GitHub repository**

For git users, you can also get the examples by cloning the hydromt github repository and checking the version
you have installed:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git
  $ git checkout v1.4.0

**Option 2 - Download and unzip the examples manually**

To manually download the examples on Windows, do (!replace with your own hydromt version!):

.. code-block:: console

  $ curl https://github.com/Deltares/hydromt/archive/refs/tags/v1.4.0.zip -O -L
  $ tar -xf v1.4.0.zip
  $ ren hydromt-1.4.0 hydromt

You can also download, unzip and rename manually if you prefer, rather than using the windows command prompt.

**Running the examples**

Finally, start a jupyter notebook inside the ``examples`` folder after activating the ``hydromt`` environment, see below.
Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_ if you have that installed.

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ cd hydromt/examples
      $ uvx jupyter notebook

  .. tab-item:: pixi
    :sync: pixi

    .. code-block:: console

      $ cd hydromt/examples
      $ pixi run jupyter notebook
