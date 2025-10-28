.. _installation_guide:

==================
Installation guide
==================

HydroMT is available on PyPI and conda-forge. The most common usage of HydroMT is
through a model plugin, e.g. via hydromt_wflow, hydromt_sfincs etc. These plugins
install HydroMT as a dependency. However, if you want to use HydroMT directly, e.g. to
develop your own model plugin or to use the HydroMT Python API without a plugin, you can
install HydroMT using the following steps.

.. _installation_prerequisites:

Prerequisite: python installation
=================================

You'll need **Python 3.11 or greater** and a package manager such as conda, mamba or
others in order to use HydroMT. These package managers help you to install (Python)
packages and
`manage environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
such that different installations do not conflict.

If you do not yet have such a package manager, we recommend using either:

- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_: uses the `conda package manager <https://docs.conda.io/en/latest/>`_
- `Miniforge <https://github.com/conda-forge/miniforge#mambaforge>`_: uses the `mamba package manager <https://github.com/mamba-org/mamba>`_


.. _installation_hydromt:

Installing HydroMT
==================

HydroMT is available from PyPI and conda-forge, and can be installed using pip, conda or
mamba. For an even faster installation, we recommend using `uv <https://docs.astral.sh/uv/>`_
package manager. It is an equivalent to pip but 10-100x faster for installing packages.

Here we will describe the installation using **uv** and the **conda** package
manager (for mamba just replace `conda` with `mamba` in the commands below).

Basic installation
------------------

We strongly recommend installing HydroMT in a separate environment to avoid conflicts
with other packages. You can create a new environment and install HydroMT in it
using the following command:

.. code-block:: console

    $ conda create -n hydromt uv python -c conda-forge
    $ conda activate hydromt
    $ uv pip install hydromt

This will create a new isolated environment called **hydromt** and install hydromt into
it using uv and pip. To test whether the installation was successful you can run
:code:`hydromt --plugins` and the output should look approximately like the one below:

.. code-block:: shell

    $ hydromt --plugins
        Model plugins:
                - model (hydromt 1.0.0)
        Component plugins:
                - ConfigComponent (hydromt 1.0.0)
                - GeomsComponent (hydromt 1.0.0)
                - GridComponent (hydromt 1.0.0)
                - TablesComponent (hydromt 1.0.0)
                - VectorComponent (hydromt 1.0.0)
                - MeshComponent (hydromt 1.0.0)
                - DatasetsComponent (hydromt 1.0.0)
        Driver plugins:
                - geodataframe_table (hydromt 1.0.0)
                - geodataset_vector (hydromt 1.0.0)
                - geodataset_xarray (hydromt 1.0.0)
                - pandas (hydromt 1.0.0)
                - pyogrio (hydromt 1.0.0)
                - raster_xarray (hydromt 1.0.0)
                - rasterio (hydromt 1.0.0)
        Catalog plugins:
                - deltares_data (hydromt 1.0.0)
                - artifact_data (hydromt 1.0.0)
                - aws_data (hydromt 1.0.0)
                - gcs_cmip6_data (hydromt 1.0.0)



Installing optional dependencies
--------------------------------

HydroMT comes with a minimal set of dependencies. However, depending on your use case,
you might want to install additional optional dependencies. For example, if you want to
work with cloud data from AWS or Google Cloud Storage.

Some of the optional dependencies can be installed using uv/pip and predefined lists of
dependencies:

- **io**: for additional input data support (e.g. cloud data, parquet or excel files...).
- **extra**: for additional functionalities. So far includes matplotlib and pyet.
- **examples**: for running jupyter notebooks and HydroMT examples.
- **slim**: installs optional dependencies in io, extra and examples.

To install these optional dependencies, you can use the following uv/pip commands:

.. code-block:: console

    $ uv pip install "hydromt[io]"
    $ uv pip install "hydromt[extra]"
    $ uv pip install "hydromt[examples]"
    $ uv pip install "hydromt[slim]"

.. note::

  If you are using caching of mosaic rasters and vrt files, the gdal library needs to be
  installed in your conda environment. Unfortunately this cannot be done using pip.
  Therefore, if you want to use this functionality, please install gdal using conda or
  mamba:

  .. code-block:: console

    $ conda install -c conda-forge gdal


Developer's installation
------------------------
If you want to contribute to the HydroMT codebase, or make some local changes, we advise
you to install HydroMT in developer mode. We have some different recommendations for this
available in the :ref:`developer's installation guide <guides/core_dev/dev_install>`_.

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

    hydroMT version: 1.0.0

**Option 1 - Clone the HydroMT GitHub repository**

For git users, you can also get the examples by cloning the hydromt github repository and checking the version
you have installed:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git
  $ git checkout v1.0.0

**Option 2 - Download and unzip the examples manually**

To manually download the examples on Windows, do (!replace with your own hydromt version!):

.. code-block:: console

  $ curl https://github.com/Deltares/hydromt/archive/refs/tags/v1.0.0.zip -O -L
  $ tar -xf v1.0.0.zip
  $ ren hydromt-1.0.0 hydromt

You can also download, unzip and rename manually if you prefer, rather than using the windows command prompt.

**Running the examples**

Finally, start a jupyter notebook inside the ``examples`` folder after activating the ``hydromt`` environment, see below.
Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_ if you have that installed.

.. code-block:: console

  $ conda activate hydromt
  $ cd hydromt/examples
  $ jupyter notebook
