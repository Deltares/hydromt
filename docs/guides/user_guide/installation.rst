.. _installation_guide:

==================
Installation guide
==================

Prerequisites
=============

Python and conda/mamba
-----------------------
You'll need **Python 3.9 or greater** and a package manager such as conda or mamba.
These package managers help you to install (Python) packages and `manage
environments`_ such that different installations do not conflict.

We recommend using the Mambaforge_ Python distribution. This installs Python
and the `mamba package manager`_. Miniforge_ and Miniconda_ will install
Python and the `conda package manager`_. Differences to note, in a nutshell:

* **mamba** is much faster than **conda**, but has identical commands.
* Mambaforge and miniforge are community driven installers, installing by
  default from the **conda-forge channel**.
* Miniconda is a company driven (Anaconda) installer, installing by default
  from the **anaconda channel**.
* Installing from the anaconda channel has certain (legal) `limitations for commercial use <limitations>`_.

Installing Mambaforge/Miniforge/Miniconda does not require administrative
rights to your computer and doesn't interfere with any other Python
installations in your system.

Dependencies
------------

The HydroMT Python package makes extensive use of the modern scientific Python
ecosystem. The most important dependencies are listed here (for a complete list,
see the pyproject.toml file in the repository root). These dependencies are automatically installed when
installing HydroMT with a package manager, such as conda or mamba.

Data structures:

* `pandas <https://pandas.pydata.org/>`__
* `numpy <https://www.numpy.org/>`__
* `xarray <https://xarray.pydata.org/>`__

Delayed/out-of-core computation, parallelization:

* `dask <https://dask.org/>`__

Spatial and statistical operations:

* `scipy <https://docs.scipy.org/doc/scipy/reference/>`__

(Hydro-)Geospatial libraries:

* `geopandas <https://geopandas.org/en/stable/>`__
* `pyproj <https://pyproj4.github.io/pyproj/stable/>`__
* `rasterio <https://rasterio.readthedocs.io/en/latest/>`__
* `pyflwdir <https://deltares.github.io/pyflwdir/latest/>`__

Command line interface

* `click <https://click.palletsprojects.com/>`__


Installation
============

HydroMT is available from pypi and conda-forge, but we recommend installing from conda-forge in a new conda environment.

.. Tip::

    In the commands below you can exchange `conda` for `mamba`, see above for the difference between both.

.. Note::

    If you would like to try out the hydromt examples notebook, follow instead the
    :ref:`examples installation guide <examples>`.

Install HydroMT in a new environment
------------------------------------
.. Tip::

    This is our recommended way of installing HydroMT!

To install HydroMT in a new environment called `hydromt` from the conda-forge channel do:

.. code-block:: console

    $ conda create -n hydromt -c conda-forge hydromt

Then, activate the environment (as stated by conda create) to start making use of HydroMT.
To test whether the installation was successful you can run :code:`hydromt --models` and the output should
look approximately like the one below:


.. code-block:: console

    $ hydromt --models

    model plugins:
    generic models (hydromt 0.7.2):
     - grid_model
     - vector_model
     - mesh_model
     - network_model

Optionally, specific versions of python or other dependencies can be set and additional packages can be added,
for example Python version 3.9 and GDAL 3.4.1 and the HydroMT-Wflow plugin:

.. code-block:: console

    $ conda create -n hydromt -c conda-forge hydromt python=3.9 gdal=3.4.1 hydromt_wflow


Install HydroMT in an existing environment
------------------------------------------

To install HydroMT **using mamba or conda** execute the command below after activating the correct environment.
Note that if some dependencies are not installed from conda-forge the installation may fail.

.. code-block:: console

    $ conda install hydromt -c conda-forge

You can also install HydroMT **using pip** from pypi (not recommended):

.. code-block:: console

    $ pip install hydromt

To install the **latest (unreleased) version from github**, execute the command below.
Note that you might have to uninstall HydroMT first to successfully install from github.

.. code-block:: console

    $ pip install git+https://github.com/Deltares/hydromt.git

.. _plugin_install:

Install HydroMT plugins
------------------------
To use HydroMT to set up specific models, we  separate plugins that are available as separate python packages.
Most plugins are available on conda-forge and can be installed in the same environment. For instance,
to install HydroMT-Wflow in the environment where you have already installed HydroMT do:

.. code-block:: console

    $ conda install hydromt_wflow -c conda-forge

For detailed instructions, please visit the plugin documentation pages, see :ref:`overview of plugins <plugins>`.

Developer installation
----------------------

To be able to test and develop the HydroMT package see instructions in the :ref:`Developer installation guide <dev_install>`.

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Mambaforge: https://github.com/conda-forge/miniforge#mambaforge
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _limitations: https://www.anaconda.com/blog/anaconda-commercial-edition-faq
.. _mamba package manager: https://github.com/mamba-org/mamba
.. _conda package manager: https://docs.conda.io/en/latest/
.. _pip package manager: https://pypi.org/project/pip/
.. _manage environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
