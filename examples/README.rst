.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

Several iPython notebook examples have been prepared for **HydroMT** which you can
use as a HydroMT tutorial.

These examples can be run online or on your local machine.
To run these examples online press the **binder** badge above.

Local installation
------------------

To run these examples on your local machine you need a copy of the repository and
an installation of HydroMT including some additional packages. The most reliable
way to do this installation is by creating a new environment as described below.

1 - Install python and conda/mamba
**********************************
If not already done, you'll need Python 3.9 or greater and a package manager such as conda or mamba. These package managers help you to install (Python) packages and
`manage environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ such that different installations do not conflict.

We recommend using the `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_ Python distribution. This installs Python and the
`mamba package manager <https://github.com/mamba-org/mamba>`_. Alternatively, `Miniforge <https://github.com/conda-forge/miniforge>`_ and
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ will install Python and the `conda package manager <https://docs.conda.io/en/latest/>`_.

If you already have a python & conda installation but do not yet have mamba installed, we recommend installing it into your *base* environment using:

.. code-block:: console

  $ conda install mamba -n base -c conda-forge


2 - Install HydroMT and the other python dependencies in a separate Python environment
**************************************************************************************
The next step is to install all the python dependencies required to run the notebooks, including HydroMT.

**If you do not have HydroMT yet installed**, first create a new empty environment with the base hydromt installation:

.. code-block:: console

  $ mamba create -n hydromt -c conda-forge hydromt

To run the notebooks, you need to install the ``slim`` version of HydroMT using pip. The slim version installs additional dependencies to Hydromt
such as jupyter notebook to run the notebooks, matplotlib to plot or xugrid to also try out examples for the MeshModel. It is a more complete
installation of hydromt. To install or update in an existing environment (example hydromt environment), do:

.. code-block:: console

  $ conda activate hydromt
  $ pip install "hydromt[slim]"

3 - Download the content of the examples and notebooks
******************************************************
To run the examples locally, you will need to download the content of the HydroMT repository.
You have two options:

  1. Download and unzip the examples manually
  2. Clone the HydroMT GitHub repository

.. warning::

  Depending on your installed version of HydroMT, you will need to download the correct versions of the examples.
  To check the version of HydroMT that you have installed, do:

  .. code-block:: console

    $ hydromt --version

    hydroMT version: 0.8.0

** Option 1: manual download and unzip**

To manually download the examples on Windows, do (!replace with your own hydromt version!):

.. code-block:: console

  $ curl https://github.com/Deltares/hydromt/archive/refs/tags/v0.8.0.zip -O -L
  $ tar -xf v0.8.0.zip
  $ ren hydromt-0.8.0 hydromt

You can also download, unzip and rename manually if you prefer, rather than using the windows command prompt.

** Option 2: cloning the hydromt repository**

For git users, you can also get the examples by cloning the hydromt github repository and checking the version
you have installed:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git
  $ git checkout v0.8.0


4 - Running the examples
************************
Finally, start a jupyter notebook inside the ``examples`` folder after activating the ``hydromt`` environment, see below.
Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_ if you have that installed.

.. code-block:: console

  $ conda activate hydromt
  $ cd hydromt/examples
  $ jupyter notebook
