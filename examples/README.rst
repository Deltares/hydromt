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
If not already done, you'll need Python 3.8 or greater and a package manager such as conda or mamba. These package managers help you to install (Python) packages and 
`manage environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ such that different installations do not conflict.

We recommend using the `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_ Python distribution. This installs Python and the 
`mamba package manager <https://github.com/mamba-org/mamba>`_. Alternatively, `Miniforge <https://github.com/conda-forge/miniforge>`_ and 
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ will install Python and the `conda package manager <https://docs.conda.io/en/latest/>`_.

If you already have a python & conda installation but do not yet have mamba installed, we recommend installing it into your *base* environment using:

.. code-block:: console

  $ conda install mamba -n base -c conda-forge

2 - Install HydroMT and the other python dependencies in a separate Python environment
**************************************************************************************
The last step is to install all the python dependencies required to run the notebooks, including HydroMT. All required dependencies can be found
in the `environment.yml <https://github.com/Deltares/hydromt/blob/main/environment.yml>`_ file in the root of the repository. This will create
a new environment called **hydromt**. If you already have this environment with this name either remove it with `conda env remove -n hydromt`
**or** set a new name for the environment by adding `-n <name>` to the line below. 

Note that you can exchange mamba for conda in the line below if you don't have mamba installed (see above for how to install mamba).

.. code-block:: console

  $ mamba env create -f https://raw.githubusercontent.com/Deltares/hydromt/main/environment.yml

3 - Download the content of the HydroMT github repository
*********************************************************
To run the examples locally, you will need to download the content of the HydroMT repository. You can either do a
`manual download <https://github.com/Deltares/hydromt/archive/refs/heads/main.zip>`_ and extract the content of the downloaded ZIP folder 
**or** clone the repository locally:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git

4 - Running the examples
************************
Finally, start a jupyter notebook inside the ``examples`` folder after activating the ``hydromt`` environment, see below.
Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_ if you have that installed.

.. code-block:: console

  $ conda activate hydromt
  $ cd examples
  $ jupyter notebook


