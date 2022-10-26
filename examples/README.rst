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
`mamba package manager <https://github.com/mamba-org/mamba>`_. `Miniforge <https://github.com/conda-forge/miniforge>`_ and 
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ will install Python and the `conda package manager <https://docs.conda.io/en/latest/>`_.

2 - Download the content of the HydroMT github repository
*********************************************************
To run the exercises, you will need to download the content of the HydroMT repository locally. You can either do a
`manual download <https://github.com/Deltares/hydromt/archive/refs/heads/main.zip>`_ and extract the content of the downloaded ZIP folder 
**or** clone the repository locally:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git


3 - Install HydroMT and the other python dependencies in a separate Python environment
**************************************************************************************
The last step is to install all the python dependencies required to run the notebooks, including HydroMT. All required dependencies can be found
in the `environment.yml <https://github.com/Deltares/hydromt/blob/main/binder/environment.yml>`_ file. 

First navigate into the extracted ``hydromt`` folder (where the binder and examples folder are located). Create a new *hydromt* environment using the environment.yml file 
in the binder folder (you can exchange mamba/conda in the example below):

.. code-block:: console

  $ cd hydromt
  $ mamba env create -f binder/environment.yml

4 - Running the examples
************************
Finally, start a jupyter notebook inside the ``examples`` folder and activated ``hydromt`` environment.

.. code-block:: console

  $ conda activate hydromt
  $ cd examples
  $ jupyter notebook


