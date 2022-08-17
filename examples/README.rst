.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

Several iPython notebook examples have been prepared for **HydroMT** which you can 
use as a HydroMT tutorial. 

These examples can be run online or on your local machine. 
To run these examples online press the **binder** badge above.

To run these examples on your local machine you need a copy of the repository and 
an installation of HydroMT including some additional packages. The most reliable 
way to do this installation is by creating a new environment as described below.

First, download and extract the `zipped HydroMT github repository <https://github.com/Deltares/hydromt/archive/refs/heads/main.zip>`_

Then, navigate into the extracted ``hydromt`` folder and execute the following lines from the command line.

Next, create a conda environment based on the ``environment.yml`` in the binder folder, 
activate this environment and install the HydroMT version from downloaded files.

.. code-block:: console

  $ conda env create -f binder/environment.yml
  $ conda activate hydromt
  $ flit install --dep production

Finally, start a jupyter notebook inside the ``examples`` folder and activated ``hydromt`` environment.

.. code-block:: console

  $ cd examples
  $ jupyter notebook

