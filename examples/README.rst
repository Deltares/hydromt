.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

Several iPython notebook examples have been prepared for **HydroMT** which you can 
use as a HydroMT tutorial. 

These examples can be run online or on your local machine. 
To run these examples online press the **binder** badge above.

To run these examples on your local machine you need a copy of the repository and an installation 
of HydroMT including some additional packages. Please refer to the online :ref:`installation guide <installation_guide>`
for more information about installing HydroMT.

First, clone the HydroMT github repository, this creates a local copy of the repository on your local machine.

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt.git

Then, navigate into the cloned ``hydromt`` folder:

.. code-block:: console

  $ cd hydromt/examples

Create a conda environment based on the ``environment.yml`` in the binder folder and activate this environment: 

.. code-block:: console

  $ conda env create -f ../binder/environment.yml
  $ conda activate hydromt

Install the latest HydroMT version from cloned files into the environment.

.. code-block:: console

  $ flit install --dep production

Finally, start a jupyter notebook inside the ``examples`` folder and activated ``hydromt`` environment.

.. code-block:: console

  $ jupyter notebook

