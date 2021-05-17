.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

This folder contains several ipython notebook examples for **hydroMT**. 

To run these examples start with the **binder** badge above.

To run these examples on your local machine create a conda e nvironment based on the 
environment.yml in the binder folder of this repository and than start jupyer notebook. 
Run the following steps the examples folder:

.. code-block:: console

  conda env create -f ../binder/environment.yml  # install dependencies
  conda activate hydromt
  flit install --dep production  # install hydromt
  jupyter notebook