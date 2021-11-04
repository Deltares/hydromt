Installation
============

User install
------------

HydroMT is available from pypi and conda-forge, but we recommend installing with conda.

To install HydroMT using conda do:

.. code-block:: console

    $ conda install hydromt -c conda-forge

This will install the core HydroMT libray including the model API. In order to install other models (example: 
wflow, SFINCS etc.), then separate :ref:`HydroMT plugin <plugin_install>` need to be installed as well.

You can also install hydromt using pip:

.. code-block:: console

    $ pip install hydromt


Developper install
------------------
If you want to download HydroMT directly from git to easily have access to the latest developments or 
make changes to the code you can use the following steps.

First, clone hydromt's ``git`` repo from
`github <https://github.com/Deltares/hydromt.git>`_, then navigate into the 
the code folder (where the envs folder and pyproject.toml are located):

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt.git
    $ cd hydromt

Then, make and activate a new hydromt-dev conda environment based on the envs/hydromt-dev.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f envs/hydromt-dev.yml
    $ conda activate hydromt-dev

Finally, build and install hydromt using pip.

.. code-block:: console

    $ pip install .

If you wish to make changes in hydromt, then you should make an editable install of hydromt. 
This is possible using the `flit <https://flit.readthedocs.io/en/latest/>`_ package and install command.

For Windows:

.. code-block:: console

    $ flit install --pth-file

For Linux:

.. code-block:: console

    $ flit install -s

.. _plugin_install:

Plugins install
---------------
HydroMT core can easily access and navigate through different model implementations thanks to its plugin architecture.
The hydromt package contains core functionnalities and the command line interface. The models, following HydroMT's 
architecture, are then implemented in separate plugins available in other python packages. 

Known model plugin packages linked to HydroMT are:

- `hydromt_delwaq <https://github.com/Deltares/hydromt_delwaq>`_ for delwaq
- `hydromt_fiat <https://github.com/Deltares/hydromt_fiat>`_ for fiat
- `hydromt_ribasim <https://github.com/Deltares/hydromt_ribasim>`_ for ribasim
- `hydromt_sfincs <https://github.com/Deltares/hydromt_sfincs>`_ for SFINCS
- `hydromt_wflow <https://github.com/Deltares/hydromt_wflow>`_ for wflow and wflow_sediment

You can follow installation instructions of the different plugins in their own documentation pages (same steps as for the core).
