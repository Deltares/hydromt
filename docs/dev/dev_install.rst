.. _dev_env:

Development environment
-----------------------

Developing HydroMT requires Python >= 3.6. We prefer developing with the most recent 
version of Python. We strongly encourage you to develop in a separate conda environment.
All Python dependencies required to develop HydroMT can be found in `envs/hydromt-dev.yml <environment.yml>`__.

Developer installation guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, clone hydromt's ``git`` repo from
`github <https://github.com/Deltares/hydromt.git>`_.

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt.git
    $ cd hydromt

Then, navigate into the the code folder (where the envs folder and pyproject.toml are located):
Make and activate a new hydromt-dev conda environment based on the envs/hydromt-dev.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f envs/hydromt-dev.yml
    $ conda activate hydromt-dev

Finally, create a developer installation of HydroMT:
This is possible using the `flit <https://flit.readthedocs.io/en/latest/>`_ package and install command.

For Windows:

.. code-block:: console

    $ flit install --pth-file

For Linux:

.. code-block:: console

    $ flit install -s