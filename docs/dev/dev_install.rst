.. _dev_env:

Developer's environment
-----------------------

Developing HydroMT requires Python >= 3.8. We prefer developing with the most recent
version of Python. We strongly encourage you to develop in a separate conda environment.
To make sure all dependencies are up to dat we only provide a pyproject.toml that lists the dependencies.
However, there is a script that can generate the conda enviroment specification for you.

.. _dev_install:

Developer installation guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, clone the HydroMT ``git`` repo using `ssh <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ from
`github <https://github.com/Deltares/hydromt.git>`_.

.. code-block:: console

    $ git clone git@github.com:Deltares/hydromt.git
    $ cd hydromt

.. Note::

    In the commands below you can exchange `conda` for `mamba`, see :ref:`installation guide <installation_guide>` for the difference between both.

Then, navigate into the the code folder (where the envs folder and pyproject.toml are located):
To generate an `enviromnment.yml` file necessary to develop HydroMT simply run:

.. code-block:: console

    $ pip install tomli
    $ python make_env.py full

When the script is finished, a file called `environment.yml` will be created which you can pass to conda
as demonstrated in the sections below.

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydromt

Finally, create a developer installation of HydroMT:

.. code-block:: console

    $ pip install -e .
