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
To generate a usable mamba/conda environment you'll need to have `tomli` installed. You can simply install this with pip:

.. code-block:: console

    $ pip install tomli

You wil also need a suable instalation of `make`. If you're using linux this should already be installed by defautl. If you are using windows we
recommend you install it using `chocolatey <https://chocolatey.org/install>`_ like so:

.. code-block:: console

    $ choco install make

Afterwards to create an environment you can simply call:

.. code-block:: console

    $ make env

This will create an enviromnment file, create an environment using a packagemangager and install hydromt into it. By default
the makefile will attempt to use `micromamba`. If you want it to use a diferent one like, conda, you only have to set the
environment variable like this if you're on linux:

.. code-block:: console

    $ export PY_ENV_MANAGER=conda

or like this if you're on windows:

.. code-block::console

    $ set PY_ENV_MANAGER=conda

If you want you can also do these steps manually. The first step is to create the `environment.yml`:

.. code-block:: console

    $ python make_env.py

When the script is finished, a file called `environment.yml` will be created which you can pass to conda
as demonstrated in the sections below.

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydromt

Finally, create a developer installation of HydroMT:

.. code-block:: console

    $ pip install -e .
