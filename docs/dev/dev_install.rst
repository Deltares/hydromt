.. _dev_env:

Developer's environment
-----------------------

Developing HydroMT requires Python >= 3.9. We prefer developing with the most recent
version of Python. We strongly encourage you to develop in a separate conda environment.
To make sure all dependencies are up to dat we only provide a pyproject.toml that lists the dependencies.
However, there is a script that can generate the conda enviroment specification for you.

.. _dev_install:

Developer installation guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we describe two ways to install HydroMT for development.
The first is using the `makefile` which boils down to running a single command.
The second is a step-by-step guide that shows you exactly what the makefile does
and how to do it manually.

But first, you need to clone the HydroMT ``git`` repo using
`ssh <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_
from `github <https://github.com/Deltares/hydromt.git>`_.
Then, navigate into the the code folder (where the pyproject.toml is located)

.. code-block:: console

    $ git clone git@github.com:Deltares/hydromt.git
    $ cd hydromt


Makefile based installation
---------------------------

You wil need a usable instalation of `make`. If you're using linux this should already be installed by default.
If you are using windows we recommend you install it using `chocolatey <https://chocolatey.org/install>`_ like so:

.. code-block:: console

    $ choco install make

Afterwards to create an environment you can simply call:

.. code-block:: console

    $ make dev

This will create an enviromnment file, create an environment called `hydromt-dev` using a package mangager
and install hydromt into it. By default the makefile will attempt to use `mamba`. If you want it to use a
diferent one like, conda, you only have to set the environment variable like this:

.. code-block:: console

    $ make dev PY_ENV_MANAGER=conda

If you would like this to be the case you can also add make this the default behaviour with the following command:

.. code-block:: console

    $ echo 'export PY_ENV_MANAGER=conda' >> ~/.$(basename $0)rc

Step-by-step installation
--------------------------

If you want you can also do the steps of `make dev` manually.

To generate a usable mamba/conda environment you'll need to have `tomli` installed.
You can simply install this with pip (only required for python < 3.11):

.. code-block:: console

    $ pip install tomli

The first step is to create the `environment.yml`:

.. code-block:: console

    $ python make_env.py full -n hydromt-dev

When the script is finished, a file called `environment.yml` will be created which you can pass to mamba
as demonstrated in the sections below. This will include all optional dependencies of HydroMT.

After the environment file has been created you can create an environment out of it by running:

.. code-block:: console

    $ mamba env create -f environment.yml
    $ mamba activate hydromt-dev

Finally, create a developer installation of HydroMT:

.. code-block:: console

    $ pip install -e .

.. Note::

    In the commands above you can exchange `mamba` for `conda`,
    see :ref:`installation guide <installation_guide>` for the difference between both.

Fine tuned installation
-----------------------

If you want a more fine tuned installation you can also specify exactly
which dependency groups you'd like. For instance, this will create an environment
with the extra, io and doc dependencies.

.. code-block:: console

    $ make env OPT_DEPS="extra,io,doc" ENV_NAME="hydromt-extra-io-doc"


Or manually:

.. code-block:: console

    $ pip install tomli # only required for python < 3.11
    $ python make_env.py "extra,io,doc" -n hydromt-extra-io-doc
    $ mamba env create -f environment.yml


We have 7 optional dependency groups you can specify (see `pyproject.toml` for list of dependencies in each group):

1. `io`: Reading and writing various formats like excel but also cloud file systems
2. `extra`: Couldn't think of a better name for this one, but it has some extra for ET and mesh calculations
3. `dev`: everything you need to develop and publish HydroMT
4. `test` What you need to run the test suite. Test suite should be setup that only tests that use the dependencies that are installed are run, so this should always pass no matter what other dependencies you may or may not have installed.
5. `doc` generate the docs
6. `examples` Run Jupyter notebook examples. Used this for binder support mostly.
7. `deprecated` dependencies that we hope to remove soon, but aren't quite ready to yet.


We also have 3 "flavors". These are more or less just collections of one or more groups designed for common use cases:
1. `min` no optional dependencies. mostly as a base to build your DIY stack on.
2. `slim` Just the operational bits, what most people will probably want if you using HydroMT and what the cloud will most likely use
3. `full` absolutely everything, useful for developing.

We also have docker images for each of the flavours that should be published soon (but are not yet as the writing of this section)
