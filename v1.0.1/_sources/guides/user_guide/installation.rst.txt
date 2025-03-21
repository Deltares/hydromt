.. _installation_guide:

==================
Installation guide
==================

Prerequisites
=============

You'll need **Python 3.9, 3.10, 3.11, 3.12, or 3.13** and a package manager.
We recommend using pixi.

Installation
============

HydroMT is available from pypi and conda-forge,
but we recommend installing from conda-forge using the pixi package manager.

Install HydroMT CLI
------------------------------------
.. Tip::

    This is our recommended way of installing HydroMT!

To make the HydroMT cli available anywhere on the system using pixi execute the command:

.. code-block:: console

    $ pixi global install hydromt

This will create a new isolated environment and install hydromt into it.
To test whether the installation was successful you can run :code:`hydromt --plugins` and the output should look approximately like the one below:

.. code-block:: shell

    $ hydromt --plugins
        Model plugins:
                - model (hydromt 1.0.0)
        Component plugins:
                - ConfigComponent (hydromt 1.0.0)
                - GeomsComponent (hydromt 1.0.0)
                - GridComponent (hydromt 1.0.0)
                - TablesComponent (hydromt 1.0.0)
                - VectorComponent (hydromt 1.0.0)
                - MeshComponent (hydromt 1.0.0)
                - DatasetsComponent (hydromt 1.0.0)
        Driver plugins:
                - geodataframe_table (hydromt 1.0.0)
                - geodataset_vector (hydromt 1.0.0)
                - geodataset_xarray (hydromt 1.0.0)
                - pandas (hydromt 1.0.0)
                - pyogrio (hydromt 1.0.0)
                - raster_xarray (hydromt 1.0.0)
                - rasterio (hydromt 1.0.0)
        Catalog plugins:
                - deltares_data (hydromt 1.0.0)
                - artifact_data (hydromt 1.0.0)
                - aws_data (hydromt 1.0.0)
                - gcs_cmip6_data (hydromt 1.0.0)



Installing HydroMT in a python environment
------------------------------------------

If you wish to use hydromt through it's Python API, you can use pixi to create an environment for this too.
If you do not have a ``pyproject.toml`` yet you can make one by executing the command:

.. code-block:: shell

    $ pixi init --format pyproject

Which will create it for you.
After this simply add HydroMT as a dependency with the following command:

.. code-block:: shell

    $ pixi add hydromt

Once you have your new (or existing ``pyproject.toml``) file install the pixi
environment and activate it with the following commands to be able to start using it:

.. code-block:: shell

    $ pixi install
    $ pixi shell activate
