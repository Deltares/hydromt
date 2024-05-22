
.. _existing_catalog:


Pre-defined data catalogs
=========================

This page contains a list of (global) datasets which can be used with various HydroMT models and workflows.
Below are drop down lists with datasets per pre-defined data catalog for use with HydroMT.
The summary per dataset contains links to the online source and available literature.

The ``deltares_data`` catalog is only available within the Deltares network. However a selection of this data for a the
Piave basin (Northern Italy) is available online in the ``artifact_data`` archive and will be used if no data catalog is provided.
Local or other datasets can also be included by extending the data catalog with new yaml :ref:`data catalog files <data_yaml>`.
We plan to provide more data catalogs with open data sources in the (near) future.
See the data catalog `changelog <https://github.com/Deltares/hydromt/blob/main/data/catalogs/changelog.rst>`_ for recent updates on the pre-defined catalogs.

Using a predefined catalog
--------------------------

From CLI
~~~~~~~~

To use a predefined catalog, you can specify the catalog name with the ``--dd`` or ``--data_catalog`` option when running a HydroMT command.
For example, to use the ``deltares_data`` catalog with the `hydromt build` command, you can run the following:

.. code-block:: bash

    hydromt build MODEL --dd deltares_data ...

You can specify a version of the catalog by adding the version number after the catalog name, e.g. ``deltares_data=2024.2``.

.. code-block:: bash

    hydromt build MODEL --dd deltares_data=2024.2 ...

Once you have set the data catalog you can specify the datat source(s) for each method in the HydroMT
:ref:`model configuration file <model_config>` as shown in the example below with the `setup_precip_forcing` method.

.. code-block:: yaml

    setup_region:
      region:
        bbox: [4.5, 51.5, 6.5, 53.5]

    setup_maps_from_rasterdataset:
      raster_fn:
        source: 'eobs'
        version: 'v22.0e'



From Python
~~~~~~~~~~~

To use a predefined catalog in Python, you can specify the catalog name with the
``data_libs`` argument when initializing a :py:class:`DataCatalog` class.
You can specify a data catalog version by adding the version number after the
catalog name. You can then get data from the catalog using the
:py:meth:`DataCatalog.get_rasterdataset` or other :ref:`DataCatalog methods <api_data_catalog>`.

.. code-block:: python

    from hydromt import DataCatalog
    data_catalog = DataCatalog(data_libs=["deltares_data"])
    # specify a data catalog version
    data_catalog = DataCatalog(data_libs=["deltares_data=v2024.2"])
    # get data from the catalog
    ds = data_catalog.get_rasterdataset("eobs") # get the most recently added
    ds = data_catalog.get_rasterdataset("eobs", version="22.0e") # get a specific version


Similar when building a model using the :py:class:`Model` class you can specify the
data catalog and version. Subsequently you can use specific data sources for each
model :ref:`setup method <setup_methods>`

.. code-block:: python

    from hydromt import Model
    # initialize a model with a specific data catalog version
    mod = Model(data_libs=["deltares_data=v2024.2"])
    # setup a region and create a map based on eobs orography
    mod.setup_region(region = {'bbox': [4.5, 51.5, 6.5, 53.5]})
    # create a map using the latest version
    mod.setup_maps_from_rasterdataset(
        raster_fn='eobs_orography',
        name="orography_latest",
    )
    # create a map using a specific version
    mod.setup_maps_from_rasterdataset(
        raster_fn={'source': 'eobs_orography',  "version": "22.0e"},
        name="orography_v22.0e",
    )

Available pre-defined data catalogs
-----------------------------------

.. include:: ../_generated/predefined_catalogs.rst
