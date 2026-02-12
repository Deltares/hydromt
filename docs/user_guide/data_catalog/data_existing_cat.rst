
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

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        To use a predefined catalog, you can specify the catalog name with the ``-d`` or ``--data`` option when running a HydroMT command.
        For example, to use the ``deltares_data`` catalog with the `hydromt build` command, you can run the following:

        .. code-block:: bash

            hydromt build MODEL -d deltares_data ...

        Alternatively, deltares_data can also be accessed with the ``--dd`` option:

        .. code-block:: bash

            hydromt build MODEL --dd ...


        You can specify a version of the catalog by adding the version number after the catalog name, e.g. ``deltares_data=v1.0.0``.

        .. code-block:: bash

            hydromt build MODEL -d deltares_data=v1.0.0 ...

        Once you have set the data catalog you can specify the data source(s) for each method in the HydroMT
        :ref:`model workflow file <model_workflow>` as shown in the example below with the `setup_precip_forcing` method.

        .. code-block:: yaml

            steps:
              - setup_region:
                  region:
                  bbox: [4.5, 51.5, 6.5, 53.5]

              - setup_maps_from_rasterdataset:
                  raster_fn:
                    source: 'eobs'
                    version: 'v22.0e'

    .. tab-item:: Python API

        To use a predefined catalog in Python, you can specify the catalog name with the
        ``data_libs`` argument when initializing a :py:class:`DataCatalog` class.
        You can specify a data catalog version by adding the version number after the
        catalog name. You can then get data from the catalog using the
        :py:meth:`DataCatalog.get_rasterdataset` or other :ref: `DataCatalog methods`.

        .. code-block:: python

            from hydromt import DataCatalog
            data_catalog = DataCatalog(data_libs=["deltares_data"])
            # specify a data catalog version
            data_catalog = DataCatalog(data_libs=["deltares_data=v1.0.0"])
            # get data from the catalog
            ds = data_catalog.get_rasterdataset("eobs") # get the most recently added
            ds = data_catalog.get_rasterdataset("eobs", version="22.0e") # get a specific version

Available pre-defined data catalogs
-----------------------------------

Deltares data catalog
^^^^^^^^^^^^^^^^^^^^^
Data available for Deltares colleagues (p: drive). For non Deltares users, you can use it as inspiration to create your own. The catalog and it's different versions can be viewed here: https://github.com/Deltares/hydromt/tree/main/data/catalogs/deltares_data

Available data:

.. include:: ../../_generated/deltares_data.rst

Artifact data catalog
^^^^^^^^^^^^^^^^^^^^^
Global data extract around the Piave basin in Northern Italy used for documentation, training and testing of HydroMT. The catalog and its different versions can be viewed here: https://github.com/Deltares/hydromt/tree/main/data/catalogs/artifact_data

Available data:

.. include:: ../../_generated/artifact_data.rst

AWS data catalog
^^^^^^^^^^^^^^^^
Data openly available in Amazon s3 bucket. The catalog and its different versions can be viewed here: https://github.com/Deltares/hydromt/tree/main/data/catalogs/aws_data

Available data:

.. include:: ../../_generated/aws_data.rst

GCS CMIP6 data catalog
^^^^^^^^^^^^^^^^^^^^^^
CMIP6 dataset openly available and stored on a public Google Cloud Store. The catalog and its different versions can be viewed here: https://github.com/Deltares/hydromt/tree/main/data/catalogs/gcs_cmip6_data

Available data:

.. include:: ../../_generated/gcs_cmip6_data.rst

Earth Data Hub data catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data stored in `Earth Data Hub <https://earthdatahub.destine.eu/>` (Destination Earth). In order to use this  catalog, you need to setup credentials for accessing the data on Earth Data Hub.
This includes creating an account on Earth Data Hub and setting up a .netrc file with your credentials.
You can find more information on how to do this in the `Earth Data Hub documentation <https://earthdatahub.destine.eu/getting-started#configuring-netrc>`_.

The catalog and its different versions can be viewed here: https://github.com/Deltares/hydromt/tree/main/data/catalogs/earthdatahub_data

Available data:

.. include:: ../../_generated/earthdatahub_data.rst
