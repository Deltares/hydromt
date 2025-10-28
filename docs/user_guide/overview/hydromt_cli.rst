.. _hydromt_cli:

HydroMT Command Line Interface
==============================

The HydroMT command line interface (CLI) is a command line tool that allows you to run
HydroMT commands from the terminal. It is installed as part of the HydroMT package.

To use the HydroMT CLI, open a terminal, (activate the environment where HydroMT is installed)
and type ``hydromt`` followed by the command you want to run. you can also run ``hydromt --help``
to get an overview of which commands are available.  The following commands are available:

.. dropdown:: **Information commands**

   - :ref:`hydromt \-\-help <hydromt_help>`
   - :ref:`hydromt \-\-version <hydromt_version>`
   - :ref:`hydromt \-\-models <hydromt_models>`
   - :ref:`hydromt \-\-plugins <hydromt_plugins>`

.. dropdown:: **Model commands**

   - :ref:`hydromt build <hydromt_build>`
   - :ref:`hydromt update <hydromt_update>`

.. dropdown:: **Data catalog commands**

   - :ref:`hydromt export <hydromt_export>`

.. dropdown:: **Validation commands**

   - :ref:`hydromt check <hydromt_check>`

Information commands
--------------------
The base commands or options are here to get some information about HydroMT like the
help, installed version and available models.

.. _hydromt_help:
help
^^^^^
The ``hydromt --help`` command prints the help message for the HydroMT CLI. It shows the available commands and options
that can be used with HydroMT. For example:

.. code-block:: console

    > hydromt --help

    Usage: hydromt [OPTIONS] COMMAND [ARGS]...

    Command line interface for hydromt models.

    Options:
      --version     Show the version and exit.
      --models      Print available model plugins and exit.
      --components  Print available component plugins and exit.
      --plugins     Print available component plugins and exit.
      --help        Show this message and exit.

    Commands:
      build   Build models
      check   Validate config / data catalog / region
      export  Export data
      update  Update models

.. _hydromt_version:
version
^^^^^^^
The ``hydromt --version`` command prints the installed version of HydroMT. For example:

.. code-block:: console

    > hydromt --version

    HydroMT version: 1.0.0

.. _hydromt_models:
models
^^^^^^
The ``hydromt --models`` command prints the available generic models from HydroMT core and
the installed plugins together with their versions. For example:

.. code-block:: console

    > hydromt --models

    Model plugins:
        - model (hydromt 1.3.0)
        - wflow_sbm (hydromt_wflow 1.0.0)
        - wflow_sediment (hydromt_wflow 1.0.0)

.. _hydromt_plugins:
plugins
^^^^^^^
The ``hydromt --plugins`` command prints the installed HydroMT plugins together with their versions.
This includes the model plugins (e.g hydromt_wflow), available pre-defined data catalogs (e.g deltares_data),
available drivers to read different types of data (e.g raster_xarray, geodataset_xarray).
For plugin developers, it also includes available model components and URI resolvers. For example:

.. code-block:: console

    > hydromt --plugins

    Model plugins:
        - model (hydromt 1.0.0)
    Component plugins:
        - ConfigComponent (hydromt 1.0.0)
        - DatasetsComponent (hydromt 1.0.0)
        - GeomsComponent (hydromt 1.0.0)
        - GridComponent (hydromt 1.0.0)
        - ...
    Driver plugins:
        - dataset_xarray (hydromt 1.0.0)
        - geodataframe_table (hydromt 1.0.0)
        - geodataset_vector (hydromt 1.0.0)
        - geodataset_xarray (hydromt 1.0.0)
        - ...
    Catalog plugins:
        - deltares_data (hydromt_data 1.0.0)
        - artifact_data (hydromt_data 1.0.0)
        - aws_data (hydromt_data 1.0.0)
        - gcs_cmip6_data (hydromt_data 1.0.0)
    Uri_resolver plugins:
        - convention (hydromt 1.0.0)
        - raster_tindex (hydromt 1.0.0)

Model commands
--------------

.. _hydromt_build:
build
^^^^^
The ``hydromt build`` command is used to build models from scratch. It has two mandatory
arguments:

- `MODEL`: The name of the model to build. The available models can be printed using the ``hydromt --models`` command.
- `MODEL_ROOT`: Absolute or relative path to the output folder of the model to build.

The ``hydromt build`` command has several options to specify the configuration file, the
region, the data catalog, and other options. The most important ones are:

- `-i, \-\-config`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to prepare for our model (data to use, processing options etc.).
- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to build the model.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt build wflow_sbm /path/to/model_root -i /path/to/wflow_build_config.yml  -d deltares_data  -v

You can find more information on the steps to build a model in the :ref:`Building a model <model_build>` section.
In this section, you will also find how to :ref:`prepare a workflow file <model_workflow>`. To know more about the data catalog, you can refer to the
:ref:`Working with data in HydroMT <get_data>` section.

Finally you can check the :ref:`hydromt build API <build_api>` for all the available options for the build command.

.. _hydromt_update:
update
^^^^^^
The ``hydromt update`` command is used to update an existing model. It is quite similar to the
build command and has two mandatory arguments:

- `MODEL`: The name of the model to update. The available models can be printed using the ``hydromt --models`` command.
- `MODEL_ROOT`: Absolute or relative path to the model to update.

The ``hydromt update`` command has several options to specify the configuration file, the
the data catalog, and other options. The most important ones are:

- `-i, \-\-config`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to update for our model (data to use, processing options etc.).
- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to update the model.
- `-o, \-\-model-out`: Relative or absolute path to the output folder of the updated model. If not provided, the current model will be overwritten.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt update wflow_sbm /path/to/model_to_update -o /path/to/updated_model -i /path/to/wflow_update_config.yml -d /path/to/data_catalog.yml -v

You can find more information on the steps to update a model in the :ref:`Updating a model <model_update>` section.
In this section, you will also find how to :ref:`prepare a workflow file <model_workflow>`. To know more about the data catalog,
you can refer to the :ref:`Working with data in HydroMT <get_data>` section.

Finally you can check the :ref:`hydromt update API <update_api>` for all the available options for the update command.

Data catalog commands
---------------------

.. _hydromt_export:
export
^^^^^^
The ``hydromt export`` command is used to export sample data from a data catalog for
example to export global data for a specific region and time extent.
It has one mandatory argument:

- `EXPORT_DEST_PATH`: Absolute or relative path to the output folder of the exported data.

The input data catalogs are specified using the `-d, \-\-data` option as in the `build` or `update` commands.

There are two ways to specify the sources/extent of the data to export: either fully from the command line or by using a configuration file.

If you are using the command line, the main options are:

- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to export the data.
- `s, --source`: The data source to export. Only one can be specified from the command line.
- `-t, \-\-time`: Set the time extent for which to export the data. The time extent is specified as a list with the start and end date.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt export /path/to/exported_data -d /path/to/data_catalog.yml -s "era5" -t "['2010-01-01', '2010-01-31']" -v

If you want to export several sources or for more options, you can also use a configuration file instead.
In that case, the main options are:

- `-i, \-\-config`: Relative or absolute path to the export configuration file. The export configuration file is a yaml file that contains the sources, region, and time extent to export.

And the command line would look like:

.. code-block:: console

    > hydromt export /path/to/exported_data -i /path/to/export_config.yml -v

An example of the export file is:

.. code-block:: yaml

    export_data:
        data_libs:
            - /path/to/data_catalog.yml
        region:
            bbox: [4.68,53.19,4.69,53.20]
        time_range: ['2010-01-01', '2020-12-31']
        sources:
            - hydro_lakes
            - era5
        unit_conversion: False
        append: False
        meta:
            version: 0.1

You can find detailed document on the function in `hydromt.DataCatalog.export_data <../_generated/hydromt.data_catalog.DataCatalog.export_data.rst>`_.
For the region, only the ``bbox`` and ``geom`` types are supported, see the :ref:`region <region>` section for more information.

Finally you can check the :ref:`hydromt export API <export_api>` for all the available options for the export command.

Validation commands
-------------------

.. _hydromt_check:
check
^^^^^
The ``hydromt check`` command is used to validate the configuration file, and the data catalog.
It can be useful to validate files before running other command lines to avoid errors. Please note that it
will only check the syntax of the files provided. The actual data or calculations referenced will not be checked,
loaded or performed.

The command does not have any required arguments but several options that you can choose from:

- ``-m, --model``: The name of the model to validate. The available models can be printed using the ``hydromt --models`` command.
- ``-i, --config``: Relative or absolute path to the HydroMT configuration file to validate. Note that hydromt v1 cannot validate v0 config files, and vice versa.
- ``-d, --data``: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog to validate.
- ``--format`` specify which format of data catalog to validate. Accepted options are ``v0`` or ``v1``
- ``--upgrade`` when validating ``v0`` data catalogs you can supply the ``--upgrade`` flag, and hydromt will convert the data catalog to the ``v1`` format and write it to a file with the same name but with the suffix ``_v1`` added to the file stem.

Here are some examples of how to use the command:

.. code-block:: console

    > hydromt check -m wflow -i /path/to/wflow_config.yml -d /path/to/data_catalog.yml -v

    > hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v

    > hydromt check -m wflow -i /path/to/wflow_config.yml -v

The validation is so far limited:

- data catalog: only the format and options are validated but it does not try to load any of the data.
- configuration file: it will check if the methods exists and if the correct arguments are called. No validation is done on the content and type of the arguments themselves.

Finally you can check the :ref:`hydromt check API <check_api>` for all the available options for the check command.
