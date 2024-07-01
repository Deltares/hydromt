.. _hydromt_cli:

HydroMT Command Line Interface
==============================

The HydroMT command line interface (CLI) is a command line tool that allows you to run
HydroMT commands from the terminal. It is installed as part of the HydroMT package. To
use the CLI, you need to have HydroMT installed in your Python environment. If you have
not installed HydroMT yet, please refer to the
:ref:`Installation <installation_guide>` section.

To use the HydroMT CLI, open a terminal and type `hydromt` followed by the command you
want to run. The following commands are available:

- **hydromt \-\-help**: Print help on HydroMT and the different commands.
- **hydromt \-\-version**: Print the version of HydroMT.
- **hydromt \-\-models**: Print the available model from HydroMT and the installed plugins.
- **hydromt build**: Build models.
- **hydromt update**: Update models.
- **hydromt clip**: Clip models.
- **hydromt export**: Export data.
- **hydromt check**: Validate config / data catalog / region.

Example for printing the help of HydroMT (do not forget to activate the python environment
where HydroMT is installed):

.. code-block:: console

    > hydromt --help

    usage: hydromt [OPTIONS] COMMAND [ARGS]...

    Command line interface for hydromt models.

    Options:
    --version  Show the version and exit.
    --models   Print available model plugins and exit.
    --help     Show this message and exit.

    Commands:
    build   Build models
    check   Validate config / data catalog / region
    clip    Clip models
    export  Export data
    update  Update models

Base commands
-------------
The base commands or options are here to get some information about HydroMT like the
help, installed version and available models.

Help command
^^^^^^^^^^^^

The `hydromt \-\-help` command prints the help of HydroMT and the different commands,
such as build. For the general help, you can see the example above. To get help for
a specific command, you can type `hydromt <command> \-\-help`. For example, to get help
for the build command, you can type:

.. code-block:: console

    > hydromt build --help

    Usage: hydromt build [OPTIONS] MODEL MODEL_ROOT

    Build models from scratch.

    Example usage:
    --------------

    To build a wflow model for a subbasin using a point coordinates snapped to
    cells with upstream area >= 50 km2 hydromt build wflow /path/to/model_root
    -i /path/to/wflow_config.ini  -r "{'subbasin': [-7.24, 62.09], 'uparea':
    50}" -d deltares_data -d /path/to/data_catalog.yml -v

    To build a sfincs model based on a bbox hydromt build sfincs
    /path/to/model_root  -i /path/to/sfincs_config.ini  -r "{'bbox':
    [4.6891,52.9750,4.9576,53.1994]}"  -d /path/to/data_catalog.yml -v

    Options:
    --opt TEXT               Method specific keyword arguments, see the method
                            documentation of the specific model for more
                            information about the arguments.
    -i, --config PATH        Path to hydroMT configuration file, for the model
                            specific implementation.
    -r, --region TEXT        Set the region for which to build the model, e.g.
                            {'subbasin': [-7.24, 62.09]}
    -d, --data TEXT          Path to local yaml data catalog file OR name of
                            predefined data catalog.
    --dd, --deltares-data    Flag: Shortcut to add the "deltares_data" catalog
    --fo, --force-overwrite  Flag: If provided overwrite existing model files
    --cache                  Flag: If provided cache tiled rasterdatasets
    -v, --verbose            Increase verbosity.
    -q, --quiet              Decrease verbosity.
    --help                   Show this message and exit.


Version command
^^^^^^^^^^^^^^^
The `hydromt \-\-version` command prints the version of HydroMT. For example:

.. code-block:: console

    > hydromt --version

    HydroMT version 0.9.2

Models command
^^^^^^^^^^^^^^
The `hydromt \-\-models` command prints the available generic models from HydroMT core and
the installed plugins together with their versions. For example:

.. code-block:: console

    > hydromt --models

    model plugins:
    - wflow (hydromt_wflow 0.4.1)
    - wflow_sediment (hydromt_wflow 0.4.1)
    generic models (hydromt 0.9.2):
    - grid_model
    - vector_model
    - network_model

Build command
-------------
The **hydromt build** command is used to build models from scratch. It has two mandatory
arguments:

- `MODEL`: The name of the model to build. The available models can be printed using the `hydromt \-\-models` command.
- `MODEL_ROOT`: Absolute or relative path to the output folder of the model to build.

The **hydromt build** command has several options to specify the configuration file, the
region, the data catalog, and other options. The most important ones are:

- `-i, \-\-config`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to prepare for our model (data to use, processing options etc.).
- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to build the model. In order to use several catalogs, you can use the `-d` option several times.
- `-r, \-\-region`: Set the region for which to build the model. The region is specified as a dictionary with the region type and the region coordinates.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt build wflow /path/to/model_root -i /path/to/wflow_build_config.yml -r "{'subbasin': [-7.24, 62.09], 'uparea': 50}" -d deltares_data -d /path/to/data_catalog.yml -v

You can find more information on the steps to build a model in the :ref:`Building a model <model_build>` section.
In this section, you will also find how to :ref:`prepare a configuration file <model_config>` and a
:ref:`region <region>`. To know more about the data catalog, you can refer to the
:ref:`Working with data in HydroMT <get_data>` section.

Finally you can check the :ref:`hydromt build API <build_api>` for all the available options for the build command.

Update command
--------------
The **hydromt update** command is used to update an existing model. It is quite similar to the
build command and has two mandatory arguments:

- `MODEL`: The name of the model to update. The available models can be printed using the `hydromt \-\-models` command.
- `MODEL_ROOT`: Absolute or relative path to the model to update.

The **hydromt update** command has several options to specify the configuration file, the
the data catalog, and other options. The most important ones are:

- `-i, \-\-config`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to update for our model (data to use, processing options etc.).
- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to update the model. In order to use several catalogs, you can use the `-d` option several times.
- `-o, \-\-model-out`: Relative or absolute path to the output folder of the updated model. If not provided, the current model will be overwritten.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt update wflow /path/to/model_to_update -o /path/to/updated_model -i /path/to/wflow_update_config.yml -d /path/to/data_catalog.yml -v

You can find more information on the steps to update a model in the :ref:`Updating a model <model_update>` section.
In this section, you will also find how to :ref:`prepare a configuration file <model_config>`. To know more about the data catalog,
you can refer to the :ref:`Working with data in HydroMT <get_data>` section.

Finally you can check the :ref:`hydromt update API <update_api>` for all the available options for the update command.

Clip command
------------
The **hydromt clip** command is used to clip an existing model. It has four mandatory arguments:

- `MODEL`: The name of the model to clip. The available models can be printed using the `hydromt \-\-models` command.
- `MODEL_ROOT`: Absolute or relative path to the model to clip.
- `MODEL_DESTINATION`: Absolute or relative path to the output folder of the clipped model.
- `REGION`: Set the region for which to clip the model. The region is specified as a dictionary with the region type and the region coordinates.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt clip wflow /path/to/model_to_clip /path/to/clipped_model "{'subbasin': [4.68,53.19], 'wflow_uparea': 50}" -v

You can find more information on the steps to clip a model in the :ref:`Clipping a model <model_clip>` section.
In this section, you will also find how to :ref:`prepare a region <region>`.

Finally you can check the :ref:`hydromt clip API <clip_api>` for all the available options for the build command.

Note that this function is not available for all models and you should check the plugin documentation for more information.

Export command
--------------
The **hydromt export** command is used to export sample data from a data catalog for
example to export global data for a specific region and time extent. It is based
on the `hydromt.DataCatalog.export_data` function.
It has one mandatory argument:

- `EXPORT_DEST_PATH`: Absolute or relative path to the output folder of the exported data.

The input data catalogs are specified using the `-d, \-\-data` option as in the `build` or `update` commands.
You are also free to specify several catalogs.

There are two ways to specify the sources/extent of the data to export: either fully from the command line or by using a configuration file.

If you are using the command line, the main options are:

- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to export the data. In order to use several catalogs, you can use the `-d` option several times.
- `s, --source`: The data source to export. Only one can be specified from the command line.
- `-r, \-\-region`: Set the region for which to export the data. The region is specified as a dictionary with the region type (here only `bbox` or `geom`) and the region coordinates. Please note that the `bbox` coordinates should be in `EPSG:4326`.
- `-t, \-\-time`: Set the time extent for which to export the data. The time extent is specified as a list with the start and end date.

Here is an example of how to use the command:

.. code-block:: console

    > hydromt export /path/to/exported_data -d /path/to/data_catalog.yml -s "era5" -r "{'bbox': [4.68,53.19,4.69,53.20]}" -t "['2010-01-01', '2010-01-31']" -v

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
For the region, only the `bbox` and `geom` types are supported, see the :ref:` region <region>` section for more information.

Finally you can check the :ref:`hydromt export API <export_api>` for all the available options for the export command.

Check command
-------------
The **hydromt check** command is used to validate the configuration file, the data catalog, and the region.
It can be useful to validate files before running other command lines to avoid errors. Please note that it
will only check the syntax of the files provided. The actual data or calculations referenced will not be checked,
loaded or performed.

The command does not have any required arguments but several options that you can choose from:

- `-m, \-\-model`: The name of the model to validate. The available models can be printed using the `hydromt \-\-models` command.
- `-i, \-\-config`: Relative or absolute path to the HydroMT configuration file to validate.
- `-d, \-\-data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog to validate.
- `-r, \-\-region`: Set the region to validate (only `bbox` and `geom` types supported). The region is specified as a dictionary with the region type and the region coordinates. Please note that the `bbox` coordinates should be in `EPSG:4326`.

If you only wish to validate a region or a data catalog files, they can be run separately.
Because configuration files are dependant on the model to build/update, to check a configuration file, you need to specify the model.

Here are some examples of how to use the command:

.. code-block:: console

    > hydromt check -m wflow -i /path/to/wflow_config.yml -d /path/to/data_catalog.yml -r "{'bbox': [4.68,53.19,4.69,53.20]}" -v

    > hydromt check -d /path/to/data_catalog.yml -v

    > hydromt check -m wflow -i /path/to/wflow_config.yml -v

    > hydromt check -r "{'bbox': [4.68,53.19,4.69,53.20]}" -v

The validation is so far limited:

- region: full validation is only supported for the `bbox` and `geom` types. For other type, it will only check if the type is supported but will not validate the region itself.
- data catalog: only the format and options are validated but it does not try to load any of the data.
- configuration file: it will check if the methods exists and if the correct arguments are called. No validation is done on the content and type of the arguments themselves.

Finally you can check the :ref:`hydromt check API <check_api>` for all the available options for the check command.
