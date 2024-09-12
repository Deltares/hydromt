.. _custom_components:

Custom components
=================

Components are a large part of how HydroMT defines its behavior.
A component is basically a part of the model architecture that is responsible
for handling a specific part of the data, such as the Grid, Config, or Mesh
but also potentially something like rivers, catchments, pipes or other custom behavior
your plugin might need.



You can do this in a similar way to how you define your custom Model or
ModelComponents. We'll assume you have a custom data catalog called
`PluginDataCatalog.yml` that you want to make available to your plugin users.

Firstly you should make a `PluginDataCatalog` class which inherits from `PredefinedCatalog`. This will ensure that HydroMT can find your catalog
and also know how to interact with it. The base `PredefinedCatalog` makes use
of the library pooch to fetch the correct data catalogs. Please refer to it's documentation for more information on how to use it: https://www.fatiando.org/pooch/dev/
This class will help fetch the correct files for the data catalog. The yaml
file specifying the data catalog should be the same format as any other catalog.

After you've implemented this class you should just have to register it's
entry point in your `pyproject.toml` like so:

```toml
[project.entry-points."hydromt.data.PredefinedCatalog"]
my_plugin_catalog = "hydromt_plugin.path.to.catalog.class:PluginDataCatalog"

```

You might have to make sure that both hydromt and your plugin are installed before
it will discover your plugin. After you should be able to verify that your catalog
is discovered correctly from the command line:

```sh
hydromt --plugins
```

The output should then contain the name of your class
