.. _custom_catalog:

Custom Catalog
==============

It is likely that you have specific data that is often used with your plugin
for this reason HydroMT offers an entrypoint so that you can extend the default
Deltares data catalog, and provide your users with often used data.

Firstly you should make a `PluginDataCatalog` class which inherits from `PredefinedCatalog`. This will ensure that HydroMT can find your catalog
and also know how to interact with it. The base `PredefinedCatalog` makes use
of the library pooch to fetch the correct data catalogs. Please refer to it's documentation for more information on how to use it: https://www.fatiando.org/pooch/dev/
This class will help fetch the correct files for the data catalog. The yaml
file specifying the data catalog should be the same format as any other catalog.

You might have to make sure that both hydromt and your plugin are installed before
it will discover your plugin. After you should be able to verify that your catalog
is discovered correctly from the command line:

```sh
hydromt --plugins
```

The output should then contain the name of your class
