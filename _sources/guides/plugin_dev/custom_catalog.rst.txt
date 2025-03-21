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

If you structure your data catalog and their versioning just like hydromt core does,
then the base class doesn't actually have to do much more than point to a different
repository by overriding the `base_url` class variable. As long as the url points to a
correctly configured repository everything should work automatically. Please refer to
the HydroMT core and pooch documentation for more information.
