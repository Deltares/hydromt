.. _plugin_examples:

==============================
Examples: Extending HydroMT
==============================

This section provides detailed examples of how to extend HydroMT for your own plugin.
Each page describes one aspect of customization, from implementing a new model class to defining data catalogs and resolvers.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: custom_model_builder
        :link-type: ref

        :octicon:`puzzle;5em;sd-text-icon blue-icon`
        +++
        Implementing your own HydroMT Model class
        +++
        Guidance on building custom models, setup methods, and component integration. # TODO: Add example diagram or schematic

    .. grid-item-card::
        :text-align: center
        :link: custom_components
        :link-type: ref

        :octicon:`bricks;5em;sd-text-icon blue-icon`
        +++
        Creating your own Model Components
        +++
        Instructions for implementing `ModelComponent` and `SpatialModelComponent` with initialization, required attributes, and workflow integration. # TODO: Add sample code snippets

    .. grid-item-card::
        :text-align: center
        :link: custom_catalog
        :link-type: ref

        :octicon:`database;5em;sd-text-icon blue-icon`
        +++
        Pre-defined Data Catalog for your plugin
        +++
        How to define a `PluginDataCatalog` and structure datasets for HydroMT. # TODO: Include YAML example

    .. grid-item-card::
        :text-align: center
        :link: custom_driver
        :link-type: ref

        :octicon:`server;5em;sd-text-icon blue-icon`
        +++
        Custom Data Driver
        +++
        Implement custom drivers to read/write non-standard datasets. # TODO: Add advanced examples for combined usage

    .. grid-item-card::
        :text-align: center
        :link: custom_resolver
        :link-type: ref

        :octicon:`server;5em;sd-text-icon blue-icon`
        +++
        Custom Resolver
        +++
        Steps to create a custom URIResolver for unique data discovery needs. # TODO: Provide common use cases



.. toctree::
    :hidden:

    model
    component
    catalog
    driver
    resolver
