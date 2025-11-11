.. _architecture_index:

HydroMT Architecture
====================

Explore HydroMT's core architecture components and their relationships. Click on each card to jump to the detailed documentation.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: model_architecture
        :link-type: ref

        :octicon:`log;5em;sd-text-icon blue-icon`
        +++
        Model

    .. grid-item-card::
        :text-align: center
        :link: model_component_architecture
        :link-type: ref

        :octicon:`versions;5em;sd-text-icon blue-icon`
        +++
        ModelComponent

    .. grid-item-card::
        :text-align: center
        :link: data_catalog_architecture
        :link-type: ref

        :octicon:`database;5em;sd-text-icon blue-icon`
        +++
        DataCatalog

    .. grid-item-card::
        :text-align: center
        :link: data_source_architecture
        :link-type: ref

        :octicon:`file-directory;5em;sd-text-icon blue-icon`
        +++
        DataSource

    .. grid-item-card::
        :text-align: center
        :link: uri_resolver_architecture
        :link-type: ref

        :octicon:`location;5em;sd-text-icon blue-icon`
        +++
        URIResolver

    .. grid-item-card::
        :text-align: center
        :link: driver_architecture
        :link-type: ref

        :octicon:`server;5em;sd-text-icon blue-icon`
        +++
        Driver
        +++
        Reads resolved data into Python objects and handles I/O operations.

    .. grid-item-card::
        :text-align: center
        :link: data_adapter_architecture
        :link-type: ref

        :octicon:`gear;5em;sd-text-icon blue-icon`
        +++
        DataAdapter

    .. grid-item-card::
        :text-align: center
        :link: register_plugins
        :link-type: ref

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        Extensibility

    .. grid-item-card::
        :text-align: center
        :link: conventions
        :link-type: doc

        :octicon:`file-code;5em;sd-text-icon blue-icon`
        +++
        Conventions

.. toctree::
    :hidden:

    architecture
    conventions
