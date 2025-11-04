.. _architecture_index:

HydroMT Architecture
====================

Explore HydroMT’s core architecture components and their relationships. Click on each card to jump to the detailed documentation.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: model
        :link-type: ref

        :octicon:`puzzle;5em;sd-text-icon blue-icon`
        +++
        Model
        +++
        Defines the complete model workflow and manages components and data.

    .. grid-item-card::
        :text-align: center
        :link: model_component
        :link-type: ref

        :octicon:`bricks;5em;sd-text-icon blue-icon`
        +++
        ModelComponent
        +++
        Modular building blocks of a model, such as specific datasets or processes.

    .. grid-item-card::
        :text-align: center
        :link: data_catalog
        :link-type: ref

        :octicon:`database;5em;sd-text-icon blue-icon`
        +++
        DataCatalog
        +++
        Core data access layer connecting models and components to datasets.

    .. grid-item-card::
        :text-align: center
        :link: data_source
        :link-type: ref

        :octicon:`file-directory;5em;sd-text-icon blue-icon`
        +++
        DataSource
        +++
        Encapsulates all logic required to retrieve and standardize datasets.

    .. grid-item-card::
        :text-align: center
        :link: uri_resolver
        :link-type: ref

        :octicon:`location;5em;sd-text-icon blue-icon`
        +++
        URIResolver
        +++
        Resolves catalog references to actual file paths or service endpoints.

    .. grid-item-card::
        :text-align: center
        :link: driver
        :link-type: ref

        :octicon:`server;5em;sd-text-icon blue-icon`
        +++
        Driver
        +++
        Reads resolved data into Python objects and handles I/O operations.

    .. grid-item-card::
        :text-align: center
        :link: data_adapter
        :link-type: ref

        :octicon:`gear;5em;sd-text-icon blue-icon`
        +++
        DataAdapter
        +++
        Transforms and standardizes data after reading.

    .. grid-item-card::
        :text-align: center
        :link: plugin_system
        :link-type: ref

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        Extensibility
        +++
        Mechanisms for extending HydroMT with custom classes and plugins.

    .. grid-item-card::
        :text-align: center
        :link: architecture_conventions
        :link-type: ref

        :octicon:`file-code;5em;sd-text-icon blue-icon`
        +++
        Conventions
        +++
        Learn about architecture conventions and design patterns used in HydroMT.

.. toctree::
    :hidden:

    architecture
    design_conventions
