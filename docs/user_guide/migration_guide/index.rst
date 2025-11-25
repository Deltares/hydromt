.. _migration_guide:

Migration Guide
===============

HydroMT is now at version 1.0.0 :octicon:`sparkle-fill;1.5em`

This update introduces several significant changes to the model structure, configuration files, and data handling.
The architecture has been redesigned to enhance flexibility, usability, and performance.
HydroMT is now organized into a component-based architecture to replace the previous inheritance model.
Instead of all model functionality being defined in a single ``Model`` class, a model is now composed of modular ``ModelComponent``
classes such as ``GridComponent``, ``VectorComponent``, or ``ConfigComponent``.
Similarly, the ``DataCatalog`` has been redesigned along ``Driver`` and ``DataAdapter`` classes to allow for
more flexible reading of different data formats and sources and the harmonization of data to standard HydroMT data structures.

This section describes how to migrate HydroMT models and configurations to the newer version of the HydroMT core.
It includes detailed steps, references to updated data structures, and example migration workflows.

It is divided into four main parts:

.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :text-align: center
        :link: data_catalog
        :link-type: doc

        :octicon:`file-moved;5em;sd-text-icon blue-icon`
        +++
        Migrating the Data Catalog

    .. grid-item-card::
        :text-align: center
        :link: model_workflow
        :link-type: doc

        :octicon:`file-moved;5em;sd-text-icon blue-icon`
        +++
        Migrating the model workflow file

    .. grid-item-card::
        :text-align: center
        :link: python_updates
        :link-type: doc

        :octicon:`terminal;5em;sd-text-icon blue-icon`
        +++
        Updates for python users

    .. grid-item-card::
        :text-align: center
        :link: migration_plugin
        :link-type: ref

        :octicon:`database;5em;sd-text-icon blue-icon`
        +++
        Migrating your HydroMT plugin

Users migrating from earlier versions of HydroMT should follow these general steps:

1. Update their HydroMT YAML workflow file to match the v1 schema. (This includes converting `.ini` and `.toml` files to YAML format.)
2. Migrate their data catalog following the updated v1 format.

For python users, you will have to review your scripts and some of the functions calls as some methods have been moved or renamed.
The main changes to the HydroMT python API are documented in the :ref:`Python updates in version 1 <python_updates_v1>`.

For plugin developers, we include a more detailed guide about the architecture and how to change your ``Model`` plugin
to the new component-based structure in the :ref:`Migrating your HydroMT plugin <migration_plugin>` section.

This guide provides the main updates and steps to migrate to HydroMT v1. All detailed changes can be found in the :ref:`changelog <changelog>`

.. toctree::
    :hidden:
    :maxdepth: 1

    Migrating the Data Catalog <data_catalog>
    Migrating the model workflow file <model_workflow>
    Updates for python users <python_updates>
