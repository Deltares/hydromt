.. _intro_plugin_guide:

Plugin Developer Guide
======================

This guide provides step-by-step instructions on building, registering, and testing your own HydroMT plugin — from models and components to data catalogs and resolvers.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: plugin_quickstart
        :link-type: ref

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        Starting your own HydroMT Plugin
        +++
        Overview of creating your first plugin, repository structure, and registering with HydroMT.

    .. grid-item-card::
        :text-align: center
        :link: register_plugins
        :link-type: ref

        :octicon:`plug;5em;sd-text-icon blue-icon`
        +++
        Linking your own custom objects to HydroMT core API
        +++
        Instructions on registering entry points in `pyproject.toml` for Models, Components, Drivers, Resolvers, and Catalogs. # TODO: Add best practices for naming conventions

    .. grid-item-card::
        :text-align: center
        :link: test_your_plugin
        :link-type: ref

        :octicon:`check-circle;5em;sd-text-icon blue-icon`
        +++
        Testing your plugin
        +++
        Guidance on unit testing model components, models, and complete plugins. # TODO: Include example CI workflow

.. toctree::
   :hidden:

   quickstart
   Implement your own <custom_implementation/index>
   workflows
   testing
   migrating_to_v1
