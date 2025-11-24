.. currentmodule:: hydromt

.. _plugin_quickstart:

================================
Starting Your Own HydroMT Plugin
================================

You've discovered HydroMT and would like to use it for your own model — but you're wondering how to get started?
This page provides guidance, best practices, and examples for creating your own HydroMT plugin.
Before reading this, make sure you're familiar with HydroMT by reviewing at least the :ref:`intro_user_guide` and the
:ref:`Working with models <model_main>` sections.

.. note::

  Most of this section focuses on creating *new* functionality.
  If you already have a plugin and want to migrate it to the new V1 architecture, please refer to the :ref:`migration_plugin` guide.

The new V1 architecture of HydroMT provides much greater flexibility to customize behavior —
for example, by adding your own ``ModelComponents`` without needing a plugin.
However, there are still many cases where creating a plugin is the right approach, including:

* Reading or writing custom data formats
* Model specific data processing or parameter estimation methods
* Implementing new components
* Providing standardized data catalogs for others to use

If you plan to do any of the above, and your work might be useful to others,
then creating a plugin is the right choice.

HydroMT makes it straightforward to create your own plugin.
A plugin can customize or introduce new implementations of the following objects:

* ``Model``
* ``ModelComponent``
* ``Resolver``
* ``Driver``
* ``Catalog``

.. _plugin_create:

Creating Your Plugin Repository
-------------------------------

See :ref:`register_plugins` for detailed information on how to ensure HydroMT discovers your plugin.
The main requirement is that your plugin is a **Python package** that registers one or more of the supported entry points
and inherits from the appropriate HydroMT base class.

Your plugin does *not* need to be written entirely in Python — but if you use another language,
you'll need to create Python bindings. While this can improve performance, it adds complexity,
so if you're unsure, we recommend sticking with pure Python.


.. _plugin_components:

Typical HydroMT Repository Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical HydroMT plugin repository follows a clear and consistent structure.
This layout is **not mandatory**, but is commonly used across HydroMT and its plugins.
The subfolders inside your main package directory (e.g., ``hydromt_awesome/``) are entirely optional —
as long as your package is a valid Python package, registers your custom classes as plugins,
and follows the conventions and API described in the HydroMT documentation.

.. code-block:: text

    hydromt_awesome/
    ├── docs/                     # Documentation pages and source files
    ├── examples/                 # Example models, templates, and notebooks
    ├── tests/                    # Unit and integration tests
    ├── pyproject.toml            # Build-system configuration (for PyPI/Conda)
    ├── README.md                 # Main project overview and usage guide
    ├── LICENSE                   # License information
    └── hydromt_awesome/          # Main Python package for your plugin
        ├── __init__.py           # Package initializer
        ├── model/                # Custom HydroMT Model class implementations
        ├── components/           # Custom ModelComponents
        ├── catalogs/             # Predefined or plugin-specific DataCatalogs
        ├── resolvers/            # Custom URIResolvers for data discovery
        ├── drivers/              # Custom Drivers for new data formats
        └── adapters/             # Custom DataAdapters for data transformation

This structure helps maintain clarity and organization as your plugin grows in complexity.
By organizing your repository in this way, contributors and users can quickly understand where to find each part of your plugin.

See also the :ref:`Implement your own HydroMT objects <plugin_examples>` section for concrete examples and steps of how to implement
your own objects (Model, ModelComponent, Resolver, Driver, Catalog) in a HydroMT plugin.
