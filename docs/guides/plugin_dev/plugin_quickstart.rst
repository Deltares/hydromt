.. currentmodule:: hydromt

.. _plugin_quickstart:

================================
Starting your own HydroMT Plugin
================================

You've discovered HydroMT and would like to use it for your own model and wonder how to get started about it?
This page gives you some tips and tricks on how start your own HydroMT plugin. Before reading this,
you should already know a little about HydroMT, so please check out at least the :ref:`intro_user_guide` and the :ref:`model_main`
section.

.. NOTE::

  Most of this section of the docs is dedicated to creating *new* functinoality. If you
  already have a plugin and want to know how to port that to the new V1 archetecture
  please refer to :ref:`migrating_to_v1` specifically.

The new V1 archetecture of HdydroMT already offers a lot more flexibility to customise
the behaviour of HydroMT without the need for a plugin, by adding `ModelComponents`
however there are still plenty of use cases you might want to make a plugin, including
but not limited to:

* reading or writing custom data formats
* Implementing new components
* providing standard data catalogs for others to use

If you want to do any of the above, and there is a decent chance that others might want
to use your code, then you have a good case for a plugin!


.. _plugin_create:

Create your own plugin repository
---------------------------------
HydroMT makes it pretty easy to create your own plugin. You can create a plugin to
customise or create new variants of any of the following objects:

* Model
* ModelComponent
* Resolvers
* Driver
* Catalog

Please see `_register_plugins` for more detail on how to make sure that HydroMT will
find your implementation. Currently the only real requirement for making a HydroMT
plugin is that it is a Python Package that uses one of the above entrypoints and
inherets from the appropriate base class so hydroMT will know how to operate it.

This does not mean that your plugin must necessarily be written entirely in python, but
if it isn't you will have to make python bindings for it. This can help with performance
but is also quite complex, so anyone unsure how to go about this, we recommend using
Python.

.. _plugin_components:

Typical HydroMT repository structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With HydroMT and some of its plugins, we usually use a classic folder structure and file organisation. It not mandatory to follow this structure
But if you choose to, a classic folder structure and files for a HydroMT plugin (eg mymodel) looks like this:

- **docs**: folder containing your documentation pages.
- **examples**: folder containing example models, templates for building/updating models, jupyter notebooks.
- **tests**: folder containing your test scripts.
- *pyproject.toml*: your build-system requirements for your python package (used by pypi and conda).
- *README.md*: your landing documentation page for your repository.
- *LICENSE*: license file for your repository.
- **hydromt_awesome**: folder containing the functions for your plugin.

  - *__init__.py*: init python script used when importing the *hydromt_awesome* package
    in a python script.
  - submodules for the objects you wish to expose to hydromt such as:
    - **resolvers**
    - **catalogs**
    - **model**
    - **components**
    - **drivers**

Please refer to the specific section of each of these objects for more information about
their specific requirements and uses.
