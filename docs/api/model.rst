.. currentmodule:: hydromt.model

.. _model_api:

=====
Model
=====


Model class
===========



High-level methods
------------------

.. autosummary::
   :toctree: ../_generated

   Model
   Model.read
   Model.write
   Model.write_data_catalog

General methods
---------------

.. autosummary::
   :toctree: ../_generated

   Model.build
   Model.update
   Model.get_component
   Model.add_component
   Model.test_equal
   Model.__enter__
   Model.__exit__

Model attributes
----------------

.. autosummary::
   :toctree: ../_generated

   Model.data_catalog
   Model.crs
   Model.root
   Model.region
   Model.components

ModelRoot
=========

.. autosummary::
   :toctree: ../_generated

   ModelRoot

Attributes
----------

.. autosummary::
   :toctree: ../_generated

   ModelRoot.mode
   ModelRoot.is_writing_mode
   ModelRoot.is_reading_mode
   ModelRoot.is_override_mode

General Methods
---------------

.. autosummary::
   :toctree: ../_generated

   ModelRoot.set
