.. currentmodule:: hydromt
.. _model_api:

=========
Model API
=========

Attributes
----------

.. autosummary::
   :toctree: generated/

   Model.region
   Model.crs
   Model.res
   Model.root
   Model.config
   Model.staticmaps
   Model.staticgeoms
   Model.forcing

High level methods
------------------

.. autosummary::
   :toctree: generated/

   Model.read
   Model.write
   Model.build
   Model.set_root

General methods
---------------

.. autosummary::
   :toctree: generated/

   Model.setup_config
   Model.get_config
   Model.set_config
   Model.read_config
   Model.write_config

   Model.set_staticmaps
   Model.read_staticmaps
   Model.write_staticmaps

   Model.set_staticgeoms
   Model.read_staticgeoms
   Model.write_staticgeoms

   Model.set_forcing
   Model.read_forcing
   Model.write_forcing