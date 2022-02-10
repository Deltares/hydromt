.. currentmodule:: hydromt

#############
Workflows
#############


Basin mask
==========

.. autosummary::
   :toctree: ../_generated

   workflows.basin_mask.get_basin_geometry
   workflows.basin_mask.parse_region

River bathymetry
================

.. autosummary::
   :toctree: ../_generated

   workflows.rivers.river_width
   workflows.rivers.river_depth


Forcing
=======

Data handling
-------------

.. autosummary::
   :toctree: ../_generated

   workflows.forcing.precip
   workflows.forcing.temp
   workflows.forcing.press
   workflows.forcing.pet


Correction methods
------------------

.. autosummary::
   :toctree: ../_generated

   workflows.forcing.press_correction
   workflows.forcing.temp_correction

Time resampling methods
-----------------------

.. autosummary::
   :toctree: ../_generated

   workflows.forcing.resample_time
   workflows.forcing.delta_freq

Computation methods
-------------------

**PET**

.. autosummary::
   :toctree: ../_generated

   workflows.forcing.pet_debruin
   workflows.forcing.pet_makkink

