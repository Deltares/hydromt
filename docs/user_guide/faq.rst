.. _faq:

Frequently asked questions
==========================

This page contains some FAQ / tips and tricks to work with HydroMT.

Working with models in HydroMT
------------------------------

 | **Q**: Does HydroMT contain any model kernels/software to run model simulations?

HydroMT focusses on the setup of models and analysis of model simulations, but does not contain the model software itself.
In between the setup and analysis the model software needs to be executed to run a model simulation.

 | **Q**: Can I re-use the same method when building / updating a model from the command line interface with an .yaml configuration file.

Yes, that is possible. In the YAML file, each setup or update method is listed under the steps section.
So you can easily repeat the same method multiple times with different arguments. For example:

.. code-block:: yaml

    steps:
      - config.update:
          forcing_file: era5_2010.nc
      - setup_precip_forcing:
          precip_data: "era5"
          start: 2010-01-01
          end: 2010-12-31
      - forcing.write:
      - config.update:
          forcing_file: chirps_2010.nc
      - setup_precip_forcing:
          precip_data: "chirps"
          start: 2010-01-01
          end: 2010-12-31
      - forcing.write:

Here ``config.update``, ``setup_precip_forcing`` and ``forcing.write`` are called several times to
create several forcing files in one go.

 | **Q**: How can I just write specific model data component
   (i.e.: grid, geoms, forcing, config or states) instead of the all model data when updating?

Each model plugin implements a combined ``write()`` method that writes the entire model and is
called by default at the end of a ``build`` or ``update``. If you however add a write method
(e.g. ``grid.write`` for a Grid model, ``forcing.write``, ``config.write``, etc.) to the .yaml file the call to the
general write method is disabled and only the selected model data attributes are written.

Working with data in HydroMT
----------------------------

 | **Q**: Does HydroMT contain (global) datasets which can be used to build/update models?

HydroMT does not contain any datasets. A small spatial subset for the Piave basin in northern Italy
of some data that is often used in combination with HydroMT is made available for testing purposes.
The data will automatically be downloaded to the "~/.hydromt" folder on your machine if no
other data catalogs are provided. See also :ref:`Working with data in HydroMT <get_data>` page.
We are working on creating more data catalogs from (cloud optimized analysis read) open data sources.

 | **Q**: Can I supply my own data to HydroMT?

Yes, absolutely! Checkout the :ref:`Preparing a data catalog <own_catalog>` page in the user guide.
