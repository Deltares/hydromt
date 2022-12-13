.. _faq:

Frequently asked questions
==========================

This page contains some FAQ / tips and tricks to work with HydroMT.

Working with models in HydroMT
------------------------------

 | **Q**: Does HydroMT contain any model kernels/software to run model simulations?

HydroMT focusses on the setup of models and analysis of model simulations, but does not contain the model software itself. 
In between the setup and analysis the model software needs to be executed to run a model simulation. 

 | **Q**: Can I re-use the same method when building / updating a model from the command line interface with an .ini configuration file.

Yes, that is possible. You just need to start enumerating the methods by adding a number to the end 
of the method name. For instance, the second time that you use the setup_config method write 
`[setup_config2]` in your .ini file, etc. Note that the actual numbers don't really matter, 
the sequence in which these are in the .ini file determines the sequence in which these are called.

 | **Q**: How can I just write specific :ref:`model data component <model_interface>` 
   (i.e.: grid, geoms, forcing, config or states) instead of the all model data when updating?

Each model plugin implements a combined `write()` method that writes the entire model and is 
called by default at the end of a build or update. If you however add a write method 
(e.g. `write_grid` for a Grid model, `write_forcing`, `write_config`, etc.) to the .ini file the call to the 
general write method is disabled and only the selected model data attributes are written.

 | **Q**: Can I define more than one data catalog when building / updating models?

Yes! You can provide several datasets by repeating the `-d` 
:ref:`option in the command line interface <get_data_cli>`

Working with data in HydroMT
----------------------------

 | **Q**: Does HydroMT contain (global) datasets which can be used to build/update models?

HydroMT does not contain any datasets. A small spatial subset for the Piava basin in northern Italy 
of some data that is often used in combination with HydroMT is made available for testing purposes.
The data will automatically be downloaded to the "~/.hydromt_data" folder on your machine if no 
other data catalogs are provided. See also :ref:`Working with data in HydroMT <get_data>` page.
We are working on creating more data catalogs from (cloud optimized analysis read) open data sources. 

 | **Q**: Can I supply my own data to HydroMT?

Yes, absolutely! Checkout the :ref:`Preparing a data catalog <own_catalog>` page in the user guide.


