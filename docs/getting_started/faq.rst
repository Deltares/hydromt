.. _faq:

Frequently asked questions
==========================

This page contains some FAQ / tips and tricks to work with HydroMT.

**Q**: Does HydroMT contain any model kernels/software to run model simulations?

 | HydroMT focusses on the setup and analysis of models and does not contain the model software. 
   In between the setup and analysis the model software needs to be executed to create model output. 

**Q**: Does HydroMT contain (global) datasets which I can use to build models?

 | HydroMT does not contain any datasets. A small spatial subset for the Piava basin in northern Italy 
   of some data that is often used in combination with HydroMT is made available for testing purposes.
   The data will automatically be downloaded to the "~/.hydromt_data" folder on your machine if no 
   other data catalogs are provided. See also :ref:`Working with data in HydroMT <get_data>` page.
   We are working on creating more data catalogs from (cload optimized analysis read) open data sources. 

**Q**: Can I supply my own data to HydroMT?

 | Yes, absolutely! The :ref:`data catalog <own_catalog>` for just that purpose!

**Q**: Can I define more than one data catalog when building / updating models?

 | Yes! You can provide several datasets by repeating the `-d` 
   :ref:`option in the command line interface <get_data_cli>`

**Q**: Can I re-use the same model component when building / updating a model from the command line interface with an .ini configuration file.

 | Yes, that is possible. You just need to start enumerating the components by adding a number to the end 
   of the component name. For instance, the second time that you use the setup_config component write 
   `[setup_config2]` in your .ini file, etc. Note that the actual numbers don't really matter, 
   the sequence in which these are in the .ini file determines the sequence in which these are called.

**Q**: How can I just write a specific model data attribute of my model (i.e.: staticmaps, staticgeoms, forcing, config or states) instead of the whole model when updating?

 | Each model plugin implements a combined `write()` method that writes the entire model and is 
   called by default at the end of the build or update workflow. If you however add a write method 
   (e.g. `write_staticmaps`, `write_forcing`, `write_config`, etc.) to the .ini file the call to the 
   general write method is disabled and only the selected model data attributes are written..  
