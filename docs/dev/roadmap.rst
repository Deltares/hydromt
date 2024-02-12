Roadmap
=======

Ambition
--------

This package aims to make the building process of any *hydro* model **fast**, **modular** and **reproducible**.

We always have ideas about how the package could be `improved <https://github.com/Deltares/hydromt/labels/enhancement>`_ in the future and welcome
contribution from the (hydro) geoscientific community. If you would like to help with any of these developments or have ideas, consider reading the :ref:`developer's guide <contributing>`.

Short-term plans
----------------

Support for vector, mesh and network models
"""""""""""""""""""""""""""""""""""""""""""
Before version 0.6.0, the Model class implementation assumed regular grid data in the staticmaps attribute (grid attribute of the Grid model from v0.6.0).
These requirements have been relaxed from v0.6.0 by implementing model classes tailored for grid, vector, mesh and network models. Advanced testing and submodel specific attributes
and generic workflows to support specific submodel class methods are still missing and are planned to be added in future releases.

Sharing data and support for more data formats
"""""""""""""""""""""""""""""""""""""""""""""""
Currently, many useful datasets have been downloaded and prepared for the Deltares data catalog which is only accessible within the Deltares network.
We are working towards making these download and (if any) the preprocessing of these datasets more transparent.
At the same time we are looking for alternative open and analysis ready hosts of these data, such as through the pangeo data initiative.

To be able to intake more data formats we will support the intake python module, see discussion in https://github.com/Deltares/hydromt/issues/113

Dashboard / web interface
"""""""""""""""""""""""""
Besides the CLI and Python Interface, we are building a web based interface to HydroMT to make the tool more user friendly.


Plugins in development
""""""""""""""""""""""
New plugins are currently in preparation for:

- Delft3D-FM: a flexible mesh hydrodynamic model.
- RIBASIM: water allocation model.
- iMOD: groundwater model.

If you wish to add new model plugins, please check out the :ref:`Developer's guide <contributing>`.
