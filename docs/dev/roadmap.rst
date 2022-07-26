Roadmap
=======

Ambition
--------

This package aims to make the building process of any *hydro* model **fast**, **modular** and **reproducible**.

Short-term plans
----------------

Support for lumped, mesh and network models
"""""""""""""""""""""""""""""""""""""""""""
Currently, the Model class implementation requires some regular grid data in the staticmaps attribute. 
We will relax these requirements and implement model classes tailored for grid, lumped, mesh and network models.
More info in https://github.com/Deltares/hydromt/issues/86

Sharing data and support for more data formats
"""""""""""""""""""""""""""""""""""""""""""""""
Currently, many useful datasets have been downloaded and prepared for the Deltares data catalog which is only accessible within the Deltares network.
We will work towards making these download and (if any) the preprocessing of these datasets more transparent.
At the same time we will look for alternative open and analysis ready hosts of these data, such as through the pangeo data initiative.

Furthermore, we will add a new data type for (non spatial) tabular data
To be able to intake more data formats we will support the intake python module, see discussion in https://github.com/Deltares/hydromt/issues/113

*Note: You can find the Deltares data catalog in the* `hydromt artifacts repository <https://github.com/DirkEilander/hydromt-artifacts>`_ *. In this catalog, all links to directly download the data are listed under the "source_url" meta attribute.*

Dashboard / web interface
"""""""""""""""""""""""""
Besides the CLI and Python Interface, we will build a web based interface to HydroMT to make the tool more user friendly.

Integration with other scientific python packages
"""""""""""""""""""""""""""""""""""""""""""""""""
For raster data we aim to replace all duplicate functionality from our raster accessor for xarray Dataset objects with rioxarray and contribute
with new solutions to that package where those fit the rioxarray scope.

Plugins in development
""""""""""""""""""""""
New plugins are currently in preparation for:

- Delft3D-FM: a flexible mesh hydrodynamic model.
- RIBASIM: water allocation model.
- iMOD: groundwater model.
 
If you wish to add new model plugins, please check out the Developer's guide.
