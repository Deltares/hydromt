hydroMT: Build and analyze hydro models
#######################################

.. image:: https://codecov.io/gh/Deltares/hydromt/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt

.. image:: https://readthedocs.org/projects/hydromt/badge/?version=latest
    :target: https://hydromt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**hydroMT** is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. hydroMT builds on the latest packages in the
scientific and geospatial python eco-system: It adopts the xarray_ data structure for 
model maps, rasterio_ for raster I/O, geopandas_ data structure for model geometries and 
vector I/O and the pyflwdir_ data structure flow direction data.


.. _xarray: https://xarray.pydata.org
.. _geopandas: https://geopandas.org
.. _rasterio: https://rasterio.readthedocs.io
.. _pyflwdir: https://deltares.gitlab.io/wflow/pyflwdir

Why hydroMT?
------------

Installation
------------

NOTE: This part is still in development.

hydroMT is availble from pypi and conda-forge, but we recommend installing with conda.

To install hydromt using conda do:

.. code-block:: console

  conda install hydromt -c conda-forge

To create a hydromt environment with conda installed do:

.. code-block:: console

  conda create hydromt -n hydromt -c conda-forge

Documentation
-------------

Learn more about hydroMT in its `online documentation <https://hydromt.readthedocs.io>`_

Contributing
------------

You can find information about contributing to hydroMT at our `Contributing page <https://hydromt.readthedocs.io/contributing.html>`_.

License
-------

Copyright (c) 2019, Deltares

Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#TODO add licences of thrid party software
