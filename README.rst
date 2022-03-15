hydroMT: Build and analyze models like a data-wizard!
#####################################################

.. image:: https://codecov.io/gh/Deltares/hydromt/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/hydromt/latest
    :alt: Latest developers docs

.. image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt/stable
    :alt: Stable docs last release

.. image:: https://badge.fury.io/py/hydromt.svg
    :target: https://pypi.org/project/hydromt/
    :alt: Latest PyPI version

.. image:: https://anaconda.org/conda-forge/hydromt/badges/installer/conda.svg
    :target: https://anaconda.org/conda-forge/hydromt

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

.. image:: https://zenodo.org/badge/348020332.svg
   :target: https://zenodo.org/badge/latestdoi/348020332

**hydroMT** is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. hydroMT builds on the latest packages in the
scientific and geospatial python eco-system: It adopts the xarray_ data structure for 
model maps, rasterio_ for raster I/O, geopandas_ data structure for model geometries and 
vector I/O and the pyflwdir_ data structure flow direction data.


.. _xarray: https://xarray.pydata.org
.. _geopandas: https://geopandas.org
.. _rasterio: https://rasterio.readthedocs.io
.. _pyflwdir: https://deltares.github.io/pyflwdir

Why hydroMT?
------------

Installation
------------

hydroMT is available from pypi and conda-forge, but we recommend installing with conda.

To install hydromt using conda do:

.. code-block:: console

  conda install hydromt -c conda-forge

Documentation
-------------

Learn more about hydroMT in its `online documentation <https://deltares.github.io/hydromt>`_

Contributing
------------

You can find information about contributing to hydroMT at our `Contributing page <https://deltares.github.io/hydromt/contributing.html>`_.

Citing
------

For citing our work see the Zenodo badge above, that points to the latest release.
