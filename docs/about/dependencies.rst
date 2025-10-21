.. _dependencies:

Dependencies
============

HydroMT builds on the latest packages in the scientific and geospatial python eco-system including:

- **Core libraries:** xarray_, rioxarray_, pandas_, geopandas_, numpy_, pyflwdir_,
dask, numba
- **Geospatial libraries:** affine, pyproj, shapely, xugrid
- **Statistics:** bottleneck, scipy
- **File I/O:** fsspec, mercantile, netCDF4, pooch, pyogrio, pyarrow, pystac, rasterio,
requests, universal-pathlib, xmltodict, zarr
- **Configuration and CLI tools:** click, importlib-metadata, pydantic, pydantic-settings, pyyaml
- **System and support libraries:** packaging, toml, tomli-w
- **Other (indirect) dependencies:** aiohappyeyeballs, aiohttp, aiosignal, annotated-types, asciitree,
async-timeout, attrs, branca, certifi, cftime, charset-normalizer, click-plugins, cligj, cloudpickle,
contourpy, cycler, fasteners, folium, fonttools, frozenlist, geoalchemy2, geographiclib,
geopy, greenlet, idna, importlib-resources, jinja2, joblib, kiwisolver, llvmlite,
locket, mapclassify, markupsafe, matplotlib, multidict, networkx, numba-celltree, numcodecs, partd, pillow,
platformdirs, propcache, psycopg-binary, pydantic-core, pyparsing, python-dateutil, python-dotenv, pytz,
scikit-learn, six, sqlalchemy, threadpoolctl, toolz, typing-extensions, typing-inspection,
tzdata, urllib3, xyzservices, yarl, zipp

HydroMT also comes with additional optional dependencies that can be installed
to enable extra functionality:

- **io**: gcsfs, fastparquet, openpyxl, pillow, s3fs
- **gdal**: gdal
- **extra**: matplotlib, pyet
- **examples**: cartopy, jupyterlab, notebook
- **slim**: gcsfs, fastparquet, openpyxl, pillow, s3fs, matplotlib, pyet, cartopy, jupyterlab, notebook

You can use pip to install these extra dependencies, e.g.,

.. code-block:: bash

    pip install "hydromt[slim]"


.. _xarray: https://xarray.pydata.org
.. _geopandas: https://geopandas.org
.. _pandas: https://pandas.pydata.org
.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _numpy: https://numpy.org
.. _pyflwdir: https://deltares.github.io/pyflwdir
.. _dask: https://dask.org
.. _numba: https://numba.pydata.org
