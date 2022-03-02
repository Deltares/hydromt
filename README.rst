.. _readme:

=====================================================
hydroMT: Build and analyze models like a data-wizard!
=====================================================

|pypi| |conda forge| |docs_latest| |docs_stable| |codecov| |license| |doi| |binder|


What is hydroMT?
----------------
**hydroMT** is an open-source Python package that facilitates the process of building and 
analyzing spatial geoscientific models with a focus on water system models. 
It does so by automating the workflow to go from raw data to a complete model instance which 
is ready to run and to analyse model results once the simulation has finished. 

hydroMT builds on the latest packages in the scientific and geospatial python eco-system including 
xarray_, rasterio_, rioxarray_, geopandas_, scipy_ and pyflwdir_.


Why hydroMT?
------------
Setting up spatial geoscientific models typically requires many (manual) steps 
to process input data and might therefore be time consuming and hard to reproduce. 
Especially improving models based on global geospatial datasets, which are which are 
rapidly becoming available at increasingly high resolutions, might be challenging. 
Furthermore, analyzing model schematization and results from different models, 
which often use model-specific peculiar data formats, can be time consuming.
This package aims to make the model building process **fast**, **modular**, **reproducible** 
by configuring the model building process from a single *ini* configuration file
and **model and data source agnostic** through a common model and data API. 


How to use hydroMT?
-------------------
hydroMT can be used as a **command line** application, which provides command to *build*,
*update* and *clip* models with a single line, or **from python** to exploit its rich interface.
You can learn more about how to use hydroMT in its `online documentation <docs>`_
For a smooth installing experience we recommend installing hydroMT and its dependencies 
from conda-forge in a clean environment, see `installation guide <install_guide>`_.


hydroMT model plugins
---------------------
hydroMT is commonly used in combination with a **model plugin** which 
provides an implementation of the model API for specific model software. 
Known models for which a plugin has been developed include:

* hydromt_wflow_: A framework for distributed rainfall-runoff (wflow_sbm) sediment transport (wflow_sediment) modelling.
* hydromt_delwaq_: A framework for water quality (D-Water Quality) and emissions (D-Emissions) modelling.
* hydromt_sfincs_: A fast 2D hydrodynamic flood model.
* hydromt_fiat_: A flood impact model


How to cite?
------------
For publications, please cite our work using the DOI provided in the Zenodo badge |doi| that points to the latest release.


How to contribute?
-------------------
If you find any issues in the code or documentation feel free to leave in issue on the `github issue tracker <issues>`_ 
You can find information about how to contribute to the hydroMT project at our `contributing page <contributing>`_.

hydroMT seeks active contribution from the (hydro) geoscientific community. 
So, far it has been developed and tested with a range of _Deltares models, but 
we believe it is applicable to a much wider set of geoscientific models and are 
happy to discuss how it can be implemented for your model.


.. _scipy: https://scipy.org/
.. _xarray: https://xarray.pydata.org
.. _geopandas: https://geopandas.org
.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _rasterio: https://rasterio.readthedocs.io
.. _pyflwdir: https://deltares.github.io/pyflwdir
.. _Deltares: https://www.deltares.nl/en/
.. _hydromt_wflow: https://deltares.github.io/hydromt_wflow
.. _hydromt_sfincs: https://deltares.github.io/hydromt_sfincs
.. _hydromt_delwaq: https://deltares.github.io/hydromt_delwaq
.. _hydromt_fiat: https://deltares.github.io/hydromt_fiat
.. _install_guide: https://deltares.github.io/hydromt/preview/getting_started/installation.html
.. _contributing: https://deltares.github.io/hydromt/preview/dev_guide/contributing.html
.. _docs: https://deltares.github.io/hydromt/preview
.. _issues: https://github.com/Deltares/hydromt/issues

.. |pypi| image:: https://img.shields.io/pypi/v/hydromt.svg?style=flat
    :alt: PyPI
    :target: https://pypi.org/project/hydromt/

.. |conda forge| image:: https://anaconda.org/conda-forge/hydromt/badges/version.svg
    :alt: Conda-Forge
    :target: https://anaconda.org/conda-forge/hydromt

.. |codecov| image:: https://codecov.io/gh/Deltares/hydromt/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :alt: Coverage
    :target: https://codecov.io/gh/Deltares/hydromt

.. |docs_latest| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :alt: Latest developers docs
    :target: https://deltares.github.io/hydromt/latest

.. |docs_stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt/stable
    :alt: Stable docs last release

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :alt: Binder
    :target: https://mybinder.org/v2/gh/Deltares/hydromt/main?urlpath=lab/tree/examples

.. |doi| image:: https://zenodo.org/badge/348020332.svg
    :alt: Zenodo
   :target: https://zenodo.org/badge/latestdoi/348020332

.. |license| image:: https://img.shields.io/github/license/Deltares/hydromt?style=flat
   :alt: License
   :target: https://github.com/Deltares/hydromt/blob/main/LICENSE