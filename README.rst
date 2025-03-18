.. _readme:

===============================================================
HydroMT: Automated and reproducible model building and analysis
===============================================================

|pypi| |conda forge| |docs_latest| |docs_stable| |binder| |codecov| |sonarqube| |license| |doi| |joss_paper|


What is HydroMT?
----------------
**HydroMT** (Hydro Model Tools) is an open-source Python package that facilitates the process of
building and analyzing spatial geoscientific models with a focus on water system models.
It does so by automating the workflow to go from raw data to a complete model instance which
is ready to run and to analyse model results once the simulation has finished.
HydroMT builds on the latest packages in the scientific and geospatial python eco-system including
xarray_, rasterio_, rioxarray_, geopandas_, scipy_ and pyflwdir_.


Why HydroMT?
------------
Setting up spatial geoscientific models typically requires many (manual) steps
to process input data and might therefore be time consuming and hard to reproduce.
Especially improving models based on global geospatial datasets, which are
rapidly becoming available at increasingly high resolutions, might be challenging.
Furthermore, analyzing model schematization and results from different models,
which often use model-specific peculiar data formats, can be time consuming.
This package aims to make the model building process **fast**, **modular** and **reproducible**
by configuring the model building process from a single *yaml* configuration file
and **model- and data-agnostic** through a common model and data interface.


How to use HydroMT?
-------------------
HydroMT can be used as a **command line** application (CLI) which provides commands to *build*,
*update* and *clip* models with a single line, or **from Python** to exploit its rich interface.
You can learn more about how to use HydroMT in its `online documentation. <https://deltares.github.io/hydromt/latest/>`_
For a smooth installing experience, we recommend installing HydroMT and its dependencies
from conda-forge in a clean environment, see `installation guide. <https://deltares.github.io/hydromt/latest/getting_started/installation>`_


HydroMT model plugins
---------------------
HydroMT is commonly used in combination with a **model plugin** which
provides a HydroMT implementation for specific model software. Using the plugins allows to prepare a ready-to-run set of input files from raw geoscientific datasets and analyse model results in a fast and reproducible way.
Known model plugins include:

* hydromt_wflow_: A framework for distributed rainfall-runoff (wflow_sbm) and sediment transport (wflow_sediment) modelling.
* hydromt_delwaq_: A framework for water quality (D-Water Quality) and emissions (D-Emissions) modelling.
* hydromt_sfincs_: A fast 2D hydrodynamic flood model (SFINCS).
* hydromt_fiat_: A flood impact model (FIAT).
* hydromt_delft3dfm_: A flexible mesh hydrodynamic suite for 1D2D and 2D3D modelling (Delft3D FM).


How to cite?
------------
For publications, please cite our JOSS paper |joss_paper|

::
    Eilander et al., (2023). HydroMT: Automated and reproducible model building and analysis. Journal of Open Source Software, 8(83), 4897, https://doi.org/10.21105/joss.04897

To cite a specific software version please use the DOI provided in the Zenodo badge |doi| that points to the latest release.


How to contribute?
-------------------
If you find any issues in the code or documentation feel free to leave an issue on the `github issue tracker. <https://github.com/Deltares/hydromt/issues>`_
You can find information about how to contribute to the HydroMT project at our `contributing page. <https://deltares.github.io/hydromt/latest/dev/contributing>`_

HydroMT seeks active contribution from the (hydro) geoscientific community.
So far, it has been developed and tested with a range of `Deltares <https://www.deltares.nl/en/>`_ models, but
we believe it is applicable to a much wider set of geoscientific models and are
happy to discuss how it can be implemented for your model.


.. _scipy: https://scipy.org/
.. _xarray: https://xarray.pydata.org
.. _geopandas: https://geopandas.org
.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _rasterio: https://rasterio.readthedocs.io
.. _pyflwdir: https://deltares.github.io/pyflwdir
.. _hydromt_wflow: https://deltares.github.io/hydromt_wflow
.. _hydromt_sfincs: https://deltares.github.io/hydromt_sfincs
.. _hydromt_delwaq: https://deltares.github.io/hydromt_delwaq
.. _hydromt_fiat: https://deltares.github.io/hydromt_fiat
.. _hydromt_delft3dfm: https://deltares.github.io/hydromt_delft3dfm

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

.. |joss_paper| image:: https://joss.theoj.org/papers/10.21105/joss.04897/status.svg
   :target: https://doi.org/10.21105/joss.04897

.. |sonarqube| image:: https://sonarcloud.io/api/project_badges/measure?project=Deltares_hydromt&metric=alert_status
    :target: https://sonarcloud.io/summary/new_code?id=Deltares_hydromt
    :alt: SonarQube status
