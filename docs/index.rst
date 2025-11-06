.. _readme:

===============================================================
HydroMT: Automated and reproducible model building and analysis
===============================================================

|pypi| |conda forge| |docs_latest| |docs_stable| |binder| |license| |doi| |joss_paper| |sonarqube_coverage| |sonarqube|

HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the
process of building and analysing spatial geoscientific models with a focus on water
system models. It does so by automating the workflow to go from raw data to a complete
model instance which is ready to run and to analyse model results once the simulation
has finished. As such it is an interface between **user**, **data** and **models**.

This documentation describes the core functionality of HydroMT. In practice, HydroMT
uses **model plugins** to interface with specific software like Wflow or SFINCS.
Users are encouraged to explore the :ref:`available model plugins <plugins>`
and consult both this documentation and that of the plugin they are using.

.. figure:: _static/hydromt_using.jpg

**Useful links**

.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: getting_started
        :link-type: ref

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        **Getting started**

        New to HydroMT? Start here with our installation instructions and a brief overview of HydroMT.

    .. grid-item-card::
        :text-align: center
        :link: intro_user_guide
        :link-type: ref

        :octicon:`book;5em;sd-text-icon blue-icon`
        +++
        **User Guide**

        Want to know more about how to use HydroMT? Then visit our user guide.

    .. grid-item-card::
        :text-align: center
        :link: intro_plugin_guide
        :link-type: ref

        :octicon:`terminal;5em;sd-text-icon blue-icon`
        +++
        **Creating your own plugin**

        Interested in developing your own HydroMT plugin? Check out our plugin developer guide to get started.

    .. grid-item-card::
        :text-align: center
        :link: changelog
        :link-type: doc

        :octicon:`graph;5em;sd-text-icon blue-icon`
        +++
        **What's new**

        Stay up to date with the latest changes and improvements in HydroMT by visiting our changelog.


.. toctree::
   :titlesonly:
   :hidden:

   About <about/intro>
   Getting started <overview/intro>
   User guide <user_guide/intro>
   Developer guide <dev/intro>
   API <api/api>
   What's new <changelog>

.. |pypi| image:: https://img.shields.io/pypi/v/hydromt.svg?style=flat
    :alt: PyPI
    :target: https://pypi.org/project/hydromt/

.. |conda forge| image:: https://anaconda.org/conda-forge/hydromt/badges/version.svg
    :alt: Conda-Forge
    :target: https://anaconda.org/conda-forge/hydromt

.. |sonarqube_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=Deltares_hydromt&metric=coverage
    :alt: Coverage
    :target: https://sonarcloud.io/summary/new_code?id=Deltares_hydromt

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
