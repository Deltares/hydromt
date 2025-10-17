.. _plugins:

Plugins
=======

HydroMT model plugins
---------------------
HydroMT is commonly used in combination with a **model plugin** which
provides a HydroMT implementation for specific model software. Using the plugins allows
to prepare a ready-to-run set of input files from raw geoscientific datasets and analyse
model results in a fast and reproducible way.

Known model plugins include:

.. grid:: 5
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt_wflow/
        :link-type: url

        .. image:: ../_static/wflow.png

        +++
        Wflow plugin :octicon:`link-external`

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt_sfincs/
        :link-type: url

        .. image:: ../_static/SFINCS_logo.png

        +++
        SFINCS plugin :octicon:`link-external`

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt_delwaq/
        :link-type: url

        .. image:: ../_static/Delft3D.png

        +++
        Delwaq plugin :octicon:`link-external`


    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt_fiat/
        :link-type: url

        .. image:: ../_static/fiat_logo.svg

        +++
        Delft-FIAT plugin :octicon:`link-external`

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt_delft3dfm/
        :link-type: url

        .. image:: ../_static/Delft3D.png

        +++
        Delft3D FM plugin :octicon:`link-external`



About the model software
------------------------

- Wflow_ is Deltares' solution for modelling hydrological processes, allowing users to account
  for precipitation, interception, snow accumulation and melt, evapotranspiration, soil water,
  surface water and groundwater recharge in a fully distributed environment.
- SFINCS_ is Deltares' reduced-complexity model, designed for super-fast modelling of compound
  flooding events in a dynamic way.
- Delwaq_ is Deltares' water quality process library used to quantify point source and
  diffuse emissions (D-Emissions) and model the fate and transport (D-Water Quality)
  of various substances/pollutants in surface water systems.
- Delft-FIAT_ is Deltares' Flood Impact Assessment Toolbox and used to quantify the impact and risk
  of flooding and other perils.
- Delft3D-FM_ is Deltares' Delft3D Flexible Mesh Suite (Delft3D FM) for hydrodynamic modelling

.. _Wflow: https://deltares.github.io/Wflow.jl/dev/
.. _SFINCS: https://sfincs.readthedocs.io/en/latest/
.. _Delwaq: https://www.deltares.nl/en/software/module/d-water-quality/
.. _Delft-FIAT: https://publicwiki.deltares.nl/display/DFIAT/Delft-FIAT+Home
.. _Delft3D-FM: https://oss.deltares.nl/web/delft3dfm

Starting your own plugin
------------------------
If you would like to start your own HydroMT plugin for another model or software, we
have some tips and a quick start page for you: :ref:`plugin_quickstart`.

If you would like your plugin to appear on this page, please contact us via the issue board of HydroMT!
