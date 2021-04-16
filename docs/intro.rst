Introduction
============

What is HydroMT
---------------
HydroMT is a set of tools at the interface between **data**, **user** and hydro **models**. It assists the expert modeller in: 

- Quickly setting up base models for hydrology, groundwater, hydrodynamic, water quality and water demand
- Making maximum use of the best available global or local data
- Easily connecting different models together
- Adjusting and updating models in a consistent way
- Analyzing model outputs.

HydroMT puts data at the centre of the model building process and makes the best use of available high temporal 
and spatial resolution datasets. The approach is to enable the user to quickly get an initial model setup using 
global data, to start discussing where improvements are needed and collect the useful local data to improve the 
model and finally use it for assessment of the impact of strategies.

.. image:: img/hydromt_approach.png

HydroMT and the BlueEarth Initiative
------------------------------------
HydroMT, with imod-python and HYDROLIB, is part of the Model Builder Engine of Deltares 
|BlueEarth|: `<https://blueearth.deltares.org/>`_ 

.. image:: img/BE_model_tools.png

Scope of HydroMT
----------------
HydroMT is a very flexible tool and helps the user to interact with the different components of model preparation so 
that the modeller can prepare exactly what he needs from the data of its choice. Currently supported models are:

- Delwaq
- SFINCS
- RIBASIM
- Wflow: sbm and sediment

Support for the following models is in development:

- iMOD
- Delft3D-FM
- Delft-FIAT
- Delft-FEWS

.. image:: img/supported_models.png

.. |BlueEarth| image:: img/BlueEarth-icon.png