.. _about_why_hydromt:

Why HydroMT?
============

Purpose and Scope
-----------------

HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the
process of building and analyzing spatial geoscientific models with a focus on water
system models. It automates the workflow to go from raw data to complete model instances
that are ready to run, making the model building process **fast**, **modular**,
**reproducible**, and **model-agnostic**.

The framework addresses the challenge of setting up spatial geoscientific models, which
typically requires many manual steps to process input data and can be time-consuming and
hard to reproduce. HydroMT solves this by configuring the entire model building process
from a single YAML configuration file through a common model and data interface.

The framework can be used both as a **command line interface** (CLI) providing commands
to build, update, and analyze models, and **from Python** to exploit its rich
programmatic interface.

Key features and Benefits
-------------------------

HydroMT provides several core capabilities that distinguish it from traditional model
building approaches:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Description
     - Benefit
   * - **Automated Workflows**
     - Automates data processing and model setup from raw geo-spatial datasets
     - Saves time and reduces manual effort in model preparation
   * - **Modular Configuration**
     - Single YAML configuration drives entire model building process
     - Enables easy adjustments and customization of the model setup
   * - **Model-Agnostic Interface**
     - Common interface for various water system models
     - Consistent workflow regardless of target model
   * - **Plugin Architecture**
     - Extensible system for model-specific implementations
     - Easy integration of new models and data sources
   * - **Reproducible Builds**
     - Version-controlled configuration and data catalogs
     - Ensures consistent model builds across environments
   * - **Rich Data Ecosystem**
     - Built-in support for global and local geo-spatial datasets via data catalogs
     - Simplifies access to high-quality input data without manual processing
