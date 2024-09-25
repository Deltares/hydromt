.. _terminology:

Terminology
===========

HydroMT and this documentation use a specific terminology to describe specific objects or processes.

==============================  ======================================================================================
Term                            Explanation
==============================  ======================================================================================
Command Line Interface (CLI)    high-level interface to HydroMT *build*, *update*, *check* and *export* methods.
Configuration (HydroMT)         (.yaml) file describing the complete pipeline with all methods and their arguments to
                                *build* or *update* a model.
Data catalog                    A set of data sources available for HydroMT. It is build up from *yaml* files containing
                                one or more data sources with information about how to read and optionally preprocess
                                the data and contains meta-data about the data source.
Data source                     Input data to be processed by HydroMT. Data sources are listed in yaml files.
Model                           A set of files describing the schematization, forcing, states, simulation configuration
                                and results for any supported model kernel and model classes. The final set of files is
                                dependent on the model type (grid, vector or mesh model for examples) or the model plugin.
Model class                     A model instance can be instantiated from different model schematization concepts. Generalized
                                model classes currently supported within HydroMT are Grid Model (distributed models), vector Model
                                (semi-distributed), Mesh Model (unstructured) and in the future
                                Network Model (relational model). Specific model classes for specific softwares have been implemented
                                as plugins, see Model plugin.
Model attributes                Direct properties of a model, such as the model root. They can be called when using
                                HydroMT from python.
Model component                 A model is described by HydroMT with the following components: maps,
                                geoms (vector data), forcing, results, states, config, grid (for a grid model), vector
                                (for a vector model), mesh (for a mesh model).
Model plugin                    (Plugin) Package that links the HydroMT Model class to a specific model software so that HydroMT can build
                                and update models and analyze its simulation results. For example *HydroMT-Wflow*, *HydroMT-SFINCS* etc.
                                Plugins are installed separately from HydroMT and are not part of the HydroMT core package.
                                Plugins are the most common way of using HydroMT to build and update specific models.
Model kernel                    The model software to execute a model simulation. This is *not* part of any HydroMT plugin.
Region                          Argument of the *build*CLI methods that specifies the region of interest where
                                the model should be prepared / which spatial subregion should be clipped.
==============================  ======================================================================================
