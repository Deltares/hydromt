Common usage
------------

The most common usage of HydroMT is to build a model from scratch and to update and visualize an existing model.
Here, a high-level example of how to build a model using HydroMT is provided. Building a model from scratch with
HydroMT involves the following generic steps:

1) Define the input data in a :ref:`yaml data catalog file <own_catalog>` or selects available datasets from a
   :ref:`pre-defined data catalog <existing_catalog>`.
2) Define the model :ref:`region <region>` which describes the area of interest. The model region can be based on a
   simple bounding box or geometry, but also a (sub)(inter)basin that is delineated on-the-fly based on available
   hydrography data.
3) Configure the model setup in an :ref:`yaml configuration file <model_workflow>`. A HydroMT yaml configuration file
   represents a reproducible recipe to build a model by listing (in order of execution) the model methods and
   their arguments. These methods and their arguments are described in the documentation.
4) Run the HydroMT :ref:`build method <model_build>` from either command line (as shown in the figure) or Python.
