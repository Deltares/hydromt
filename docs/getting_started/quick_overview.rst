Quick overview
==============

Typically, we use HydroMT together with a plugin to build a model from scratch. 
Here, we illustrate this for the a Wflow_ rainfall-runoff model, but you can follow 
the procedure is identical for other models. 

Install HydroMT & model plugin
------------------------------

If you haven't already done so, first install HydroMT and the `HydroMT-Wflow plugin`_ 
in a new ``hydromt`` environment and activate this environment.
For more information about the installation, please refer to the :ref:`installation guide <installation_guide>`.

.. code-block:: console

    $ mamba create -n hydromt -c conda-forge python=3.9 hydromt hydromt-wflow
    $ conda activate hydromt

Next, check if the installation was successful by running the command below. 
This returns the available models for HydroMT and should at least contain wflow and wflow_sediment.

.. code-block:: console

    $ hydromt --models

    >> hydroMT model plugins: wflow (vx.x.x), wflow_sediment (vx.x.x)

Build a model
-------------

Now you can create a model from raw data. To do so, you need to define 

1) the **source data**: To try out HydroMT, you can make use of the publicly available :ref:`HydroMT artifacts data catalog <existing_catalog>` 
   which contains data for the Piave basin in Northern Italy and is the default catalog if no other one is specified.
2) the **model region**: There are many options to define the :ref:`model region <cli_region>`. In this example the model region is defined 
   by the Piave subbasin upstream from a outlet point defined: ``"{'subbasin': [12.2051, 45.8331], 'strord': 4}"``
3) the **model setup configuration**: Finally, the model setup needs to be configured. Here, the example configuration from the HydroMT-Wflow repository 
   is used. You can download the *ini* file `here (right click & save as) <https://raw.githubusercontent.com/Deltares/hydromt_wflow/main/examples/wflow_build.ini>`_ and save it in the current directory. 
   For information about specific options, please visit the documentation of the HydroMT plugin of your model of interest.

These steps are combined into the following command which saves all Wflow model files and a `hydromt.log` file 
in the `wflow_test` folder. This Wflow model instance is ready to be run with Wflow_. 

.. code-block:: console

    $ hydromt build wflow ./wflow_test "{'subbasin': [12.2051, 45.8331], 'strord': 4}" -vv -i build_wflow.ini

.. _Wflow: https://deltares.github.io/Wflow.jl/dev
.. _HydroMT-Wflow plugin: https://deltares.github.io/hydromt_wflow/