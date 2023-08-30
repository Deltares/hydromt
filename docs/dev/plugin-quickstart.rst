================================
Starting your own HydroMT Plugin
================================

Prerequisite
^^^^^^^^^^^^
There are three important concepts in HydroMT core that are important to cover:

- ``DataAdapters``: These are what basically hold all information about how to approach and read data as well as some of the metadata about it. While a lot of the work in HydroMT happens here, plugins or users shouldn't really need to know about these beyond using the proper ``data_type`` in their configuration.
- ``DataCatalog``: these are basically just a thin wrapper around the ``DataAdapters`` that does some book keeping.
- ``Model``: This is where the magic happens (as far as the plugin is concerned). We have provided some base models that you can override to get basic/generic functionality, but using the model functionality is where it will be at for you.

Currently HydroMT uses the `entrypoints` package to advertise it's, well, entrypoints. Entrypoints are how you can tell HydroMT core about your plugin. As an example we can look at the current ``hydromt_wflow`` model. Specifically it's this line in the pyproject.toml:

.. code-block:: toml

	[project.entry-points."hydromt.models"]
	wflow = "hydromt_wflow.wflow:WflowModel"


This snippet will tell HydroMT core three things:

1. there will be an api in the plugin we call "wflow" that HydroMT core can use
2. it is located in the file hydromt_wflow.wflow
3. it implements the ``hydromt.models`` API

To make sure that your model is compatible with HydroMT core, you can use one of the model classes in HydroMT core as a base, (``grid``, ``mesh``, ``lumped`` or ``netowrk``). You can then use the base model class methods as well as create any functionality on top of that.  If everything has gone well, you should be able to access your code through HydroMT now! If you want to mix models functionalities, most model classes also provide ``Mixin`` varients that you can use to mix and match a bit more modularly.


Reading & Writing
^^^^^^^^^^^^^^^^^

The ``DataAdapter`` is very useful for reading and writing data, but because the way that models consume data is so hyper specialised there is no real generic way to solve this entire problem. Therefore part of writing a plug in is writing the Input/Output (IO) methods to prepare data exactly as your model expects them. To do this you'll be writing most `read` or `write` methods on your model class. That will ensure that the data comes in and goes out expecting exactly as you would.

Setup methods
^^^^^^^^^^^^^

In general, a HydroMT model does 4 things:

1. read or otherwise fetch data
2. fiddle with that data in some way
3. record attributes based on that data
4. write the attributes and data to whatever location is desired.

The first and last point is what your IO functions will do (see previous paragraph). Point 2 & 3 is what your setup methods should do. They should be given data and be able to manipulate it in such a way that it can be written for a model to read and understand. This is again, highly dependant on what your model will need but hopefully you can get a long way with the building blocks and generic functionalieties already present in HyroMT.


Model Configs
^^^^^^^^^^^^^

Okay, so you have your data prepared, your plugin has all the functionalities needed for IO and setup. We're nearlly thare. The final bit that you'll have to do is write a model config file. This is a yaml file that is basically the final mannifest of how the functions you've written should be used. At the top level the model config keys should correspond to the names of the functions, and the mappings beneath that should be the arguments that should be passed to those functions. Note that the functions in the yaml file will be executed in the order they appear in the file, so be careful here!


Using the template
^^^^^^^^^^^^^^^^^^

To make it easier to get started we have provided a `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_ template to help you get started.
The template itself is located at `https://github.com/Deltares/hydromt-plugin-template <https://github.com/Deltares/hydromt-plugin-template>`.

To use it you must first install cookiecutter itself like so:

.. code-block:: console

	$ pip install cookiecutter

After that navigate to the folder in your terminal where you want to create the project. After doing that all you need to do is run the
following command:

.. code-block:: console

	$ cookiecutter https://github.com/Deltares/hydromt-plugin-template

You will be prompted for some information. After you've entered the information the project should be automatically created for you! Let's
say you just created the plugin called `hydromt_plugin` before you can start using it you'll need to initialise git within it like so:

.. code-block:: console

	$ cd hydromt_plugin
	$ git init

If your project has dependencies you can add them in the pyprojec.toml under the `dependencies` array. If you have `tomli` installed, you can
use the `make_env.py` script to generate a conda environment specifcation see :ref:`The developoer instalation page <dev_install>` for
more information on how to use this script.

Now, assuming that you've made a repository on github within the Deltares organisation you just need to add it as a remote in the repository
and push it.

.. code-block:: console

	$ git remote add origin https://github.com/Deltares/hydromt_plugin
	$ git push

After this you can open up the github repository website, and you should see your generated project. You are now ready to start developing your own
plugin! Well done, and good luck!
