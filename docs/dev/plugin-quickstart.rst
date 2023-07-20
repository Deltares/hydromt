================================
Starting your own HydroMT Plugin
================================

Prerequisite
^^^^^^^^^^^^
Here we can explain plugin archetecture


Reading & Writing
^^^^^^^^^^^^^^^^^

Here we will explain the model class and how the read/write functions work


Setup methods
^^^^^^^^^^^^^

Here we will explain the setup methods


Using the template
^^^^^^^^^^^^^^^^^^

To make it easier to get started we have provided a `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_ template to help you get started.
The template itself is located at `https://github.com/savente93/hydromt-plugin-template <https://github.com/savente93/hydromt-plugin-template>`.

To use it you must first install cookiecutter itself like so:

.. code-block:: console

	$ pip install cookiecutter

After that navigate to the folder in your terminal where you want to create the project. After doing that all you need to do is run the
following command:

.. code-block:: console

	$ cookiecutter https://github.com/savente93/hydromt-plugin-template

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
