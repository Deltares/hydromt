.. _installation_user_guide:

==================
Installation guide
==================

This installation guide is for people who only intend to use HydroMT though
the command line tool. For other use cases see the corresponding pages:

* If you plan to use the Python API :ref:`Advanced user installation<installation_advanced_user>`
* If are a plugin developer :ref:`Advanced user installation<installation_plugin_dev>`
* If you are a core developer :ref:`Advanced user installation<installation_core_dev>`

Below have several installation guides depending on which package manager you
use:

* :ref:`Pixi installation <pixi_installation>`
* :ref:`Conda installation <conda_installation>`
* :ref:`Pip installation <pip_installation>`

Pixi Installation
=================

.. Tip::

    This is our recommended way of installing HydroMT!


You can install HydroMT as a CLI application using pixi very easily.
To install it on your system of choice follow the instructions on the
`pixi installation page <https://pixi.sh/latest/>`_. Once you've done that
simply execute the following command:

.. code-block:: console

    $ pixi global install hydromt

Pixi will install HydroMT in it's own new environment so that you will not have
any conflicts with other environments you might have on your system and make
sure it is available from the command line throughout your system.

It is not currently possible to add additional packages into a global
pixi environment but this features is being worked on. For additional details
see `this page <https://github.com/prefix-dev/pixi/issues/342>`_.

To determine whether it is correctly install you can follow the steps in
:ref:`Testing your installation <testing_your_installation>`.

Conda Installation
=================

To install HydroMT in a conda environment called `hydromt` from the conda-forge channel do:

.. code-block:: console

    $ conda create -n hydromt -c conda-forge hydromt

Then, activate the environment (as stated by conda create) to start making use of HydroMT.

To add additional packages into the installed environment (for example `hydromt_wflow`)
you can use the command

.. code-block:: console

    $ conda install -n hydromt hydromt_wflow

To determine whether it is correctly install you can follow the steps in
:ref:`Testing your installation <testing_your_installation>`.

Pip Installation
=================
As python is used by most operating systems, it is recommended to install
hydromt through pip into a separate virtual environment to make sure your
system remains stable.

To create a new virtual environment in your current folder execute the following
commands:

.. code-block:: console

    $ pip install virtualenv
    $ virtualenv hydromt_env

This will create a folder called `hydromt_env` in the current directory where
it will install your dependencies. To activate the environment execute

.. code-block:: console

    $ source hydromt_env/bin/activate

The virtual environment will also contain a script called `deactivate`
which you can use to deactivate the environment, meaning it will no longer get
modified, when you are done using HydroMT.

Now that the system is aware of your virtual environment all you have to do
is install the rest of the software:

.. code-block:: console

    $ pip install hydromt


After you've activated your environment you can install any pip packages you want
like you would normally.

To determine whether it is correctly install you can follow the steps in
:ref:`Testing your installation <testing_your_installation>`.

Testing your installation
=========================

To test whether the installation was successful you can run :code:`hydromt --models` and the output should
look approximately like the one below:


.. code-block:: console

    $ hydromt --models

     (hydromt 1.0.0):
     - Model
