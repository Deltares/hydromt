.. _installation_advanced_user:

Installation
============

.. _installation_user_guide:

==================
Installation guide
==================

This installation guide is for people who intend to use HydroMT though
the Python API line tool. We will assume you know how to create an environment from
scratch for whatever package manager you are using.
For other use cases see the corresponding pages:

* If you plan to use the CLI only :ref:`Advanced user installation<installation_user_guide>`
* If are a plugin developer :ref:`Advanced user installation<installation_plugin_dev>`
* If you are a core developer :ref:`Advanced user installation<installation_core_dev>`

Below have several installation guides depending on which package manager you
use:

* :ref:`Pixi installation <pixi_installation>`
* :ref:`Conda installation <conda_installation>`
* :ref:`Pip installation <pip_installation>`
* :ref:`Docker installation <docker_installation>`

Pixi Installation
=================

.. Tip::

    This is our recommended way of installing HydroMT!

If you wish to use HydroMT thought the Python API, you will have to install
it in whatever environment you are using. In the case of pixi all you have to
do is add it to your `pixi.toml` or `pyproject.toml` under the relevant dependencies section:

.. code-block:: toml

    [dependencies]
    ...
    hydromt = "*"

Common editors such as `VSCode <https://github.com/microsoft/vscode-python/issues/22978>`_ and `JupyterLab <https://pixi.sh/latest/ide_integration/jupyterlab/>`_ should correctly detect
your environment


Conda Installation
=================

If you wish to use HydroMT thought the Python API, you will have to install
it in whatever environment you are using. In the case of conda, you can either
add `hydromt` to your `environment.yml` and recreate your environment or add
it manually as so:

.. code-block:: console

    $ conda install -n <env_name> -c conda-forge hydromt

.. code-block:: console

    $ conda install -n hydromt hydromt_wflow

To determine whether it is correctly install you can follow the steps in
:ref:`Testing your installation <testing_your_installation>`.

Pip Installation
=================

After you've activated your environment you can install any pip packages you want
like you would normally.

Additionally if you use a `pyproject.toml` or `requiremnts.txt` you can simply
add `hydromt` to the relevant sections of those files.
