Installation
============

First install
-------------

We are working on making the package available through pip and conda package managers.
For now we advise the following steps to install the package.

First, clone hydromt's ``git`` repo or download and unzip the latest release from
`github <https://github.com/Deltares/hydromt/releases>`_, then navigate into the 
the code folder (where environment.yml and setup.py are located):

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt.git
    $ cd hydromt

Then, make and activate a new hydromt conda environment based on the environment.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydromt

Finally, build and install hydromt using pip. If you wish to develop in hydromt, then 
make sure you've cloned from git and make an editable install of hydromt by adding 
``-e`` after install:

.. code-block:: console

    $ pip install .

or for developpers:

.. code-block:: console

    $ pip install -e .

For more information about how to contribute, see :ref:`Contributing to hydroMT <contributing>`.

Update
------

To update hydromt execute the following steps. Note that un- and re-installing 
hydromt with pip is not required if hydromt is installed as editable package using the 
*-e flag* in pip install. 

First navigate to your local hydromt repository and update it using git pull:

.. code-block:: console

    $ cd /path/to/hydromt/
    $ git pull

Then, activate the conda environment in which you've install hydromt and uinstall the 
hydromt package:

.. code-block:: console

    $ conda activate hydromt
    $ pip uninstall hydromt

Finally, update the environment using the environment.yml file, then re-install pip based on the updated repository:

.. code-block:: console

    $ conda env update -f=environment.yml -n hydromt
    $ pip install .


