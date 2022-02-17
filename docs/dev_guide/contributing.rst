.. _contributing:

Contributing
============

Welcome to the hydroMT project. All contributions, bug reports, bug fixes, 
documentation improvements, enhancements, and ideas are welcome. Here's how we work.

Rights
------

The MIT license (see LICENSE.txt) applies to all contributions.

Issue Conventions
-----------------

The hydroMT issue tracker is for actionable issues.

hydroMT is a relatively new project and highly active. We have bugs, both
known and unknown.

Please search existing issues, open and closed, before creating a new one.

Please provide these details as well as tracebacks and relevant logs. Short scripts and 
datasets demonstrating the issue are especially helpful!

Design Principles
-----------------

hydroMT contains methods to build and analyze models in the hydrology sphere. We try
to build a modular framework in which data can be interchanged in workflows and 
method and workflows can be reused between models. To achieve this we designed a 
GeneralDataApapter to work with any type of regural gridded data in Xarray and vector
data in geopandas. An abstract class for models is designed to create a unifor API for
different models.

- I/O from the filesystem is achieved trough geopandas for vector data, methods in 
  open_raterio for gdal raster data and xarray for netcdf data. 
- Flow direction data is parsed to the pyflwdir.FlwdirRaster object.

Git Conventions
---------------

First of all, if git is new to you, here are some great resources for learning Git:

- the GitHub `help pages <https://docs.github.com/en/github/getting-started-with-github/getting-started-with-git>`__.
- the NumPy’s `documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__.

The code is hosted on GitHub. To contribute you will need to sign up for a free 
GitHub account. We follow the `GitHub workflow 
<https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/github-flow>`__
to allow many people to work together on the project.

After discussing a new proposal or implementation in the issue tracker, you can start 
working on the code. You write your code locally in a new branch hydroMT repo or in a 
branch of a fork. Once you're done with your first iteration, you commit your code and 
push to your hydroMT repository. 

To create a new branch after you've downloaded the latest changes in the project: 

.. code-block:: console

    $ git pull 
    $ git checkout -b <name-of-branch>

Develop your new code and keep while keeping track of the status and differences using:

.. code-block:: console

    $ git status 
    $ git diff

Add and commit local changes, use clear commit messages and add the number of the 
related issue to that (first) commit message:

.. code-block:: console

    $ git add <file-name OR folder-name>
    $ git commit -m "this is my commit message. Ref #xxx"

Regularly push local commits to the repository. For a new branch the remote and name 
of branch need to be added.

.. code-block:: console

    $ git push <remote> <name-of-branch> 

When your changes are ready for review, you can merge them into the main codebase with a 
merge request. We recommend creating a merge request as early as possible to give other 
developers a heads up and to provide an opportunity for valuable early feedback. You 
can create a merge request online or by pushing your branch to a feature-branch. 

Code Conventions
----------------

We use `black <https://black.readthedocs.io/en/stable/>`__ for standardized code formatting.

Tests are mandatory for new features. We use `pytest <https://pytest.org>`__. All tests
should go in the tests directory.

During Continuous Integration testing, several tools will be run to check your code for 
based on pytest, but also stylistic errors.

Development Environment
-----------------------

Developing hydroMT requires Python >= 3.6. We prefer developing with the most recent 
version of Python. We strongly encourage you to develop in a seperate conda environment.
All Python dependencies required to develop hydroMT can be found in `environment.yml <environment.yml>`__.

Initial Setup
^^^^^^^^^^^^^

First, clone hydroMT's ``git`` repo and navigate into the repository:

.. code-block:: console

    $ git https://github.com/Deltares/hydromt.git
    $ cd hydromt

Then, make and activate a new hydromt conda environment based on the environment.yml 
file contained in the repository:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydromt

Finally, build and install hydromt:

.. code-block:: console

    $ pip install -e .

Running the tests
^^^^^^^^^^^^^^^^^

hydroMT's tests live in the tests folder and generally match the main package layout. 
Test should be run from the tests folder.

To run the entire suite and the code coverage report:

.. code-block:: console

    $ cd tests
    $ python -m pytest --verbose --cov=hydromt --cov-report term-missing

A single test file:

.. code-block:: console

    $ python -m pytest --verbose test_rio.py

A single test:

.. code-block:: console

    $ python -m pytest --verbose test_rio.py::test_object

Running code format checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

The code formatting will be checked based on the `black clode style 
<https://black.readthedocs.io/en/stable/the_black_code_style.html>`__ during ci. 
Make sure the check below returns *All done!* before commiting your edits.

To check the formatting of your code:

.. code-block:: console

    $ black --check . 

To automatically reformat your code:

.. code-block:: console

    $ black . 

Creating a release
^^^^^^^^^^^^^^^^^^

1. Prepare the release by bumping the version number in the __init__.py and updating the docs/changelog.rst file
2. First create a new release on github under https://github.com/Deltares/hydromt/releases. We use semantic versioning and describe the release based on the CHANGELOG.
3. Make sure to update and clean your local git folder. This remmoves all files which are not tracked by git. 

.. code-block:: console

    $ git pull
    $ git clean -xfd

4. Build wheels and sdist for the package and check the resulting files in the dist/ directory.

.. code-block:: console

    $ flit build

5. Then use publish to pypi. It will prompt you for your username and password.

.. code-block:: console

    $ flit publish --repository pypi

6. Bump the version number in __init__.py to the next release number with ".dev" postfix and push commit