.. _contributing:

Developer's guide
=================

Welcome to the HydroMT project. All contributions, bug reports, bug fixes, 
documentation improvements, enhancements, and ideas are welcome. Here's how we work.

Rights
------

The MIT `license <https://github.com/Deltares/hydromt/blob/docs/LICENSE>`_ applies to all contributions.

Issue conventions
-----------------

HydroMT is a relatively new project and highly active. We have bugs, both known and unknown.

The HydroMT `issue tracker <https://github.com/Deltares/hydromt/issues>`_ is the place to share any bugs or feature requests you might have.
Please search existing issues, open and closed, before creating a new one.

For bugs, please provide tracebacks and relevant log files to clarify the issue. 
`Minimal reproducible examples <https://stackoverflow.com/help/minimal-reproducible-example>`_, 
are especially helpful!

Checklist pull requests
-----------------------

If you found a bug or an issue you would like to tackle or contribute to a new development, please make sure do the following steps:
1. If it does not yet exist, create an issue following the :ref:`issue conventions <Issue conventions>`
2. Create a new branch where you develop your new code, see also :ref:`Git conventions <Git conventions>` 
3. Run black before committing your changes, see  :ref:`code format <Code format>`. This does only apply for *.py files. For *ipynb files make sure that you have cleared all results.
4. Update docs/changelog.rst file with a summary of your changes and a link to your pull request. See for example the
  `hydromt changelog <https://github.com/Deltares/hydromt/blob/main/docs/changelog.rst>`__
5. Push your commits to the github repository and open a draft pull request. Potentially, ask other contributors for feedback.
6. Once you're satisfied with the changes mark the pull request as "as ready for review" and ask another contributor to review the code. The review should cover the implementation as well as steps 2-4.
7. Merge the pull request once the review has been approved.

Git conventions
---------------

First of all, if git is new to you, here are some great resources for learning Git:

- the GitHub `help pages <https://docs.github.com/en/github/getting-started-with-github/getting-started-with-git>`__.
- the NumPy’s `documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__.

The code is hosted on GitHub. To contribute you will need to sign up for a free 
GitHub account. We follow the `GitHub workflow 
<https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/github-flow>`__
to allow many people to work together on the project.

After discussing a new proposal or implementation in the issue tracker, you can start 
working on the code. You write your code locally in a new branch of the HydroMT repo or in a
branch of a fork. Once you're done with your first iteration, you commit your code and 
push to your HydroMT repository. 

To create a new branch after you've downloaded the latest changes in the project: 

.. code-block:: console

    $ git pull 
    $ git checkout -b <name-of-branch>

Develop your new code while keeping track of the status and differences using:

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
pull request. We recommend creating a pull request as early as possible to give other 
developers a heads up and to provide an opportunity for valuable early feedback. You 
can create a pull request online or by pushing your branch to a feature-branch. 

HydroMT design conventions
--------------------------

General
^^^^^^^
- We use :ref:`naming and unit conventions <data_convention>` for frequently used variables to assure consistency within HydroMT

Data
^^^^
- Currently, :ref:`these data types <data_types>` are supported, but this list can be extended based on demand.
- Input data is defined in the :ref:`data catalog <data_yaml>` and parsed by HydroMT to the associated 
  Python data type through the DataAdapter class. The goal of this class is to unify the internal representation 
  of the data (its data type, variables names and units) through minimal preprocessing. When accessing data 
  from the data catalog with any ``DataCatalog.get_<data_type>`` method, it is passed through the adapter to 
  ensure a consistent representation of data within HydroMT. The `get_*` methods take additional arguments to
  define a spatial or temporal subset of the dataset.

Model Class
^^^^^^^^^^^
The HydroMT :ref:`Model class <model_api>` consists of several methods and attributes with specific design/naming conventions.
To implement HydroMT for a specific model kernel/software, a child class named `<Name>Model` (e.g. SfincsModel for Sfincs, GridModel for a gridded model) 
should be created with model-specific data readers, writers and setup methods. 

- :ref:`Model data components <model_interface>` are data attributes which together define a model instance and 
  are identical for all models. Each component represents a specific model component and is parsed to a specific 
  Python data object that should adhere to certain specifications. For instance, the ``grid`` component represent 
  all static regular grids of a model in a :py:class:`xarray.Dataset` object.
- Most model components have an associated `write_<component>` and `read_<component>` method to read/write with model 
  specific data formats and parse to / from the model component. These methods may have additional optional arguments
  (i.e. with default values), but no required arguments. The results component does not have write method.
- To build a model we specify ``setup_*`` methods which transform raw input data to a specific model variable, for instance
  the `setup_soilmaps` method in HydroMT-Wflow to transform soil properties to associated Wflow parameter maps which are part 
  of the `staticmaps` component. 
- All public model methods may only contain arguments which require one of the following basic python types: 
  string, numeric integer and float, boolean, None, list and dict types. This is requirement makes it possible to 
  expose these methods and their arguments via a :ref:`model config .ini file <model_config>`.
- Data is exposed to each model method through the ``Model.data_catalog`` attribute which is an instance of the 
  :py:class:`hydromt.DataCatalog`. Data of :ref:`supported data types <data_types>` is provided to model methods 
  by arguments which end with ``_fn`` (short for filename) which refer to a source in the data catalog 
  based on the source name or a file based on the (relative) path to the file. Within a model method the data is read 
  by calling any ``DataCatalog.get_<data_type>`` method which work for both source and file names.
- The Model class currently contains three high-level methods (:py:meth:`~hydromt.Model.build`, 
  :py:meth:`~hydromt.Model.update` and :py:meth:`~hydromt.Model.clip` which are common for all model plugins and 
  exposed through the CLI. This list of methods might be extended going forward.
- The `region` and `res (resolution)` arguments used in the command line :ref:`build <model_build>`
  and :ref:`clip <model_clip>` methods are passed to the model method(s) referred in the internal `_CLI_ARGS` model constant, which 
  in by default, as coded in the Model class, is the `setup_basemaps` method for both arguments. This is typically
  the first model method which should be called when building a model.  
- A Model child class implementation for a specific model kernel can be exposed to HydroMT as a plugin by specifying a 
  ``hydromt.models`` `entry-point <https://packaging.python.org/en/latest/specifications/entry-points/>`_ in the pyproject.toml file of a package. 
  See e.g. the `HydroMT-Wflow pyproject.toml <https://github.com/Deltares/hydromt_wflow/blob/docs/pyproject.toml>`_
- We highly recommend writing integration tests which build/update/clip example model instances and check these with previously build instances.  

Workflows
^^^^^^^^^
- Workflows define (partial) transformations of data from input data to model data. And should, if possible, be kept 
  generic to be shared between model plugins. 
- The input data is passed to the workflow by python data objects consistent with its associated data types 
  (e.g. :py:class:`xarray.Dataset` for regular rasters) and not read by the workflow itself.
- Unit tests should (see below) be written for workflows to ensure these (keep) work(ing) as intended. 


Code conventions
----------------

Naming
^^^^^^
- Avoid using names that are too general or too wordy. Strike a good balance between the two.
- Folder and script names are always lowercase and preferably single words (no underscores)
- Python classes are written with CamelCase
- Methods are written with lowercase and might use underscores for readability. 
  Specific names are used for methods of the Model class and any child classes, see above. 
- Names of (global) constants should be all upper case.
- Internal (non-public) constants and methods start with an underscore. 

Type hinting
^^^^^^^^^^^^
- We use `type hinting <https://docs.python.org/3/library/typing.html>`_ for arguments and returns of all methods and classes 
  Check this `stack overflow post <https://stackoverflow.com/questions/32557920/what-are-type-hints-in-python-3-5>`_ for more 
  background about what typing is and how it can be used. In HydroMT we use it specifically to inform external libraries to 
  about the type arguments of any HydroMT model method. This is work in progress.

Docstrings
^^^^^^^^^^
- We use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  You can easily create these docstring once method arguments have type hints (see above) with 
  the VSCode `autoDocstring pluging <https://github.com/NilsJPWerner/autoDocstring>`_.

Code format
^^^^^^^^^^^
- We use the `black code style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_ 
  for standardized code formatting.
- Make sure the check below returns *All done!* before commiting your edits.

To check the formatting of your code:

.. code-block:: console

    $ black --check . 

To automatically reformat your code:

.. code-block:: console

    $ black . 

Test and CI
-----------

We use `pytest <https://pytest.org>`__ for testing and `github actions <https://docs.github.com/en/actions>`_ for CI. 
- Unit tests are mandatory for new methods and workflows and integration tests are highly recommended for various 
- All tests should be contained in the tests directory in functions named `test_*`.
- We use `CodeCov <https://app.codecov.io/gh/Deltares/hydromt>`_ to monitor the coverage of the tests and aim for high (>90%) coverage. This is work in progress.
- Checkout this `comprehensive guide to pytests <https://levelup.gitconnected.com/a-comprehensive-guide-to-pytest-3676f05df5a0>`_ for more info and tips.

Running the tests
^^^^^^^^^^^^^^^^^

HydroMT's tests live in the tests folder and generally match the main package layout. 
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


Creating a release
------------------

1. Prepare the release by bumping the version number in the __init__.py and updating the docs/changelog.rst file
2. First create a new release on github under https://github.com/Deltares/hydromt/releases. We use semantic versioning and describe the release based on the CHANGELOG.
3. Make sure to update and clean your local git folder. This removes all files which are not tracked by git. 

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