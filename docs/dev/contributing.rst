.. _contributing:

Developer's guide
=================

Welcome to the HydroMT project. All contributions, bug reports, bug fixes,
documentation improvements, enhancements, and ideas are welcome. Here's how we work.

Rights
------

The MIT `license <https://github.com/Deltares/hydromt/blob/docs/LICENSE>`_ applies to all contributions.


.. _issue-conventions:

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

If you found a bug or an issue you would like to tackle or contribute to a new development, please make sure to do the following steps:

1. If it does not yet exist, create an issue following the :ref:`issue-conventions`.
2. Create a new branch where you develop your new code, see also :ref:`git-conventions`.
3. Make sure all pre-commit hooks pass, see  :ref:`code-format`. For ipynb files make sure that you have cleared all results.
4. Update docs/changelog.rst file with a summary of your changes and a link to your pull request. See for example the `hydromt changelog <https://github.com/Deltares/hydromt/blob/main/docs/changelog.rst>`_.
5. Push your commits to the github repository and open a draft pull request. The body of the pull request will be pre-filled with a template. Please fill out this template with the relevant information, and complete the checklist included.
6. Once you're satisfied with the changes mark the pull request as "as ready for review" and ask another contributor to review the code. The review should cover the implementation as well as steps 2-4.
7. Once your request has been approved, a bot will add a commit to your branch to automatically update the version number. Please wait for this to complete before you merge. Once it is done, a comment saying "You're free to merge now" will be added to your PR.
8. Note that if you want to make changes after this commit has been added, you should pull the branch first to avoid merge conflicts.
9. You are now free to merge your pull request. Do merge with a `squash` commit!! This is important to make sure the version counting stays accurate.


.. _git-conventions:

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

Dealing with merge conflicts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because git facilitates many people working on the same piece of code, it can happen that someone else makes changes to the repository before you do.
When this happens it's important to synchronize the code base before merging to make sure the outcome will look as we expect. For example, imagine you've made a new feature by branching off main:

.. code-block:: console

  $ git checkout main && git checkout -b feature-A
  $ touch hydromt/feature-A.py
  $ git add hydromt/feature-A.py
  $ git commit -m "implement feature A!"

in the mean time your colleague does the same:

.. code-block:: console

  $ git checkout main && git checkout -b feature-B
  $ touch hydromt/feature-B.py
  $ git add hydromt/feature-B.py
  $ git commit -m "implement feature B!"

If you want to syncronize with your colleague, it is important that you both make sure that you have the up to date version by using the `git pull` command.
After that you can bring your branch up to date this by using the `git merge` command:

.. code-block:: console

  $ git pull
  $ git merge feature-A
  Merge made by the 'ort' strategy.
   tmp-a.py | 0
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 tmp-a.py

This means that git detected that you didt not make changes to the same file and therefore no problem occured. However if we imagine that you both make changes to the same file, things will be different:

.. code-block:: console

  $ git checkout main && git checkout -b feature-c
  $ echo 'print("blue is the best colour")' > feature-c.py
  $ git add feature-c.py
  $ git commit -m "implement feature c!"
  $ git checkout main && git checkout -b feature-c-colleague
  $ echo 'print("Orange is the best colour")' > feature-c.py
  $ git add feature-c.py
  $ git commit -m "implement feature c!"
  $ git merge feature-c
  Auto-merging feature-c.py
  CONFLICT (add/add): Merge conflict in feature-c.py
  Automatic merge failed; fix conflicts and then commit the result.

If we open up the file we can see some changes have been made:

.. code-block:: python

  <<<<<<< HEAD

  print("Orange is the best colour")

  ||||||| <hash>
  =======
  print("blue is the best colour")
  >>>>>>> feature-c

Here we see the contents of both the commits. The top one are the changes the branch made that initiated the merge, and the bottom one is the other branch. The branch name is also listed after the >>>>>. If we try to commit now, it will not let us:

.. code-block:: console

  $ git commit
  U       feature-c.py
  error: Committing is not possible because you have unmerged files.
  hint: Fix them up in the work tree, and then use 'git add/rm <file>'
  hint: as appropriate to mark resolution and make a commit.
  fatal: Exiting because of an unresolved conflict.

It's telling us we first need to tell it what we want to do with the current conflict. To do this, simply edit the file how you'd like it to be, and add it to the staging, then continue with the merge like so:

.. code-block:: console

  $ echo 'print("Pruple is the best color") # a comporomise' > feature-c.py
  $ git add feature-c.py
  $ git commit
  [feature-c-colleague 7dd3f576] Merge branch 'feature-c' into feature-c-colleague

Success!
This is a simple introduction into a potentially very complicated subject. You can read more about the different possibilities here:

* `Merge Conflicts <https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts>`_
* `Merge Strategies <https://www.atlassian.com/git/tutorials/using-branches/merge-strategy>`_




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

.. _code-format:

Code format
^^^^^^^^^^^
- We use the `black code style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_ and `pre-commit <https://pre-commit.com>`_ to keep everything formatted. Please make sure all hooks pass before commiting. Pre-commit will do this for you if it's installed correctly.

You can install pre-commit by running:

.. code-block:: console

  $ pip install pre-commit

It is best to install pre-commit in your existing enviromnment. After that simply install the necessary hooks with

.. code-block:: console

  $ pre-commit install

After doing this pre-commit will check all your staged files when commiting.

For example say that you've added the following new feature:


.. code-block:: console

  $ echo 'import os\nprint("This is a new exciting feature")' > hydromt/new_feature.py

(you do not have to do this, it is just for demonstration, but you can copy and execute this code to try for yourself.)

Then you can add the new feature to the git staging area and try to commit as usual. However pre-commit will tell you that you should add some docstrings for example. You should see an output similar to the one below:

.. code-block:: console

  $ git add hydromt/new_feature.py
  $ git commit -m "The feature you've all been waiting for."
    Trim Trailing Whitespace.................................................Passed
    Fix End of Files.........................................................Failed
    - hook id: end-of-file-fixer
    - exit code: 1
    - files were modified by this hook

    Fixing hydromt/new_feature.py

    Check Yaml...........................................(no files to check)Skipped
    Check for added large files..............................................Passed
    Check python ast.........................................................Passed
    Check JSON...........................................(no files to check)Skipped
    Debug Statements (Python)................................................Passed
    Mixed line ending........................................................Passed
    Format YAML files....................................(no files to check)Skipped
    ruff.....................................................................Failed
    - hook id: ruff
    - exit code: 1
    - files were modified by this hook

    hydromt/new_feature.py:1:1: D100 Missing docstring in public module
    Found 2 errors (1 fixed, 1 remaining).

    black....................................................................Passed

This means that pre-commit has found issues in the code you submitted. In the case of the import it was able to fix it automatically. However `ruff` has also detected that you have not added a docstring for the new feature. You can find this out by running:

.. code-block:: console

  $ ruff .

which will show you the same output:

.. code-block:: console

  hydromt/new_feature.py:1:1: D100 Missing docstring in public module
  Found 1 error.

After you've fixed this problem by for example adding the docstring """Implement the cool new feature""" at the top of the new file, you just have to add the new version to the staging area again and re-attempt the commit which should now succeed:

.. code-block:: console

  $ git add hydromt/new_feature.py
  $ git commit -m "The feature you've all been waiting for."
  Trim Trailing Whitespace.................................................Passed
  Fix End of Files.........................................................Passed
  Check Yaml...........................................(no files to check)Skipped
  Check for added large files..............................................Passed
  Check python ast.........................................................Passed
  Check JSON...........................................(no files to check)Skipped
  Debug Statements (Python)................................................Passed
  Mixed line ending........................................................Passed
  Format YAML files....................................(no files to check)Skipped
  ruff.....................................................................Passed
  black....................................................................Passed
  [linting a5e9b683] The feature you've all been waiting for.
   1 file changed, 4 insertions(+)
   create mode 100644 hydromt/new_feature.py

Now you can push your commit as normal.

From time to time you might see comments like these:

.. code-block:: python

  import rioxarray # noqa: F401

The `noqa` is instructing the linters to ignore the specified rule for the line in question. Whenever possible, we try to avoid using these but it's not always possible. The full list of rules can be found here: `Ruff Rules Section <https://beta.ruff.rs/docs/rules/>`_ Some common ones are:

* E501: Line too long.
* F401: Unused import.
* D102: Public methods should have docstrings.


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

1. Create a new branch with the name "release/<version>" where <version> is the version number, e.g. v0.7.0
2. Bump the version number (without "v"!) in the __init__.py, check and update the docs/changelog.rst file and add a short summary to the changelog for this version.
   Check if all dependencies in the toml are up to date. Commit all changes
3. Create a tag using `git tag <version>`, e.g. git tag v0.7.0
4. Push your changes to github. To include the tag do `git push origin <version>`. This should trigger a test release to test.pypi.org
5. If all tests and the test release have succeeded, merge de branch to main.
6. Create a new release on github under https://github.com/Deltares/hydromt/releases.
   Use the "generate release notes" button and copy the content of the changelog for this version on top of the release notes. This should trigger the release to PyPi.
7. The new PyPi package will trigger a new PR to the `HydroMT feedstock repos of conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_.
   Check if all dependencies are up to date and modify the PR if necessary. Merge the PR to release the new version on conda-forge.


.. NOTE::

  In the next PR that get's merged into main, the version numbers in __init__.py and the changelog should be changed to the next release with ".dev" postfix.
