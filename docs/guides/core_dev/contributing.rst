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

The HydroMT `issue tracker <https://github.com/Deltares/hydromt/issues>`_ is the place to share any bugs or feature requests you might have.
Please search existing issues, open and closed, before creating a new one.

For bugs, please provide tracebacks and relevant log files to clarify the issue.
`Minimal reproducible examples <https://stackoverflow.com/help/minimal-reproducible-example>`_,
are especially helpful! If you're submiting a PR for a bug that does not yet have an
associated issue, please create one together with your PR (unless it is something
trivial). This is important for us to keep track of the changes made to core.

Checklist pull requests
-----------------------

If you found a bug or an issue you would like to tackle or contribute to a new development, please make sure to do the following steps:

1. If it does not yet exist, create an issue following the :ref:`issue-conventions`.
2. Make a fork of the repository. More information on how to do this can be found at
   `the github documentation
   <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>_`
   While you can develop on the main branch of your fork and submit PRs from there, we
   encourage you to make branches on your fork as well, especially if you plan on
   working on multiple issues at the same time. If you have commit rights on the
   repository you may also make your contributions there instead of on a fork.
3. Create a new branch where you develop your new code, see also :ref:`git-conventions`.
4. Both in the case of a new feature and in the case of a bug please add a test that
   demonstrates the correctness of your code. If it is a bug, you should generally ad a
   test that reproduces the bug, and then your code should fix the behaviour. In the
   case of a new feature your test should show and verify how the new feature should be
   used and what assumptions can be made about it. Please make sure the tests pass on
   all environments in our CI. This should be checked in your PR with every commit you push.
5. Make sure all pre-commit hooks pass, see  :ref:`code-format`. For ipynb files make
  sure that you have cleared all results.
6. Update docs/changelog.rst file with a summary of your changes and a link to your pull request. See for example the `hydromt changelog <https://github.com/Deltares/hydromt/blob/main/docs/changelog.rst>`_.
7. Push your commits to the github repository and open a draft pull request. The body of
   the pull request will be pre-filled with a template. Please fill out this template
   with the relevant information, and complete the checklist included. Filling out the
   template will greatly increase the chances of your PR getting a swift response.
8. Once you're satisfied with the changes mark the pull request as "as ready for review"
   and ask another contributor to review the code. The review should cover the
   implementation as well as steps 2-4.
9. Depending on the PR, the reviewer may ask you to either make changes or accept your
   PR.

.. _git-conventions:

Git conventions
---------------

 We follow the `GitHub workflow
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

If you want to synchronize with your colleague, it is important that you both make sure that you have the up to date version by using the `git pull` command.
After that you can bring your branch up to date this by using the `git merge` command:

.. code-block:: console

  $ git pull
  $ git merge feature-A
  Merge made by the 'ort' strategy.
   tmp-a.py | 0
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 tmp-a.py

This means that git detected that you did not make changes to the same file and therefore no problem occurred. However if we imagine that you both make changes to the same file, things will be different:

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

  $ echo 'print("Purple is the best color") # a compromise' > feature-c.py
  $ git add feature-c.py
  $ git commit
  [feature-c-colleague 7dd3f576] Merge branch 'feature-c' into feature-c-colleague

Success!
This is a simple introduction into a potentially very complicated subject. You can read more about the different possibilities here:

*  `Merge Conflicts <https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts>`_
* `Merge Strategies <https://www.atlassian.com/git/tutorials/using-branches/merge-strategy>`_




HydroMT design conventions
--------------------------

General
^^^^^^^
- We use :ref:`naming and unit conventions <data_convention>` for frequently used
  variables to assure consistency within HydroMT

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
To implement HydroMT for a specific model kernel/software, a child class named `<Name>Model` (e.g. SfincsModel for Sfincs)
should be created with model-specific data readers, writers and setup methods as is appropriate.

- :ref:`Model data components <model_interface>` are data attributes which together define a model instance and
  are identical for all models. Each component represents a specific model component and is parsed to a specific
  Python data object that should adhere to certain specifications. For instance, the ``grid`` component represent
  all static regular grids of a model in a :py:class:`xarray.Dataset` object.
- Most model components have an associated `write` and `read` method to read/write with model
  specific data formats and parse to / from the model component. These methods may have additional optional arguments
  (i.e. with default values), but no required arguments. The results component does not have write method.
- All public model methods may only contain arguments which require one of the following basic python types:
  string, numeric integer and float, boolean, None, list and dict types. This is requirement makes it possible to
  expose these methods and their arguments via a :ref:`model config .yml file <model_config>`.
- Data is exposed to each model method through the ``Model.data_catalog`` attribute which is an instance of the
  :py:class:`hydromt.DataCatalog`. Data of :ref:`supported data types <data_types>` is provided to model methods
  by arguments which end with ``_fn`` (short for filename) which refer to a source in the data catalog
  based on the source name or a file based on the (relative) path to the file. Within a model method the data is read
  by calling any ``DataCatalog.get_<data_type>`` method which work for both source and file names.
- The Model class currently contains three high-level methods (:py:meth:`~hydromt.Model.build`,
  :py:meth:`~hydromt.Model.update` and :py:meth:`~hydromt.Model.clip` which are common for all model plugins and
  exposed through the CLI. This list of methods might be extended going forward.
- A Model child class implementation for a specific model kernel can be exposed to HydroMT as a plugin by specifying a
  ``hydromt.models`` `entry-point <https://packaging.python.org/en/latest/specifications/entry-points/>`_ in the pyproject.toml file of a package.
  For a more detailed explanation of how to build a plugin please refer to the
  :ref:`plugin_dev` section.
- We highly recommend writing integration tests to ensure the correctness of your code.


Code conventions
----------------

Naming
^^^^^^
- Please avoid using short abbreviations in function and variable names unless they are
  very well known, they generally make code harder to read and follow.
- Avoid using names that are too general or too wordy. Strike a good balance between the two.
- Folder and script names are always lowercase and preferably single words (no underscores)
- Python classes are written with CamelCase
- Methods are written with lowercase and might use underscores for readability.
  Specific names are used for methods of the Model class and any child classes, see
  above.
- Names of (global) constants should be all upper case.
- Internal (non-public) constants and methods start with an underscore, these should not
  be used outside of your package's code.

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
  the VSCode `autoDocstring plugin <https://github.com/NilsJPWerner/autoDocstring>`_.
- please ensure that all public code you constribute has a valid docstring.

.. _code-format:

Code format
^^^^^^^^^^^
- We use the `black code style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_ and `pre-commit <https://pre-commit.com>`_ to keep everything formatted. We use the formatter included with `ruff <https://docs.astral.sh/ruff/formatter/>`_ which is black compatible, but much faster. Please make sure all hooks pass before commiting. Pre-commit will do this for you if it's installed correctly.

You can install pre-commit by running:

.. code-block:: console

  $ pip install pre-commit

It is best to install pre-commit in your existing environment. After that simply install the necessary hooks with

.. code-block:: console

  $ pre-commit install

After doing this pre-commit will check all your staged files when committing.

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

    ruff-format..............................................................Passed

    hydromt/new_feature.py:1:1: D100 Missing docstring in public module
    Found 2 errors (1 fixed, 1 remaining).


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
- Checkout this `comprehensive guide to pytest <https://levelup.gitconnected.com/a-comprehensive-guide-to-pytest-3676f05df5a0>`_ for more info and tips.

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

1. Go to the `actions` tab on Github, select `Create a release` from the actions listen to the left, then use the `run workflow` button to start the release process. You will be asked whether it will be a `major`, `minor` or `patch` release. Choose the appropriate action.
2. The action you just run will open a new PR for you with a new branch named `release/v<NEW_VERSION>`. (the `NEW_VERSION` will be calculated for you based on which kind of release you selected.) In the new PR, the changelog, hydromt version and sphinx `switcher.json` will be updated for you. Any changes you made to the `pyproject.toml` since the last release will be posted as a comment in the PR. You will need these during the Conda-forge release if there are any.
3. Every commit to this new branch will trigger the creation (and testing) of release artifacts. In our case those are: Documentation, the PyPi package and docker image (the conda release happens separately). After the artifacts are created, they will be uploaded to the repository's internal artifact cache. A bot will post links to these created artifacts in the PR which you can use to download and locally inspect them.
4. When you are happy with the release in the PR, you can simply merge it. We suggest naming the commit something like "Release v<NEW_VERSION>"
5. After the PR is merged, a action should start (though it will not show up under the PR itself) that will publish the latest artifacts created to their respective platform. After this, a bot will add a final commit to the `main` branch, setting the hydromt version back to a dev version, and adding new headers to the `docs/changelog.rst` for unreleased features. It will also create a tag and a github release for you automatically. The release is now done as far as this repo is concerned.
6. The newly published PyPi package will trigger a new PR to the `HydroMT feedstock repos of conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_.
   Here you can use the comment posted to the release PR to see if the `meta.yml` needs to be updated. Merge the PR to release the new version on conda-forge.
7. celebrate the new release!
