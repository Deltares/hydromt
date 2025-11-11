
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
