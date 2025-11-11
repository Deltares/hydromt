
.. _issue-conventions:

Issue conventions
-----------------

The HydroMT `issue tracker <https://github.com/Deltares/hydromt/issues>`_ is the place to share any bugs or feature requests you might have.
Please search existing issues, open and closed, before creating a new one.

For bugs, please provide tracebacks and relevant log files to clarify the issue.
`Minimal reproducible examples <https://stackoverflow.com/help/minimal-reproducible-example>`_,
are especially helpful! If you're submitting a PR for a bug that does not yet have an
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
6. Update docs/changelog.rst file with a summary of your changes and a link to your
   pull request. See for example the `hydromt changelog
   <https://github.com/Deltares/hydromt/blob/main/docs/changelog.rst>`_.
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

We follow the `GitHub workflow <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests-github-flow>`_
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
