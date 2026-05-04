.. _create_release:


Creating a release
------------------

Releases are produced by three GitHub Actions workflows:

- ``create-release-branch.yml`` — creates a long-lived ``release/vX.Y`` branch
  from ``main`` at version ``X.Y.0`` (major/minor only). Main is **not** bumped
  here.
- ``create-release.yml`` — tags the release on a release branch, creates the
  GitHub release, publishes docs, and opens a ``record-release/v…`` PR that
  merges the release branch back into ``main`` (major / minor / patch).
- ``publish-pypi.yml`` — publishes to PyPI; triggered automatically when a
  GitHub release is published.

.. important::

   After each full release, the ``record-release/v…`` PR merges the release
   branch back into ``main``. This PR bumps ``main`` to ``X.(Y+1).0.dev0``
   (if it isn't already higher), adds a fresh ``Unreleased`` changelog section,
   and carries any code changes from the release branch. Use a **regular merge**
   (not squash) to preserve the branch relationship.

Before tagging a major, minor, or patch release, run the
:ref:`plugin compatibility test <plugin_compat_test>` against the release branch.

Major / minor release
^^^^^^^^^^^^^^^^^^^^^

1. Go to the ``Actions`` tab on GitHub, select **Create release branch
   (minor/major)**, click **Run workflow** and choose ``minor`` or ``major``.
2. The workflow creates ``release/vX.Y`` from ``main``, bumps it to ``X.Y.0``,
   and updates ``docs/changelog.rst`` and ``docs/_static/switcher.json``
   (via ``setup-release.sh``). The branch is pushed but no tag is created yet.
   Main is **not** modified.
3. Push any final fixes or changelog tweaks directly to the release branch.
   When the release content is ready, run the **Create release** workflow with
   ``release_branch = release/vX.Y`` and ``release_type = major`` or ``minor``.

   - Leave **mark_as_latest** checked (default) to make this the new ``stable``
     docs version.
   - The workflow tags ``HEAD`` as ``vX.Y.0``, creates a GitHub release marked
     as latest, publishes versioned docs to GitHub Pages, and opens a
     ``record-release/vX.Y.0`` PR that merges the release branch back into
     ``main`` (bumping version + changelog + switcher).

4. Publishing the GitHub release automatically triggers ``publish-pypi.yml``
   which uploads the package to PyPI.
5. Merge the auto-opened ``record-release/vX.Y.0`` PR into ``main`` using a
   **regular merge** (not squash).
6. The newly published PyPI package will trigger a new PR to the
   `HydroMT feedstock repo on conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_.
   Check whether ``meta.yml`` needs updating and merge the PR to release on
   conda-forge.
7. Celebrate the new release!


.. _create_patch_release:

Patch release
^^^^^^^^^^^^^

Patch releases are made against an already-existing ``release/vX.Y`` branch.
No new branch needs to be created.

If the patch fixes a bug that exists on ``main``, merge the fix to ``main``
first, then cherry-pick the squashed commit onto the release branch (or open a
PR targeting the release branch directly).

1. Go to the ``Actions`` tab on GitHub, select **Create release**, click
   **Run workflow**.
2. Enter the release branch (e.g. ``release/v1.4``) and choose ``patch`` as
   the release type.
3. Decide whether to check **mark_as_latest**:

   - If this is the newest release family (the one ``main`` was prepared
     from), leave it checked.
   - If this is a patch on an older family (e.g. patching ``release/v1.4``
     while ``main`` is preparing ``1.6``), **uncheck it** so that the docs
     ``stable`` symlink stays on the newer family.

4. The workflow increments the patch version, runs ``setup-release.sh`` (bump,
   changelog, switcher), commits, tags ``vX.Y.Z``, creates the GitHub release,
   publishes versioned docs, and opens a ``record-release/vX.Y.Z`` PR. The
   package is published to PyPI automatically.
5. Merge the auto-opened ``record-release/vX.Y.Z`` PR into ``main`` using a
   **regular merge** (not squash).


.. _create_pre_release:

Release candidate
^^^^^^^^^^^^^^^^^

Release candidates are feature-complete builds expected to become the final
release unless critical issues are found. They are produced from a
``release/vX.Y`` branch using the same **Create release** workflow.

1. Go to the ``Actions`` tab on GitHub, select **Create release**, click
   **Run workflow**.
2. Enter the release branch (e.g. ``release/v1.4``) and choose ``rc`` as the
   release type. Leave **mark_as_latest** unchecked (it has no effect for
   pre-releases, but it is good practice).
3. The workflow computes the next available rc version of the form
   ``X.Y.ZrcN`` based on existing tags, commits the version bump on the release
   branch, tags it as ``vX.Y.ZrcN``, and creates a GitHub *pre-release*. No
   record-on-main PR is opened; no docs are published. The package is published
   to PyPI automatically.
4. Anyone can install the release candidate with::

       pip install hydromt==X.Y.ZrcN

   The exact install command is shown in the body of the GitHub pre-release.

.. note::
   Release candidates share the same long-lived ``release/vX.Y`` branch as the
   eventual full release, so the rc commits become part of the release history.

.. warning::
   Pre-releases are marked as pre-releases on GitHub and are not promoted as the
   latest stable release. Do not use an rc as the basis for a full release; run
   **Create release** with type ``major`` / ``minor`` / ``patch`` to produce the
   actual release.



.. _plugin_compat_test:

Plugin compatibility test
-------------------------

Before tagging a major, minor, or patch release, you must run the downstream plugin compatibility test against the release branch.
This is part of the release gate.
It checks whether the new HydroMT wheel still works with a set of mature plugins.

The workflow builds the actual wheel that would be published on PyPI and installs that wheel into each plugin's Pixi environment.
We do not use an editable install.
This makes sure we test the real release artifact, including packaging metadata and included data files.

For each plugin, the workflow runs in two modes.

In the first mode, HydroMT is installed with ``--no-deps`` (``deps=false``).
This upgrades only the HydroMT wheel and keeps the plugin's existing, already solved environment unchanged.
This simulates a user upgrading HydroMT in an existing environment.
If this fails, it means the upgrade is not fully drop-in compatible.
These failures must be reviewed, but they are not automatically release blockers.

In the second mode, HydroMT is installed allowing dependency updates (``deps=true``).
This allows the environment to re-solve and update third-party packages if needed.
This simulates a clean installation.
If this fails, the release is considered broken and must not be finalised.
These failures are release blockers.

How to run the compatibility test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Make sure your release branch (for example ``release/vX.Y``) is up to date.
2. Go to the GitHub Actions tab.
3. Select the **Downstream plugin compatibility** workflow.
4. Click **Run workflow** and choose the release branch.

If the ``with dependencies`` run fails, you must fix the problem before continuing the release.

If the ``no-deps`` run fails, review the failure and decide what to do.
You may need to restore backward compatibility in core, coordinate a plugin update, or accept that upgrading HydroMT requires re-solving the environment.

Do not finalise the release until all blocking failures are resolved and advisory failures have been reviewed.
