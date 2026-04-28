.. _create_release:


Creating a release
------------------

Releases are produced by four GitHub Actions workflows that work together:

- ``create-release-branch.yml`` — creates a long-lived ``release/vX.Y`` branch (major/minor only).
- ``create-release.yml`` — tags the release on a release branch and creates the GitHub release (major / minor / patch / rc).
- ``post-release-cleanup.yml`` — opens a follow-up PR after a release PR is merged into ``main``.
- ``publish-pypi.yml`` — publishes to PyPI; runs automatically when a GitHub release is published.

Before tagging a major, minor, or patch release, run the :ref:`plugin compatibility test <plugin_compat_test>` against the release branch.

Major / minor release
^^^^^^^^^^^^^^^^^^^^^

1. Go to the ``Actions`` tab on GitHub, select **Create release branch (minor/major)**, click **Run workflow** and choose ``minor`` or ``major``.
2. The workflow creates a ``release/vX.Y`` branch off ``main`` (e.g. ``release/v1.4``), bumps the version to ``X.Y.0``, and updates ``docs/changelog.rst`` and ``docs/_static/switcher.json``. The branch is pushed but no PR is opened yet.
3. Push any final fixes or changelog tweaks directly to the release branch. When you are happy with the release content, run the **Create release** workflow on this branch with type ``major`` or ``minor``. This tags the branch HEAD as ``vX.Y.0``, creates a GitHub release marked as latest, opens a PR from ``release/vX.Y`` into ``main``, comments the ``pyproject.toml`` diff since the previous tag on that PR, and publishes the versioned docs to GitHub Pages.
4. Publishing the GitHub release automatically triggers ``publish-pypi.yml`` which uploads the package to PyPI.
5. Merge the release PR into ``main`` **WITHOUT SQUASHING**! This automatically triggers **Post-release cleanup**, which opens a follow-up PR resetting the version in source to ``X.Y.Z.dev0`` and adding a fresh ``Unreleased`` section to ``docs/changelog.rst``.
6. The newly published PyPI package will trigger a new PR to the `HydroMT feedstock repo on conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_. Use the ``pyproject.toml`` diff comment from step 3 to check whether ``meta.yml`` needs updating. Merge the PR to release on conda-forge.
7. Celebrate the new release!


.. _create_patch_release:

Patch release
^^^^^^^^^^^^^

Patch releases are made against an already-existing ``release/vX.Y`` branch rather than creating a new one.

1. Go to the ``Actions`` tab on GitHub, select **Create release**, click **Run workflow**.
2. Enter the release branch (e.g. ``release/v1.4``) and choose ``patch`` as the release type.
3. The workflow increments the patch version (reading the current version from the branch), updates the changelog and ``switcher.json``, commits the bump, tags HEAD as ``vX.Y.Z``, creates a GitHub release marked as latest, opens a PR from ``release/vX.Y`` into ``main``, comments the ``pyproject.toml`` diff since the previous tag, and publishes the versioned docs to GitHub Pages. The package is published to PyPI automatically.
4. Merge the auto-opened PR. Once merged, **Post-release cleanup** opens its follow-up PR.


.. _create_pre_release:

Release candidate
^^^^^^^^^^^^^^^^^

Release candidates are feature-complete builds expected to become the final release unless critical issues are found. They are produced from a ``release/vX.Y`` branch using the same **Create release** workflow.

1. Go to the ``Actions`` tab on GitHub, select **Create release**, click **Run workflow**.
2. Enter the release branch (e.g. ``release/v1.4``) and choose ``rc`` as the release type.
3. The workflow computes the next available rc version of the form ``X.Y.ZrcN`` based on existing tags, commits the version bump on the release branch, tags it as ``vX.Y.ZrcN`` and creates a GitHub *pre-release*. The package is published to PyPI automatically.
4. Anyone can install the release candidate with::

       pip install hydromt==X.Y.ZrcN

   The exact install command is shown in the body of the GitHub pre-release.

.. note::
   Release candidates are built from a ``release/vX.Y`` branch only. They share the same long-lived branch as the eventual full release, so the rc commits become part of the release history.

.. warning::
   Pre-release versions are marked as pre-releases on GitHub and are not promoted as the latest stable release. They are intended for testing purposes only. Do not use a release candidate as the basis for a full release; run **Create release** with type ``major`` / ``minor`` / ``patch`` to produce the actual release.


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
