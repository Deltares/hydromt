.. _create_release:


Release process
===============

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
---------------------

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
-------------

Patch releases are made against an already-existing ``release/vX.Y`` branch.
No new branch needs to be created.

If the patch fixes a bug that exists on ``main``, commit the fix to ``main``
first, then cherry-pick the relevant fix commit(s) onto the release branches
that are currently supported (or open a PR targeting the release branch
directly).

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
-----------------

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

Before tagging a major, minor, or patch release, you must run the downstream
plugin compatibility test against the release branch. This is part of the
release gate. It checks whether the new HydroMT wheel still works with a set of
mature plugins.

The workflow builds the actual wheel that would be published on PyPI and installs
that wheel into each plugin's Pixi environment. We do not use an editable
install. This makes sure we test the real release artifact, including packaging
metadata and included data files.

For each plugin, the workflow runs in two modes.

In the first mode, HydroMT is installed with ``--no-deps`` (``deps=false``).
This upgrades only the HydroMT wheel and keeps the plugin's existing, already
solved environment unchanged. This simulates a user upgrading HydroMT in an
existing environment. If this fails, it means the upgrade is not fully drop-in
compatible. These failures must be reviewed, but they are not automatically
release blockers.

In the second mode, HydroMT is installed allowing dependency updates
(``deps=true``). This allows the environment to re-solve and update third-party
packages if needed. This simulates a clean installation. If this fails, the
release is considered broken and must not be finalised. These failures are
release blockers.

How to run the compatibility test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Make sure your release branch (for example ``release/vX.Y``) is up to date.
2. Go to the GitHub Actions tab.
3. Select the **Downstream plugin compatibility** workflow.
4. Click **Run workflow** and choose the release branch.

If the ``with dependencies`` run fails, you must fix the problem before
continuing the release.

If the ``no-deps`` run fails, review the failure and decide what to do. You may
need to restore backward compatibility in core, coordinate a plugin update, or
accept that upgrading HydroMT requires re-solving the environment.

Do not finalise the release until all blocking failures are resolved and advisory
failures have been reviewed.


.. _release_architecture:

Architecture and design
-----------------------

This section describes how the release workflows and branches fit together.

Workflow diagram
^^^^^^^^^^^^^^^^

.. mermaid::

   flowchart TD
       A[Manual dispatch:<br/>create-release-branch.yml<br/>bump = minor or major]
         -->|Creates release/vX.Y at X.Y.0<br/>setup-release.sh: changelog + switcher<br/>main is NOT bumped here| B[release/vX.Y branch at X.Y.0<br/>main unchanged]

       B --> C{Manual dispatch:<br/>create-release.yml<br/>release_type?}

       C -->|major / minor| D1[Tag vX.Y.0 at HEAD<br/>of release branch]
       C -->|patch| D2[setup-release.sh bumps Z+1<br/>commit + tag vX.Y.Z]
       C -->|rc| D3[Commit pre-release version<br/>tag vX.Y.ZrcN]

       D1 --> E[record-release-on-main.sh<br/>opens record-release/v… PR<br/>merges release branch → main<br/>bumps version + changelog + switcher]
       D2 --> E

       D1 --> F{Pre-release?}
       D2 --> F
       D3 --> F

       F -->|No| G1[gh release create --latest=true/false]
       F -->|Yes: rc| G2[gh release create --prerelease]

       G1 --> H{mark_as_latest?}
       G2 --> I[release: published event]

       H -->|Yes| H1[Deploy docs to /vX.Y.Z/<br/>+ update stable symlink]
       H -->|No| H2[Deploy docs to /vX.Y.Z/<br/>no stable update]

       H1 --> I
       H2 --> I

       I -->|auto-trigger| J[publish-pypi.yml<br/>flit build + twine publish]

       E -.->|maintainer merges PR| K[Release branch merged into main<br/>main bumped to X.Y+1.0.dev0]

The PyPI publish and docs deploy fire **directly off the GitHub release**, not
off any merge into main. The ``record-release/v…`` PRs merge the release branch
back into ``main`` (carrying code changes, changelog, and switcher) but are not
on the publishing critical path.


Release families
^^^^^^^^^^^^^^^^

A **release family** is the set of releases that share the same ``MAJOR.MINOR``.
Each ``release/vX.Y`` branch is the home of exactly one family:

- The **1.4 family** lives on ``release/v1.4`` and contains every ``v1.4.*`` tag
  (``v1.4.0``, ``v1.4.1``, …).
- The **1.5 family** lives on ``release/v1.5`` and contains every ``v1.5.*`` tag.
- ``main`` is always preparing the **next** family. When the first release from
  ``release/v1.5`` is merged back via its record-release PR, main is bumped to
  ``1.6.0.dev0``.


Key design rules
^^^^^^^^^^^^^^^^

1. **Main is bumped to the next dev version by the record-release PR.** When a
   full release is published, ``record-release-on-main.sh`` creates a PR that
   merges the release branch back into ``main``. This PR bumps main to
   ``X.(Y+1).0.dev0`` (if it isn't already higher) and adds a fresh
   ``Unreleased`` changelog section. Releases never originate from ``main``;
   they always come from a ``release/vX.Y`` branch.
2. **Release branches are merged back into main after each full release.** Every
   full release (major, minor, and patch) opens a ``record-release/v…`` PR that
   starts from the release branch and targets ``main``. This PR carries any code
   changes that exist on the release branch back into ``main``, along with the
   updated ``docs/changelog.rst`` and ``docs/_static/switcher.json``. The
   version in ``hydromt/__init__.py`` on ``main`` is preserved if it is already
   at or above ``X.(Y+1).0.dev0``; otherwise it is bumped as a safety net. Use a
   **regular merge** (not squash) to preserve the branch relationship in history.
3. **All development lands on ``main`` first.** Features and bugfixes are merged
   into ``main`` via normal PRs. When a fix needs to ship in an older release
   family, cherry-pick the commit that landed on ``main`` for the fix (that is,
   the PR merge result) onto the relevant ``release/vX.Y`` branch(es) and
   dispatch ``create-release.yml`` with ``release_type = patch`` against that
   branch. If a cherry-pick does not apply cleanly, the ``record-release/v…`` PR
   will carry the fix back to ``main`` after the patch release.

The developer dispatching ``create-release.yml`` chooses, via a
``mark_as_latest`` checkbox, whether the GitHub release should be marked as
``latest`` (and the docs ``stable`` symlink updated). For patches on older
families the developer normally **un**\ checks this so that the newest family
keeps owning ``stable``.


How the workflows fit together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **``create-release-branch.yml``**:

  - Creates ``release/vX.Y`` at ``X.Y.0`` using ``setup-release.sh`` (version
    bump, changelog header rename, switcher entry).
  - Main is **not** bumped here. The version bump and fresh Unreleased section
    are applied when the record-release PR is merged after the first release from
    the family.
  - For ``bump = minor``, takes main's current version as-is (main is already at
    the right minor). For ``bump = major``, bumps the major and resets minor to
    0.
  - Run once per family. Older release branches keep living independently.

- **``create-release.yml``** takes ``release_branch``, ``release_type``, and
  ``mark_as_latest`` as inputs. The same workflow services every release branch
  and every release type uniformly.

  - For ``major``/``minor``: tags the ``X.Y.0`` commit already on the branch.
  - For ``patch``: runs ``setup-release.sh`` to bump the patch, commit, then
    tags.
  - For ``rc``: commits a pre-release version bump, tags, creates a pre-release
    on GitHub. No record-on-main PR; no docs published.
  - For all full releases: runs ``record-release-on-main.sh`` to open a
    ``record-release/v…`` PR that merges the release branch back into main (code
    changes + changelog + switcher). The PR branch starts from the release
    branch, not from main.
  - **``mark_as_latest`` checkbox** controls the GitHub release ``--latest`` flag
    and whether the docs ``stable`` symlink is updated. Default ``true``. Uncheck
    for patches on older families.

- The **``NEW_RELEASE`` concurrency group** serializes release jobs across
  branches.
- **PyPI**: each tag's ``release: published`` event independently triggers
  ``publish-pypi.yml``. All families get published regardless of
  ``mark_as_latest``.
- **Bugfix cherry-picks**: there is no automated bugfix-commit backport workflow.
  The fix is cherry-picked onto each release branch by hand (or via a PR
  targeting the release branch). Only then is ``create-release.yml`` dispatched
  against that branch. If a cherry-pick doesn't apply cleanly, the fix can be
  applied directly to the release branch — the ``record-release/v…`` PR will
  carry it back to main after the release.
- If ``create-release-branch.yml`` fails after pushing the release branch, the
  release can still proceed normally — the record-release PR will bump main when
  the release is published.


Example git history
^^^^^^^^^^^^^^^^^^^

A concrete two-family scenario: a bugfix backported from ``main`` to
``release/v1.5`` and ``release/v1.4``.

.. mermaid::

   gitGraph
      commit id: "feature A (main at 1.4.0)"
      branch "release/v1.4"
      commit id: "Bump 1.4.0 + changelog" tag: "v1.4.0"
      branch "record-release/v1.4.0"
      commit id: "Update switcher + changelog (1.4.0)"
      checkout main
      merge "record-release/v1.4.0" id: "Merge record-release/v1.4.0 (main → 1.5.0.dev0)"
      commit id: "feature B"
      commit id: "feature C"
      branch "release/v1.5"
      commit id: "Bump 1.5.0 + changelog" tag: "v1.5.0"
      branch "record-release/v1.5.0"
      commit id: "add v1.5.0 to main's switcher / changelog"
      checkout main
      merge "record-release/v1.5.0" id: "Merge record-release/v1.5.0 (main → 1.6.0.dev0)"
      commit id: "feature D"
      commit id: "Bugfix PR" type: HIGHLIGHT
      checkout "release/v1.5"
      cherry-pick id: "Bugfix PR"
      checkout main
      commit id: "Feature E"
      checkout "release/v1.5"
      commit id: "Bump 1.5.1 + changelog" tag: "v1.5.1"
      branch "record-release/v1.5.1"
      commit id: "add v1.5.1 to main's switcher / changelog"
      checkout "release/v1.4"
      cherry-pick id: "Bugfix PR"
      checkout main
      commit id: "Feature F"
      checkout "release/v1.4"
      commit id: "Bump 1.4.1 + changelog" tag: "v1.4.1"
      branch "record-release/v1.4.1"
      commit id: "add v1.4.1 to main's switcher / changelog"
      checkout main
      merge "record-release/v1.5.1" id: "Merge record-release/v1.5.1 (main stays 1.6.0.dev0)"
      merge "record-release/v1.4.1" id: "Merge record-release/v1.4.1 (main stays 1.6.0.dev0)"
