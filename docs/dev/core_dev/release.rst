.. _create_release:


Creating a release
------------------

1. Go to the `actions` tab on Github, select `Create a release` from the actions listen to the left, then use the `run workflow` button to start the release process. You will be asked whether it will be a `major`, `minor` or `patch` release. Choose the appropriate action.
2. The action you just run will open a new PR for you with a new branch named `release/v<NEW_VERSION>`. (the `NEW_VERSION` will be calculated for you based on which kind of release you selected.) In the new PR, the changelog, hydromt version and sphinx `switcher.json` will be updated for you. Any changes you made to the `pyproject.toml` since the last release will be posted as a comment in the PR. You will need these during the Conda-forge release if there are any.
3. Every commit to this new branch will trigger the creation (and testing) of release artifacts. In our case those are: Documentation and the PyPi package (the conda release happens separately). After the artifacts are created, they will be uploaded to the repository's internal artifact cache. A bot will post links to these created artifacts in the PR which you can use to download and locally inspect them.
4. When you are happy with the release in the PR, you can simply merge it. We suggest naming the commit something like "Release v<NEW_VERSION>"
5. After the PR is merged, you will need to run the `Finalise a new release` action that will publish the latest artifacts created to their respective platform, it will also create a tag and a github release for you automatically.  After this, a bot will open a new PR to the `main` branch, setting the hydromt version back to a dev version, and adding new headers to the `docs/changelog.rst` for unreleased features. The release is now done as far as this repo is concerned.
6. The newly published PyPi package will trigger a new PR to the `HydroMT feedstock repos of conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_.
   Here you can use the comment posted to the release PR to see if the `meta.yml` needs to be updated. Merge the PR to release the new version on conda-forge.
7. celebrate the new release!

Plugin compatibility test
-------------------------

Before finishing a new HydroMT core release, you must run the downstream plugin compatibility test.
This is part of the release gate.
It checks whether the new HydroMT wheel still works with a set of mature plugins.

The workflow builds the actual wheel that would be published on PyPI and installs that wheel into each plugin's Pixi environment.
We do not use an editable install.
This makes sure we test the real release artifact, including packaging metadata and included data files.

For each plugin, the workflow runs in two modes.

In the first mode, HydroMT is installed with --no-deps. (deps=false)
This upgrades only the HydroMT wheel and keeps the plugin's existing, already solved environment unchanged.
This simulates a user upgrading HydroMT in an existing environment.
If this fails, it means the upgrade is not fully drop-in compatible.
These failures must be reviewed, but they are not automatically release blockers.

In the second mode, HydroMT is installed allowing dependency updates. (deps=true)
This allows the environment to re-solve and update third-party packages if needed.
This simulates a clean installation.
If this fails, the release is considered broken and must not be finalised.
These failures are release blockers.

How to run the compatibility test
---------------------------------

1. Make sure your release branch (for example release/vX.Y) is up to date.
2. Go to the GitHub Actions tab.
3. Select the “Downstream plugin compatibility” workflow.
4. Click “Run workflow” and choose the release branch.

If the ``with dependencies`` run fails, you must fix the problem before continuing the release.

If the ``no-deps`` run fails, review the failure and decide what to do.
You may need to restore backward compatibility in core, coordinate a plugin update, or accept that upgrading HydroMT requires re-solving the environment.

Do not finalise the release until all blocking failures are resolved and advisory failures have been reviewed.
