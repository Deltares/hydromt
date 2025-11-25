.. _create_release:


Creating a release
------------------

1. Go to the `actions` tab on Github, select `Create a release` from the actions listen to the left, then use the `run workflow` button to start the release process. You will be asked whether it will be a `major`, `minor` or `patch` release. Choose the appropriate action.
2. The action you just run will open a new PR for you with a new branch named `release/v<NEW_VERSION>`. (the `NEW_VERSION` will be calculated for you based on which kind of release you selected.) In the new PR, the changelog, hydromt version and sphinx `switcher.json` will be updated for you. Any changes you made to the `pyproject.toml` since the last release will be posted as a comment in the PR. You will need these during the Conda-forge release if there are any.
3. Every commit to this new branch will trigger the creation (and testing) of release artifacts. In our case those are: Documentation, the PyPi package and docker image (the conda release happens separately). After the artifacts are created, they will be uploaded to the repository's internal artifact cache. A bot will post links to these created artifacts in the PR which you can use to download and locally inspect them.
4. When you are happy with the release in the PR, you can simply merge it. We suggest naming the commit something like "Release v<NEW_VERSION>"
5. After the PR is merged, you will need to run the `Finalise a new release` action that will publish the latest artifacts created to their respective platform, it will also create a tag and a github release for you automatically.  After this, a bot will open a new PR to the `main` branch, setting the hydromt version back to a dev version, and adding new headers to the `docs/changelog.rst` for unreleased features. The release is now done as far as this repo is concerned.
6. The newly published PyPi package will trigger a new PR to the `HydroMT feedstock repos of conda-forge <https://github.com/conda-forge/hydromt-feedstock>`_.
   Here you can use the comment posted to the release PR to see if the `meta.yml` needs to be updated. Merge the PR to release the new version on conda-forge.
7. celebrate the new release!
