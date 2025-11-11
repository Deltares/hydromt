


Test and CI
-----------

We use `pytest <https://pytest.org>`__ for testing and `github actions <https://docs.github.com/en/actions>`_ for CI.
- Unit tests are mandatory for new methods and workflows and integration tests are highly recommended for various
- All tests should be contained in the tests directory in functions named `test_*`.
- We use `SonarQube <https://sonarcloud.io/project/overview?id=Deltares_hydromt>`_ to monitor the coverage of the tests and aim for high (>90%) coverage. This is work in progress.
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
