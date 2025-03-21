.. _contribute_documentation:

Adding Documentation
====================

There are a few guidelines when adding new documentation, or when refactoring the
current documentation.

- We use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`.
- Code examples or example ``yaml`` files should be tested using the sphinx extension
  ``doctest``.
- New APIs should be added to the ``docs/api`` folder. The builtin ``autosummary``
  and ``toctree`` are used to keep track.
