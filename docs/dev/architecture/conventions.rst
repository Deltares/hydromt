.. _architecture_conventions:

HydroMT Design Conventions
--------------------------

General
^^^^^^^
- HydroMT follows consistent :ref:`naming and unit conventions <data_convention>` for frequently used variables to ensure clarity and interoperability.
- Code and documentation should adhere to Pythonic naming standards (PEP8), and public API elements should have clear docstrings following the NumPy style.

Data
^^^^
- HydroMT supports a range of :ref:`data types <data_types>`, which can be extended as needed.
- Input data is defined in a :ref:`data catalog <data_yaml>` and parsed by HydroMT to its associated Python type through the ``DataSource`` class.
- The goal of the ``DataAdapter`` is to standardize internal data representation — including variable names, units, and structure — with minimal preprocessing.
- When accessing data from the catalog via any ``DataCatalog.get_<data_type>`` method, the adapter ensures a consistent and unified format.
- The ``get_*`` methods also support arguments to define spatial or temporal subsets of datasets, ensuring efficient and targeted data access.

Model Class
^^^^^^^^^^^
The HydroMT :ref:`Model class <model_api>` defines the structure and behavior of models within the framework.
To implement HydroMT for a specific model kernel or software, create a subclass named ``<Name>Model`` (e.g., ``SfincsModel`` for SFINCS) with model-specific
readers, writers, and setup methods.

- :ref:`Model components <model_components>` are data attributes that together define a model instance.
  Each component represents a specific aspect (file/data) of the model and is parsed into a Python class and data object with predefined specifications.
  For example, the ``GridComponent`` data represents static regular grids of a model as an :py:class:`xarray.Dataset`.

- Most model components include both ``read`` and ``write`` methods for handling model-specific formats.
  These methods may include optional keyword arguments but **must not** require positional arguments.
  Model outputs can also be handled through components but should not implement a ``write`` method.

- The ``Model`` should contain high level methods that go from raw data into model inputs and parameters.
  These methods are decorated with ``@hydromt_step`` to indicate they are part of the model workflow.
  Each method should have a clear purpose, and documented inputs and outputs.

- All public model methods, defined with ``hydromt_step`` may only accept arguments of basic Python types: ``str``, ``int``,
  ``float``, ``bool``, ``None``, ``list``, or ``dict``.
  This restriction ensures methods can be fully defined in a :ref:`workflow YAML file <model_workflow>`.

- Model methods access data through the ``Model.data_catalog`` attribute — an instance of :py:class:`hydromt.DataCatalog`.
  Any argument ending with ``_fn`` (short for *filename*) refers either to a source in the data catalog or to a file path.
  Inside the method, data can be read with any ``DataCatalog.get_<data_type>`` method, which handles both catalog entries and local file paths transparently.

- The Model class defines two high-level methods — :py:meth:`~hydromt.Model.build` and :py:meth:`~hydromt.Model.update` — which
  are available across all model plugins and exposed via the CLI. Additional high-level methods may be added in future releases.

- A model subclass can be exposed as a HydroMT plugin by declaring a ``hydromt.models``
  `entry point <https://packaging.python.org/en/latest/specifications/entry-points/>`_ in the package's ``pyproject.toml``.
  For detailed instructions, refer to the :ref:`register_plugins` section.

- We strongly recommend writing integration and unit tests for all model classes and components to ensure correctness and maintain stability across releases.
