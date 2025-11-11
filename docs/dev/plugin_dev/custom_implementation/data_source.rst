.. _custom_data_source:

====================
Custom Data Sources
====================

A **DataSource** in HydroMT defines the interface between the data catalog and the lower-level I/O system.
It combines the **URI resolution**, **driver I/O**, and **adapter transformation** layers into a single reusable object.

Where :ref:`custom_resolver` handles locating files and :ref:`custom_driver` manages I/O, a DataSource ensures that:
- Input from a data catalog entry is validated early.
- Resolvers, drivers, and adapters are correctly initialized and connected.
- The dataset can be read, exported, or converted into a standardized representation.

DataSources are dataset-type-specific (e.g., raster, vector, or time series) and define how HydroMT interacts with those datasets consistently.


Overview
--------

Each subclass of :class:`hydromt.data_catalog.sources.DataSource` represents a specific data type (e.g., `RasterDatasetSource`, `GeoDatasetSource`).
Subclasses must define the class variable `data_type`, specify a fallback driver, and implement the abstract methods required for reading and exporting data.

When the catalog is parsed, HydroMT validates all DataSource definitions and ensures that:
1. The declared `data_type` matches the subclass.
2. The appropriate driver is selected (inferred automatically if not provided).
3. The filesystem configuration is consistent between the driver and resolver.

This validation ensures workflow consistency and allows for predictable failure early in the model setup process.


Implementing a Custom DataSource
--------------------------------

To create a new DataSource type, subclass :class:`DataSource` and implement the required methods:

.. code-block:: python

    from hydromt.data_catalog.sources import DataSource
    from hydromt.data_catalog.drivers import BaseDriver
    from hydromt.data_catalog.uri_resolvers import URIResolver
    from hydromt.data_catalog.adapters import DataAdapterBase
    from hydromt.error import NoDataStrategy
    from pystac import Catalog as StacCatalog
    from pathlib import Path

    class MyCustomSource(DataSource):
        """Example DataSource implementation for a new data type."""

        data_type = "custom"
        _fallback_driver_read = "mydriver"
        _fallback_driver_write = "mydriver"

        def read_data(self, **kwargs):
            """Read data from the source and return a standardized object."""
            uris = self.uri_resolver.resolve(self.full_uri, **kwargs)
            data = self.driver.read(uris, **kwargs)
            return self.data_adapter.transform(data)

        def to_stac_catalog(
            self,
            handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
        ) -> StacCatalog | None:
            """Convert source into a STAC catalog."""
            # Example minimal STAC export
            catalog = StacCatalog(id=self.name, description="My custom data source")
            return catalog

        def to_file(self, file_path: Path | str):
            """Write the dataset to disk and return a new DataSource instance."""
            output_path = Path(file_path)
            self.driver.write(output_path, self.read_data())
            new_uri = str(output_path)
            return MyCustomSource(
                name=self.name,
                uri=new_uri,
                driver=self.driver,
                data_adapter=self.data_adapter,
                uri_resolver=self.uri_resolver,
                metadata=self.metadata,
            )


Core Responsibilities
---------------------

Each DataSource is responsible for:

- **Validation:** Ensuring that the `data_type`, driver, and resolver match expectations before use.
- **Connection:** Maintaining consistency between the resolver's and driver's filesystem definitions.
- **Resolution:** Combining `root` and `uri` into a fully qualified path or URL (`full_uri` property).
- **Standardization:** Delegating transformations to the associated `DataAdapterBase`.
- **Lifecycle management:** Marking sources as used and logging read operations for traceability.

HydroMT ensures that every source is only used once per workflow unless explicitly reused.


Abstract Methods
----------------

Every subclass must implement the following abstract methods:

- `read_data()` — orchestrates data resolution, reading, and transformation.
- `to_stac_catalog()` — converts the dataset into a STAC Catalog or returns `None` if not applicable.
- `to_file()` — writes the dataset to a file and returns a new DataSource pointing to the exported location.

Optionally, subclasses may override:

- `_detect_time_range()` — detect time bounds when not defined in metadata.
- `_detect_bbox()` — detect spatial extent and CRS if available.


Properties and Validation Logic
-------------------------------

- **`full_uri`**
  Returns an absolute URI by joining `root` and `uri`. Handles both local and remote sources.

- **`_validate_data_type`**
  Ensures that catalog-specified `data_type` matches the subclass-defined `data_type`.
  Automatically infers a default driver based on file extension if none is specified.

- **`_validate_fs_equal_if_not_set`**
  Ensures the resolver and driver reference the same filesystem unless explicitly overridden.

- **`summary()`**
  Returns a dictionary summarizing key metadata, driver, and data type information.

These built-in validators enforce consistency across catalog definitions and avoid runtime mismatches between components.


Example Catalog Integration
---------------------------

Once implemented, your DataSource can be registered through the catalog YAML configuration:

.. code-block:: yaml

    my_custom_source:
        data_type: custom
        class: my_plugin.data_sources.MyCustomSource
        uri: "https://example.org/my_data"
        driver: mydriver
        metadata:
            category: example
            license: CC-BY-4.0

HydroMT will automatically detect and instantiate your class when reading the catalog.


Example Usage
-------------

.. code-block:: python

    from hydromt import DataCatalog

    catalog = DataCatalog("custom_catalog.yml")
    ds = catalog.get_source("my_custom_source")
    data = ds.read_data()

    print(ds.summary())
    print(data)


Best Practices
--------------

- Keep URI logic isolated to resolvers; do not hardcode path resolution in `read_data`.
- Reuse driver and adapter functionality; avoid duplicating I/O logic.
- Implement `to_stac_catalog()` if your dataset can be described or shared via STAC.
- Define `_fallback_driver_read` and `_fallback_driver_write` to ensure robust defaults.
- Use the built-in logging (`_log_start_read_data`) for consistent tracing.
- Return `None` instead of raising exceptions when appropriate for `NoDataStrategy.IGNORE`.


Summary
-------

A DataSource is the central integration point between catalog metadata and HydroMT's I/O ecosystem.
By subclassing it, you gain full control over how new data types are discovered, read, standardized, and exported—while maintaining compatibility with HydroMT's broader workflow and catalog system.
