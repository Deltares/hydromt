.. _custom_behaviour:

================
Custom Resolver
================

HydroMT uses **resolvers** to translate high-level dataset references into the actual locations of files or data resources.
This is necessary when datasets are not stored as a single file or have complex layouts, such as tiled global datasets, APIs, or cloud resources.

For example, the AWS Copernicus DEM dataset (https://registry.opendata.aws/copernicus-dem/) is provided as Cloud Optimized GeoTIFFs without a spatial index.
Files are divided into tiles covering the globe and must be queried by resolution, northing, and easting.
A resolver provides a standardized way to turn such spatial queries into the list of files or URIs needed for processing.

Overview
--------

A **URIResolver** is responsible for generating a list of URIs to the files you wish to read.

- A URI can be a local file path, a URL, a REST API endpoint, a database query, or any resource that your driver can read.
- Avoid assuming that a URI is always a file path; this ensures flexibility and broader compatibility with cloud and networked resources.

HydroMT expects that resolvers only generate the URIs. Actual reading of data is handled by the corresponding driver.

Implementing a Resolver
-----------------------

A resolver must implement one public method: `resolve`.

.. code-block:: python

    from typing import List, Optional, Union
    import geopandas as gpd
    from hydromt.data_catalog import URIResolver, TimeRange, SourceMetadata, NoDataStrategy, Zoom

    class MyCustomResolver(URIResolver):
        def resolve(
            self,
            uri: str,
            *,
            time_range: Optional[TimeRange] = None,
            zoom_level: Optional[Zoom] = None,
            mask: Optional[gpd.GeoDataFrame] = None,
            variables: Union[int, tuple[float, str], None] = None,
            metadata: Optional[SourceMetadata] = None,
            handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        ) -> List[str]:
            """Return a list of URIs corresponding to the requested data."""
            ...

This method should return a list of strings representing the resolved resources.
You may also read metadata or other auxiliary information if needed, but the main responsibility is URI generation.

Resolvers can interact with:

- APIs or web services
- Local or networked file systems
- Predefined static mappings or a-priori lists of URIs

Additional arguments can be added after the `*` for backward compatibility.

Handling Missing Data
---------------------

HydroMT provides the `handle_nodata` argument to standardize missing data behavior.
It is strongly recommended to respect this in your resolver.

- **RAISE** – Immediately raise an exception if the requested data does not exist.
- **WARN** – Emit a warning and return `None`.
- **IGNORE** – Silently return `None` without logging.

Returning `None` instead of an empty dataset helps distinguish between datasets that do not exist versus datasets with missing values.
Subsequent functions in the workflow should propagate `None` as needed.

Example Resolver
----------------

A simple resolver that returns local file paths based on a fixed directory structure:

.. code-block:: python

    class SimpleFileResolver(URIResolver):
        def __init__(self, base_path: str):
            self.base_path = base_path

        def resolve(self, uri: str, handle_nodata: NoDataStrategy = NoDataStrategy.RAISE) -> List[str]:
            path = f"{self.base_path}/{uri}.tif"
            if not os.path.exists(path):
                if handle_nodata == NoDataStrategy.RAISE:
                    raise FileNotFoundError(f"{path} not found")
                elif handle_nodata == NoDataStrategy.WARN:
                    print(f"Warning: {path} not found")
                    return None
                else:
                    return None
            return [path]

Complex resolvers can combine spatial queries, temporal filtering, and multiple sources to generate a full list of URIs for large datasets.
