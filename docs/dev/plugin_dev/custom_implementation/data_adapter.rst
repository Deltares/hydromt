.. _custom_data_adapter:

================
Custom DataAdapter
================

`DataAdapterBase` is the base class for all HydroMT data adapters.
It defines the core structure and transformation utilities used to standardize and convert data after it has been read by a :ref:`driver_architecture`.

Overview
--------

A DataAdapter is responsible for transforming raw data from a :ref:`driver_architecture` into a consistent HydroMT representation.
This may include renaming variables, applying unit conversions, or adjusting time ranges.

`DataAdapterBase` provides the foundational fields and helper methods for such transformations.

Attributes
----------

- **unit_add**: `dict[str, Any]`
  Dictionary of additive adjustments for variables. Commonly used for temporal offsets.
- **unit_mult**: `dict[str, Any]`
  Dictionary of multiplicative adjustments for variables. Often used for unit conversions.
- **rename**: `dict[str, str]`
  Mapping from source variable names to standardized HydroMT variable names.

Methods
-------

**_to_source_timerange(time_range: Optional[TimeRange]) → Optional[TimeRange]**
Transforms a HydroMT `TimeRange` into the source-native time range.
If `unit_add` contains a time offset, it is subtracted from the start and end times.

Parameters:

- **time_range**: Optional[TimeRange]
  Start and end datetime of the requested HydroMT time range.

Returns:

- Optional[TimeRange] — the transformed time range in source-native units, or `None` if no range was provided.

**_to_source_variables(variables: Optional[List[str]]) → Optional[List[str]]**
Transforms HydroMT variable names into source-native variable names based on the `rename` mapping.

Parameters:

- **variables**: Optional[List[str]]
  List of variable names in HydroMT format.

Returns:

- Optional[List[str]] — List of corresponding source-native variable names.

Usage Notes
-----------

- Subclasses should extend `DataAdapterBase` to implement specific variable transformations, unit conversions, or other dataset-specific logic.
- The adapter works automatically when a :ref:`data_source_architecture` passes loaded data through it.
- `unit_add` and `unit_mult` provide a simple mechanism for adjusting numeric values or temporal offsets consistently across datasets.
- The `rename` dictionary allows standardizing variable names from heterogeneous sources.
