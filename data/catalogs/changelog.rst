==============================
Change log predefined datasets
==============================

Pre-defined catalog refactor
============================

Changed
-------
The pre-defined data catalogs have been moved to their own folders together with
a version.yml. This file describes the different versions and relative paths of
the data catalogs present in the folders.

- predefined_catalogs.yml is deprecated
- data catalog versions are associated with the SHA-256 hash of the files.
- deltares_data versioning is now in semantic style

Added
-----
- v0.0.6, v0.0.7, v0.0.9 of the artifact_data
- v0.5.0, v0.6.0 of the deltares_data


deltares_data
=============

version v0.7.0
--------------

added
^^^^^
- version argument to data source where applicable
- variants argument for data sources that are of the same dataset but different versions
- processing_script / processing_notes arguments to data sources that have been (pre-) processed
- temporal extent of datasets that have a temporal dimension.
- spatial extents to datasets

changed
^^^^^^^
- removed source_version from data source meta
- kwargs to driver_kwargs
- updated source_url if url was not working anymore
- sorted datasets by alphabetical order
- Removed version from dataset names
- prefixed 'hydro' for all HydroSHEDS datasets
- added the epoch of the dataset to the name of the dataset

See table below for mapping of old and new names:

+--------------------------------+---------------------------+
| Old name                       | New name                  |
+================================+===========================+
| basin_atlas_level12_v10        | hydro_basin_atlas_level12 |
+--------------------------------+---------------------------+
| eobs_v..                       | eobs                      |
+--------------------------------+---------------------------+
| eobs_orography_v..             | eobs_orography            |
+--------------------------------+---------------------------+
| lake_atlas_pol_v10             | hydro_lake_atlas_pol      |
+--------------------------------+---------------------------+
| river_atlas_v10                | hydro_river_atlas         |
+--------------------------------+---------------------------+
| ghs_pop_2015_54009_v2019a      | ghs_pop_2015	             |
+--------------------------------+---------------------------+
| ghs_smod_2015_54009_v2019a     | ghs_smod_2015             |
+--------------------------------+---------------------------+
| glofas_era5_v31                | glofas_era5               |
+--------------------------------+---------------------------+
| guf_bld_2012                   | guf_bld_2012              |
+--------------------------------+---------------------------+
| rivers_lin2019_v1              | hydro_rivers_lin2019      |
+--------------------------------+---------------------------+
| SM2RAIN_ASCAT_monthly_025_v1.4 | SM2RAIN_ASCAT_monthly_025 |
+--------------------------------+---------------------------+
| SM2RAIN_ASCAT_monthly_05_v1.4  | SM2RAIN_ASCAT_monthly_05  |
+--------------------------------+---------------------------+
| vito 					         | vito_2015                 |
+--------------------------------+---------------------------+
| vito_2015_v2.0.2			   	 | vito_2015                 |
+--------------------------------+---------------------------+
| vito_2016_v3.0.1               | vito_2016                 |
+--------------------------------+---------------------------+
| vito_2017_v3.0.1               | vito_2017                 |
+--------------------------------+---------------------------+
| vito_2018_v3.0.1               | vito_2018                 |
+--------------------------------+---------------------------+
| vito_2019_v3.0.1               | vito_2019                 |
+--------------------------------+---------------------------+
| esa_worldcover 			     | esa_worldcover_2020       |
+--------------------------------+---------------------------+
| corine 					     | corine_2018               |
+--------------------------------+---------------------------+
| globcover 				     | globcover_2009            |
+--------------------------------+---------------------------+


- Some datasets have multiple versions, for these datasets the default can be changed if
you do not supply a version in your config file. See the table below for which dataset
the default version has changed.

+----------------+-----------------+--------------------------+
| Dataset name   | Default version | Previous default version |
+================+=================+==========================+
| eobs           | 25.0e           | 22.0e                    |
+----------------+-----------------+--------------------------+
| eobs_orography | 25.0e           | 22.0e                    |
+----------------+-----------------+--------------------------+



version: 2024.1.30
---------------

added
^^^^^
- HydroMT version to catalog
- GRDC dataset


version: 2023.12
-----------------

changed
^^^^^^^
- Updated GADM dataset and converted the GeoPackage layers to FlatGeoBuf files

added
^^^^^
- Added waterdemand pcr_globwb dataset
- Added GADM 4.1 as FlatGeoBuff files to deltares_data catalog (#686)


version: 2023.2
----------------

changed
^^^^^^^
- convert GeoPackage files to FlatGeoBuf for cloud compatibility
- fix ERA5 nc files to read from archive of combined yearly and monthly files

added
^^^^^
- Additional variables to era5 daily and hourly with name and unit conventions
	- temp_dew: dewpoint temperature (degree C)
	- wind10_u: 10m wind U-component (m s-1)
	- wind10_v: 10m wind V-component (m s-1)
	- ssr: surface net solar radiation (W m-2)
	- tcc: total cloud cover (-)


version: 2022.7
---------------

added
^^^^^
- README with conventions regarding data (download, storage, .yml)
- change log file of deltares_data.yml
- new data sets
	- basin_atlas_level12_v10
	- river_atlas_v10
	- lake_atlas_pol_v10
	- eobs_v24.0e
	- eobs_v25.0e
	- eobs_orography_v24.0e
	- eobs_orography_v25.0e
	- SM2RAIN_ASCAT_monthly_025_v1.4
	- SM2RAIN_ASCAT_monthly_05_v1.4

changed
^^^^^^^
- Apply convention specified in the README
	- check reasonable alphabetical order in data sets and components
	- implement right versioning convention _v where possible
	- apply consistent meta information

fixed
^^^^^
- enable versioning of yml.files

cmip6_data
==========

version: 2024.1.30
---------------

added
^^^^^
- HydroMT version to catalog

version: 2023.2
---------------

added
^^^^^
- CMIP6 data from Google Cloud Storage. Only models and scenarios for which regular grids are available are listed

aws_data
========

version: 2024.1.30
---------------

added
^^^^^
- HydroMT version to catalog

version: 2023.2
---------------

added
^^^^^
- ESA Worldcover v100 2020.
