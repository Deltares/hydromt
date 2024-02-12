==============================
Change log predefined datasets
==============================

deltares_data
=============

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
