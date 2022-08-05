==============================
Change log predefined datasets
==============================

deltares_data 
=============

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