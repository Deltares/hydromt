========================
Predefined Data Catalogs
========================

This readme file contains information about the predefined data catalogs which be accessed through HydroMT.
The data can be browsed through in the (predefined data catalogs of the HydroMT docs)[https://deltares.github.io/hydromt/latest/user_guide/data_existing_cat.html]
For more information regarding working with data catalogs we refer the user to the (HydroMT user guide)[https://deltares.github.io/hydromt/latest/user_guide/data_main.html]

deltares_data
=============

The HydroMT Deltares Data is managed by the HydroMT team.
For adding new data to the deltares_data.yml please follow the conventions given hereinafter.
The data is currently only stored on the deltares network drive: p:/wflow_global/hydromt

Preferred data formats to download
-----------------------------------
vector data: flatgeobuf (because they contain a spatial index and are therefore much faster)
raster data (2D): cloud optimized geotiff
raster data (3D): zarr

Data storage in the Deltares network drive (p:/wflow_global/hydromt)
--------------------------------------------------------------------

data used by the geoserver:
DO NOT CHANGE WITHOUT CONSULTATION
- alosdem
- copdem

writing convention:
- snake_case (i.e. lower case with underscores)

folder structure:
- no subcategories

 1. data type (category)
 	bathymetry
 	geography
 	hydro
 	hydrography
 	infrastructure
 	landuse
 	meteo
 	ocean
 	sociao_economic
 	soil
 	topography
 2. data name
 	e.g.:
 	era5
 	eobs
 	grdc
 	osm
 	...
(3. parameter)
	e.g.:
	t2m
	msl
	...

deltares_data.yml
------------------
writing convention:
- snake_case (i.e. lower case with underscores)
- the key "data_type" follows this convention but the data type itself is written in cam case (RasterDataset/GeoDataFrame/GeoDataset)
- two spaces for indentation

data versioning:
- data always refers to a specific version
- short name refers to that version
- convention: `[data_name]_v[version_number]` (e.g. `eobs_v22.0e`)

structure per data set:
- use placeholders where possible
- order the data sets alphabetically
- order the components of each data set alphabetically
- for adding meta data use the following optional keys:
```yaml
category:
notes:
paper_doi:
paper_ref:
source_author: (if different from paper_ref)
source_license:
source_url:
source_version:
unit:
```

Updates
-------

To preserve reproducibility for older versions of the catalogs, we DO NOT modify old catalogs in any way. Even adding different whitespaces to the file is problematic because of how they are retrieved. Instead, we use a versioning system for the catalog files themselves. If you want to update one of the catalogs, follow the following steps

1. create new branch on github
2. make a new folder with the name of the version you are going to create
3. add the new version of the data catalog to this new folder, and make sure it is called `data_catalog.yml`. You can also copy the latest data catalog into the new folder and simply edit the copied version.
4. bump the version in the global meta section using semantic versioning
5.  run update_versions.py, this will create a registry file with the versions and SHA256 hashes of the data catalogs. It is very important that the files have Linux style line endings (LF) as opposed to windows style line endings (CRLF) to keep hashes consistent. If this is not done, pooch will not be able to find the catalogs. This is done automatically for you (CRLF -> LF) if you are updating from windows.
6. test your yml file, (for more information on testing see section below).
7. create a pull request targeting the main branch
8. Once the pull request get's merged into the `main` branch, it should be available to all HydroMT users.

Testing
-------
If you want to make catalogs for HydroMT testing purposes, or to update a catalog with new data, we must first ensure that HydroMT can properly read and use your data. To do this, we need to test the data with HydroMT in our CI environment on github. DO NOT modify the `data/catalogs/predefined_catalogs.yml` file to do this. The use of this file is deprecated and it is maintained for backwards compatibility, but should no longer be used.

Testing a new data catalog should be fairly straight forward once it is created. The level of testing we require to add new catalogs can varry depending on the size, importance, and popularity of the set. In order of importance the tests that should be done are:

1. Instantiate a catalog, and retrieve the dataset from it using the appropriate `get_*` function (e.g. `get_rasterdataset` for raster data)
2. The dataset should slice properly in whatever ways are appropriate. (e.g. requesting a dataset with only certain variables should return only that data if the driver supports it)
3. If the dataset requires special logic to merge several parts please add tests to demostrate the correct working of this as well.
4. Units should be acounted for properly using properties such as `unit`, `unti_add`, and `unit_mult` where appropriate
5. Whatever domain specific quality assurance should be done, to avoid for example, rounding errors near the boundaries.

At least, point one needs to be verified locally by the author of the PR, and preferably a test should be made for it in our CI as well. Depending on context we may ask you to verify other points (on this list) as well.
