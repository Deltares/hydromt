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
The data is currently only stored on the deltares server: p:/wflow_global/hydromt

preferred data formats to download
-----------------------------------
vector data: flatgeobuf (because they contain a spatial index and are therefore much faster)
raster data (2D): cloud optimized geotiff
raster data (3D): zarr

data storage (p:/wflow_global/hydromt)
--------------------------------------

data used by the geoserver:
DO NOT CHAGE WITHOUT CONSULTATION
- alosdem
- copdem

writing convention:
- lower case
- with underscores

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
- lower case
- with underscore
- the key "data_type" follows this convention but the data type itself is written in cam case (RasterDataset/GeoDataFrame/GeoDataset)
- two spaces for indentation

data versioning:
- data always refers to a specific version
- version is indicated within the name of the alias
- short name refers to that version
- convention: [data_name]_v[version_number]
- e.g. eobs_v22.0e

structure per data set:
- use placeholders where possible
- order the data sets alphabetically
- order the components of each data set alphabetically
- for adding meta data use the following optional keys:

category:
notes:
paper_doi:
paper_ref:
source_author: (if different from paper_ref)
source_license:
source_url:
source_version:
unit:

updates
-------

- create new branch on github
- make a new folder with the name of the version you are going to create
- copy the latest data catalog into the new folder.
- DO NOT modify old catalogs
- bump the version in the global meta section using semantic versioning
- run update_versions.py, this will create a registry file with the versions and hashes of the data catalogs. It is very important that the files have Linux style line endings (LF) as opposed to windows style line endings (CRLF) to keep hashes consistent. If this is not done, pooch will not be able to find the catalogs. This is done automatically for you (CRLF -> LF) if you are updating from windows.
- test your yml file (Can the added/changed data sources be read through HydroMT?)
- create pull request
