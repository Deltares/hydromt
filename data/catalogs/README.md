============
hydromt-data
============

The hydromt-data is managed by the hydromt team. Do not add content or edit! 
For addng new data to the deltares_data.yml please follow the conventions given hereinafter.
The deltares_data.yml is stored: 
- on the deltares server: p:/wflow_global/hydromt
- on github: 
The data is only stored on the deltares server: p:/wflow_global/hydromt


preferred data formats to dowonload
===================================
vector data: geopackage or geobuf (because they condtain a spatial index and are therefore much faster)
raster data (2D): cloud optimized geotiff
raster data (3D): zarr


data storage (p:/wflow_global/hydromt)
======================================

data used by the geoserver:
DO NOT CHAGE WITHOUT CONSULTATION
- alosdem 
- copdem 

writung convention:
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
       
deltares_data_.yml
==================
- options: https://deltares.github.io/hydromt/latest/user_guide/data_prepare_cat.html
- data conventions: https://deltares.github.io/hydromt/latest/user_guide/data_conventions.html

writing convention:
- lower case
- with underscore
- the key "data_type" follows this convention but the data type itseld is written in cam case (RasterDataset/GeoDataFrame/GeoDataset)
- two spaces for indentation

data versioning: 	
- data always refers to a specifc verion
- version is indicated within the name of the alias 
- short name refers to that version 
- convention: [data_name]_v[version_number]
- e.g. eobs_v22.0e

structure per data set: 
- use placeholders where possible 
- order the data sets alphabetically
- order the components of each data set alphbetically as follows: 

meta: 
  root: [root_path] (e.g. p:/wflow_global/hydromt)
  version: [year.month] (e.g. 2022.7)

[data_name]: 
  alias: [alias]
[alias]: 
  crs: [crs]
  data_type: [data_type]
  driver: [driver]
  kwargs:
    [kwargs_key: kwargs_value]
    [kwargs_key: kwargs_value]
  meta: 
    [meta_key: meta_value]
    [meta_key: meta_value]
  nodata: [nodata_value]
  path: [path]
  placeholders: 
    [placeholder_key: [placeholder_values]]
  rename:
    [old_variable_name: hydromt_variable_name]
  units: 
    [hydromt_variable_name: value]
  unit_add:
    [hydromt_variable_name: value]
  unit_mult: 
    [hydromt_variable_name: value]

- for adding meta data use the following optional meta_keys: 
	
category:
notes:
paper_doi: 
paper_ref: 
source_author: (if different from paper_ref)
source_license: 
source_url: 
source_version: 
unit:
	
Register a new .yml version 
===========================

- create new branch on github
- make changes 
- create pull request 
- add version to hydtomy\data\predefined_catalogs.yml

	
	
