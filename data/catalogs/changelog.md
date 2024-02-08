# Change log predefined datasets

## deltares_data

### version 2024.2

#### added
- version argument to data source where applicable
- variants argument for data sources that are of the same dataset but different versions
- processing_script / processing_notes arguments to data sources that have been (pre-) processed
- temporal extent of datasets that have a temporal dimension.
- spatial extents to datasets

#### changed
- removed source_version from data source meta
- kwargs to driver_kwargs
- updated source_url if url was not working anymore
- sorted datasets by alphabetical order
- Removed version from dataset names and prefixed 'hydro' where necessary, see table below for mapping of old and new names.

| Old name                       | New name                  |
|--------------------------------|---------------------------|
| basin_atlas_level12_v10        | hydro_basin_atlas_level12 |
| eobs_v..                       | eobs                      |
| eobs_orography_v..             | eobs_orography            |
| lake_atlas_pol_v10             | hydro_lake_atlas_pol      |
| river_atlas_v10                | hydro_river_atlas         |
| ghs_pop_2015_54009_v2019a      | ghs_pop_2015_54009        |
| glofas_era5_v31                | glofas_era5               |
| guf_bld_2012                   | guf                       |
| rivers_lin2019_v1              | hydro_rivers_lin2019      |
| SM2RAIN_ASCAT_monthly_025_v1.4 | SM2RAIN_ASCAT_monthly_025 |
| SM2RAIN_ASCAT_monthly_05_v1.4  | SM2RAIN_ASCAT_monthly_05  |
| soilgrids_2020                 | soilgrids                 |
| vito_2016_v3.0.1               | vito_2016                 |
| vito_2017_v3.0.1               | vito_2017                 |
| vito_2018_v3.0.1               | vito_2018                 |
| vito_2019_v3.0.1               | vito_2019                 |

- Some datasets have multiple versions, for these datasets the default can be changed if you do not supply a version in your config file. See the table below for which dataset the default version has changed.

| Dataset name   | Default version |
|----------------|-----------------|
| eobs           | 25.0e           |
| eobs_orography | 25.0e           |
| ghs_mod        | R2019A_v2.0     |
| pcr_globwb     | 2005            |
| soilgrids      | 2.0             |



### version 2024.1.30

#### added
- HydroMT version to catalog
- GRDC dataset


### version: 2023.12

#### changed
- updated GADM dataset and converted the GeoPackage layers to FlatGeoBuf files
- removed gtsm_codec_reanalysis dataset

#### added
- Added waterdemand pcr_globwb dataset
- Added GADM 4.1 as FlatGeoBuff files to deltares_data catalog (#686)


### version: 2023.2

#### changed
- convert GeoPackage files to FlatGeoBuf for cloud compatibility
- fix ERA5 nc files to read from archive of combined yearly and monthly files

#### added
- Additional variables to era5 daily and hourly with name and unit conventions
	- temp_dew: dewpoint temperature (degree C)
	- wind10_u: 10m wind U-component (m s-1)
	- wind10_v: 10m wind V-component (m s-1)
	- ssr: surface net solar radiation (W m-2)
	- tcc: total cloud cover (-)


### version: 2022.7

#### added

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

#### changed
- Apply convention specified in the README
	- check reasonable alphabetical order in data sets and components
	- implement right versioning convention _v where possible
	- apply consistent meta information

#### fixed
- enable versioning of yml.files

## cmip6_data

### version: 2024.1.30

#### added
- hydromt_version to data catalog meta


### version: 2023.2

#### added
- CMIP6 data from Google Cloud Storage. Only models and scenarios for which regular grids are available are listed

## aws_data

### version: 2024.1.30

#### added
- HydroMT version to catalog

### version: 2023.2

#### added
- ESA Worldcover v100 2020.
