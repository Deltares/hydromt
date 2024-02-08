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
| lake_atlas_pol_v10             | hydro_lake_atlas          |
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
