# All Model attributes

`_API`: list of all components in the model. Will be obsolete with the new components structure.
`_CF`: never used
`_CLI_ARGS`: sort of default arguments. Mainly used per component to specify what kind of region to create.
`_CONF`: default config filename if none provided, goes into ConfigComponent.
`_DATADIR`: never initialized, so always ""?. Used as a root directory when creating the config file path.
`_FOLDERS`: never used.
`_GEOMS`: never used.
`_MAPS`: never used.
`_MODEL_VERSION`: Not documented and never used.
`_NAME`: Overwritten by plugins. Used to registrer the plugins.
`_TMP_DATA_DIR`: Static variable used to store the temporary data directory. Only initialized when no permission.
`_assert_read_mode`: Throw an error when not in read mode. Should go to `ModelComponent.read`, and overwriters should call `super().read`.
`_assert_write_mode`: Throw an error when not in write mode. Should go to `ModelComponent.write`, and overwriters should call `super().write`.
`_check_get_opt`: Check function to see if all the components mentioned in the config file are methods. This method will stay, but it will check the list of components instead of methods of the model class.
`_cleanup`: part of the defered files. Perhaps take a look at [Context Managers](https://book.pythontips.com/en/latest/context_managers.html).
`_config`: Move to ConfigComponent.
`_config_fn`: Move to ConfigComponent.
`_configread`: Move to ConfigComponent.
`_configwrite`: Move to ConfigComponent.
`_defered_file_closes`: part of the defered files. Perhaps take a look at [Context Managers](https://book.pythontips.com/en/latest/context_managers.html).Probably requires an implementation per component.
`_forcing`: Move to ForcingComponent.
`_geoms`: Move to GeomsComponent.
`_get_sub_method`: Remove.
`_initialize_config`: Move to ConfigComponent.
`_initialize_forcing`: Move to ForcingComponent.
`_initialize_geoms`: Move to GeomsComponent.
`_initialize_maps`: Move to MapsComponent.
`_initialize_results`: Move to ResultsComponent.
`_initialize_states`: Move to StatesComponent.
`_initialize_tables`: Move to TablesComponent.
`_maps`: Move to MapsComponent.
`_results`: Move to ResultsComponent.
`_run_log_method`: Stay and modify.
`_states`: Move to StatesComponent.
`_staticgeoms`: deprecated. Should be removed.
`_staticmaps`: deprecated. Should be removed.
`_tables`: Move to TablesComponent.
`_test_equal`: This should probably be removed. It looks like a function that is only used internally for tests. I'd rather have this in some testutils if needed.
`_test_model_api`: Never used. I'd rather have this in some testutils if needed.
`api`: Used in `_test_equal` and `_test_model_api`. Should be moved to testutils.
`bounds`: Move to RegionComponent.
`build`: Will stay. This is the main API function together with `update`.
`config`: Move to ConfigComponent.
`coords`: Part of GridComponent.
`crs`: Move to GridComponent or RegionComponent.
`data_catalog`: Can be private.
`dims`: Move to GridComponent.
`forcing`: Move to ForcingComponent.
`geoms: Move to GeomsComponent.
`get_config`: Move to ConfigComponent.
`get_tables_merged`: Move to TablesComponent.
`height`: Move to GridComponent.
`logger`: Private.
`maps`: Move to MapsComponent.
`read`: This will be the function that reads all the components. There is now some trickery with function names. That should be removed. Should be private.
`read_config`: Move to ConfigComponent.
`read_forcing`: Move to ForcingComponent.
`read_geoms`: Move to GeomsComponent.
`read_maps`: Move to MapsComponent.
`read_nc`: Duplicate of`readers.py.read_nc`. Should be removed.
`read_results`: Move to ResultsComponent.
`read_states`: Move to StatesComponent.
`read_staticgeoms`: Deprecated, remove.
`read_staticmaps`: Deprecated, remove.
`read_tables`: Move to TablesComponent.
`region`: Move to RegionComponent.
`res`: Remove, deprecated.
`results`: Move to ResultsComponent.
`root`: Will stay.
`set_config`: Move to ConfigComponent.
`set_crs`: Deprecated, remove.
`set_forcing`: Move to ForcingComponent.
`set_geoms`: Move to GeomsComponent.
`set_maps`: Move to MapsComponent.
`set_results`: Move to ResultsComponent.
`set_states`: Move to StatesComponent.
`set_staticgeoms`: Move to GeomsComponent.
`set_staticmaps`: Move to GridComponent.
`set_tables`: Move to TablesComponent.
`setup_config`: Move to ConfigComponent.
`setup_maps_from_raster_reclass`: Move to MapsComponent.
`setup_maps_from_rasterdataset`: Move to MapsComponent.
`shape`: Move to StaticMapsComponent.
`states`: This will become a component.
`staticgeoms`: deprecated. Should be removed.
`staticmaps`: deprecated. Should be removed.
`tables`: This will become a component.
`test_model_api`: deprecated. Should be removed.
`transform`: Function will go to StaticMapsComponent.
`update`: Will stay. This is the main API function together with`build`.
`width`: Function will go to StaticMapsComponent.
`write`: This will be the function that writes all the components. There is now some trickery with function names. That should be removed. Should be private.
`write_config`: Could be kept as a private function. Eventually it is the specification of which components should be in the model.
`write_data_catalog`: Not really part of a model. Should be moved out to DataCatalog itself.
`write_forcing`: This will become a component
`write_geoms`: This will become a component
`write_maps`: This will become a component
`write_nc`: Duplicate of`writes.py:write_nc`. Should be removed.
`write_states`: This will become a component
`write_staticgeoms`: This will become a component
`write_staticmaps`: This will become a component
`write_tables`: This will become a component
