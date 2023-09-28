# -*- coding: utf-8 -*-
"""HydroMT VectorModel class definition."""

import logging
import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from ..vector import GeoDataset
from .model_api import Model, _check_equal

__all__ = ["VectorModel"]
logger = logging.getLogger(__name__)


class VectorMixin:
    _API = {"vector": xr.Dataset}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vector = None  # xr.Dataset()

    @property
    def vector(self) -> xr.Dataset:
        """Model vector (polygon) data.

        Returns xr.Dataset with a polygon geometry coordinate.
        """
        if self._vector is None:
            self._vector = xr.Dataset()
            if self._read:
                self.read_vector()
        return self._vector

    def set_vector(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray, gpd.GeoDataFrame] = None,
        name: Optional[str] = None,
        overwrite_geom: bool = False,
    ) -> None:
        """Add data to vector.

        All layers of data must have identical spatial index.
        Only polygon geometry is supported.

        If vector already contains a geometry layer different than data,
        `overwrite_geom` can be set to True to overwrite the complete vector
        object wit data (use with caution as previous data could be lost).

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset or np.ndarray or gpd.GeoDataFrame
            new data to add to vector
        name: str, optional
            Name of new data, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        overwrite_geom: bool, optional
            If True, overwrite the complete vector object with data, by default False
        """
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")

        # check the type of data
        if isinstance(data, np.ndarray) and "geometry" in self.vector:
            index_dim = self.vector.vector.index_dim
            index = self.vector[index_dim]
            if data.size != index.size:
                if data.ndim == 1:
                    raise ValueError("Size of data and number of vector do not match")
                else:
                    raise ValueError(
                        "set_vector with np.ndarray is only supported if data is 1D"
                    )
            data = xr.DataArray(dims=[index_dim], data=data)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif isinstance(data, gpd.GeoDataFrame):
            data = GeoDataset.from_gdf(data, keep_cols=True, cols_as_data_vars=True)
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        # Add to vector
        # 1. self.vector does not have a geometry yet or overwrite_geom
        # then use data directly
        if self.geometry is None or overwrite_geom:
            if data.vector.geometry is None:
                raise ValueError("Cannot instantiate vector without geometry in data")
            else:
                # check that geometry is of type polygon
                if data.vector.geom_type != "Polygon":
                    raise ValueError("Geometry of vector must be of type Polygon")
                if overwrite_geom:
                    self.logger.warning("Overwriting vector object with data")
                self._vector = data
        # 2. self.vector has a geometry
        else:
            # data has a geometry - check if it is the same as self.vector
            if data.vector.geometry is not None:
                if not np.all(
                    data.vector.geometry.geom_almost_equals(self.geometry, decimal=4)
                ):
                    raise ValueError("Geometry of data and vector do not match")
            # add data (with check on index)
            for dvar in data.data_vars:
                if dvar in self.vector:
                    self.logger.warning(f"Replacing vector variable: {dvar}")
                # check on index coordinate before merging
                dims = data[dvar].dims
                if np.array_equal(
                    data[dims[0]].values, self.vector[self.index_dim].values
                ):
                    self._vector[dvar] = data[dvar]
                else:
                    raise ValueError(
                        f"Index coordinate of data variable {dvar} "
                        "does not match vector index coordinate"
                    )

    def read_vector(
        self,
        fn: str = "vector/vector.nc",
        fn_geom: str = "vector/vector.geojson",
        **kwargs,
    ) -> None:
        """Read model vector from combined netcdf and geojson file.

        Files are read at <root>/<fn> and geojson file at <root>/<fn_geom>.

        Three options are possible:

            * The netcdf file contains the attribute data and the geojson file the
            geometry vector data.

            * The netcdf file contains both the attribute and the geometry data.
            (fn_geom is ignored)

            * The geojson file contains both the attribute and the geometry data.
            (fn is ignored)

        Key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root,
            by default 'vector/vector.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
        **kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self._assert_read_mode()
        if fn is not None:
            # Disable lazy loading of data
            # to avoid issues with reading object dtype data
            if "chunks" not in kwargs:
                kwargs["chunks"] = None
            ds = xr.merge(self.read_nc(fn, **kwargs).values())
            # check if ds is empty (default fn has a value)
            if len(ds.sizes) == 0:
                fn = None
        if fn_geom is not None and isfile(join(self.root, fn_geom)):
            gdf = gpd.read_file(join(self.root, fn_geom))
            # geom + netcdf data
            if fn is not None:
                ds = GeoDataset.from_gdf(gdf, data_vars=ds)
            # geom only
            else:
                ds = GeoDataset.from_gdf(gdf, keep_cols=True, cols_as_data_vars=True)
        # netcdf only
        elif fn is not None:
            ds = GeoDataset.from_netcdf(ds)
        else:
            self.logger.info("No vector data found, skip reading.")
            return

        self.set_vector(ds)

    def write_vector(
        self,
        fn: str = "vector/vector.nc",
        fn_geom: str = "vector/vector.geojson",
        ogr_compliant: bool = False,
        **kwargs,
    ):
        """Write model vector to combined netcdf and geojson files.

        Files are written at <root>/<fn> and at <root>/<fn_geom> respectively.

        Three options are possible:

            * The netcdf file contains the attribute data and the geojson file the
                geometry vector data. Key-word arguments are passed to
                :py:meth:`~hydromt.models.Model.write_nc`

            * The netcdf file contains both the attribute and the geometry data
                (fn_geom set to None). Key-word arguments are passed to
                :py:meth:`~hydromt.models.Model.write_nc`

            * The geojson file contains both the attribute and the geometry data
                (fn set to None). This option is possible only if all data variables
                are 1D. Else the data will be written to netcdf. Key-word arguments are
                passed to :py:meth:`~hydromt.vector.GeoDataset.to_gdf`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root,
            by default 'vector/vector.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
        ogr_compliant : bool
            If fn only, write the netCDF4 file in an ogr compliant format
            This makes it readable as a vector file in e.g. QGIS
            see :py:meth:`~hydromt.vector.GeoBase.ogr_compliant` for more details.
        **kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        if len(self.vector) == 0:
            self.logger.debug("No vector data found, skip writing.")
            return
        self._assert_write_mode()
        ds = self.vector

        # If fn is None check that vector contains only 1D data
        if fn is None:
            # If the user did specify a reducer, data can be more than 1D data
            if "reducer" in kwargs and kwargs["reducer"] is not None:
                self.logger.warning(
                    "If 2D/3D data found,"
                    f"they will be reduced to 1D using {kwargs['reducer']}"
                )
            else:
                # If no reducer then check if 1D data only is present
                snames = ["y_name", "x_name", "index_dim", "geom_name"]
                sdims = [ds.vector.attrs.get(n) for n in snames if n in ds.vector.attrs]
                if "spatial_ref" in ds:
                    sdims.append("spatial_ref")
                for name in ds.vector._all_names:
                    dims = ds[name].dims
                    if name not in sdims:
                        # check 1D variables with matching index_dim
                        if len(dims) > 1 or dims[0] != ds.vector.index_dim:
                            fn = join(
                                dirname(join(self.root, fn_geom)),
                                f"{basename(fn_geom).split('.')[0]}.nc",
                            )
                            self.logger.warning(
                                "2D data found in vector,"
                                "will write data to {fn} instead."
                            )
                            break

        # write to netcdf only
        if fn_geom is None:
            if not isdir(dirname(join(self.root, fn))):
                os.makedirs(dirname(join(self.root, fn)))
            # cannot call directly ds.vector.to_netcdf
            # because of possible PermissionError
            if ogr_compliant:
                ds = ds.vector.ogr_compliant()
            else:
                ds = ds.vector.update_geometry(geom_format="wkt", geom_name="ogc_wkt")
            nc_dict = {"vector": ds}
            self.write_nc(nc_dict, fn, engine="netcdf4", **kwargs)
        # write to geojson only
        elif fn is None:
            if not isdir(dirname(join(self.root, fn_geom))):
                os.makedirs(dirname(join(self.root, fn_geom)))
            gdf = ds.vector.to_gdf(**kwargs)
            gdf.to_file(join(self.root, fn_geom))
        # write data to netcdf and geometry to geojson
        else:
            if not isdir(dirname(join(self.root, fn_geom))):
                os.makedirs(dirname(join(self.root, fn_geom)))
            # write geometry
            gdf = ds.vector.geometry.to_frame("geometry")
            gdf.to_file(join(self.root, fn_geom))
            # write_nc requires dict - use dummy key
            nc_dict = {"vector": ds.drop_vars("geometry")}
            self.write_nc(nc_dict, fn, **kwargs)

    # Other vector properties
    @property
    def geometry(self) -> gpd.GeoSeries:
        """Returns the geometry of the model vector as gpd.GeoSeries."""
        # check if vector is empty
        if len(self.vector.sizes) == 0:
            return None
        else:
            return self.vector.vector.geometry

    @property
    def index_dim(self) -> str:
        """Returns the index dimension of the model vector."""
        return self.vector.vector.index_dim


class VectorModel(VectorMixin, Model):

    """Model class Vector Model for vector (polygons) models in HydroMT."""

    _CLI_ARGS = {"region": "setup_region"}
    _NAME = "vector_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a VectorModel for lumped and semi-distributed models."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    def read(
        self,
        components: List = None,
    ) -> None:
        """Read the complete model from model files.

        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an
            associated read_<component> method.
            By default ['config', 'maps', 'vector', 'geoms', 'tables',
            'forcing', 'states', 'results']
        """
        components = components or [
            "config",
            "vector",
            "geoms",
            "tables",
            "forcing",
            "states",
            "results",
        ]
        super().read(components=components)

    def write(
        self,
        components: List = None,
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an
            associated write_<component> method. By default ['config',
            'maps', 'vector', 'geoms', 'tables', 'forcing', 'states']
        """
        components = components or [
            "config",
            "vector",
            "geoms",
            "tables",
            "forcing",
            "states",
        ]
        super().write(components=components)

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.vector) > 0:
            gdf = self.geometry.to_frame("geometry")
            region = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)
        return region

    def _test_equal(self, other, skip_component=None) -> Tuple[bool, Dict]:
        """Test if two models including their data components are equal.

        Parameters
        ----------
        other : Model (or subclass)
            Model to compare against
        skip_component: list
            List of components to skip when testing equality. By default root.

        Returns
        -------
        equal: bool
            True if equal
        errors: dict
            Dictionary with errors per model component which is not equal
        """
        skip_component = skip_component or []
        if "vector" not in skip_component:
            # add vector to skip_component list
            skip_component.append("vector")

        equal, errors = super()._test_equal(other, skip_component=skip_component)

        # test vector separately especially for the geometry
        geods = self.vector
        geods_other = other.vector
        # test geometry
        gdf = geods.vector.geometry.to_frame("geometry")
        gdf_other = geods_other.vector.geometry.to_frame("geometry")
        errors.update(_check_equal(gdf, gdf_other, name="vector.geometry"))
        # test vector data only
        if geods.vector.geom_format == "xy":
            drop_vars = [geods.vector.x_name, geods.vector.y_name]
        else:
            drop_vars = [geods.vector.geom_name]
        ds = geods.drop_vars(drop_vars)
        ds_other = geods_other.drop_vars(drop_vars)
        errors.update(_check_equal(ds, ds_other, name="vector.data"))

        return len(errors) == 0, errors
