# -*- coding: utf-8 -*-
"""HydroMT VectorComponent class definition."""

import os
from os.path import basename, dirname, isfile, join
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS

from hydromt import hydromt_step
from hydromt.components.base import ModelComponent
from hydromt.gis.vector import GeoDataset
from hydromt.io.readers import read_nc
from hydromt.io.writers import write_nc
from hydromt.models.model import Model

__all__ = ["VectorComponent"]


class VectorComponent(ModelComponent):
    """Component to handle vector data in a model."""

    def __init__(
        self,
        model: Model,
        *,
        fn: str = "vector/vector.nc",
        fn_geom: str = "vector/vector.geojson",
    ) -> None:
        """Initialize a vector component.

        Parameters
        ----------
        model : Model
            Parent model
        fn : str, optional
            File name of the vector component, by default "vector/vector.nc"
        fn_geom : str, optional
            File name of the vector geometry, by default "vector/vector.geojson"
        """
        super().__init__(model)
        self._vector: Optional[xr.Dataset] = None
        self.fn = fn
        self.fn_geom = fn_geom

    @property
    def data(self) -> xr.Dataset:
        """Model vector (polygon) data.

        Returns xr.Dataset with a polygon geometry coordinate.
        """
        if self._vector is None:
            self._initialize_vector()
        assert self._vector is not None
        return self._vector

    def _initialize_vector(self, skip_read=False) -> None:
        if self._vector is None:
            self._vector = xr.Dataset()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray, gpd.GeoDataFrame] = None,
        *,
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
        self._initialize_vector()
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")

        # check the type of data
        if isinstance(data, np.ndarray) and "geometry" in self.data:
            index_dim = self.index_dim
            index = self.data[index_dim]
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
                if overwrite_geom:
                    self._logger.warning("Overwriting vector object with data")
                self._vector = data
        # 2. self.vector has a geometry
        else:
            # data has a geometry - check if it is the same as self.vector
            if data.vector.geometry is not None:
                if not np.all(
                    data.vector.geometry.geom_equals_exact(
                        self.geometry, tolerance=0.0001
                    )
                ):
                    raise ValueError("Geometry of data and vector do not match")
            # add data (with check on index)
            for dvar in data.data_vars:
                if dvar in self.data:
                    self._logger.warning(f"Replacing vector variable: {dvar}")
                # check on index coordinate before merging
                dims = data[dvar].dims
                if np.array_equal(
                    data[dims[0]].values, self.data[self.index_dim].values
                ):
                    self.data[dvar] = data[dvar]
                else:
                    raise ValueError(
                        f"Index coordinate of data variable {dvar} "
                        "does not match vector index coordinate"
                    )

    @hydromt_step
    def read(
        self,
        *,
        fn: Optional[str] = None,
        fn_geom: Optional[str] = None,
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
        kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self._root._assert_read_mode()
        self._initialize_vector(skip_read=True)
        fn = fn or self.fn
        fn_geom = fn_geom or self.fn_geom

        if fn is not None:
            # Disable lazy loading of data
            # to avoid issues with reading object dtype data
            if "chunks" not in kwargs:
                kwargs["chunks"] = None
            ds = xr.merge(
                read_nc(
                    fn, root=self._root.path, logger=self._logger, **kwargs
                ).values()
            )
            # check if ds is empty (default fn has a value)
            if len(ds.sizes) == 0:
                fn = None
        if fn_geom is not None and isfile(join(self._root.path, fn_geom)):
            gdf = gpd.read_file(join(self._root.path, fn_geom))
            # geom + netcdf data
            # TODO: What if ds is None?
            if fn is not None:
                ds = GeoDataset.from_gdf(gdf, data_vars=ds)
            # geom only
            else:
                ds = GeoDataset.from_gdf(gdf, keep_cols=True, cols_as_data_vars=True)
        # netcdf only
        elif fn is not None:
            ds = GeoDataset.from_netcdf(ds)
        else:
            self._logger.info("No vector data found, skip reading.")
            return

        self.set(data=ds)

    @hydromt_step
    def write(
        self,
        *,
        fn: Optional[str] = None,
        fn_geom: Optional[str] = None,
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
        ds = self.data
        if len(ds) == 0:
            self._logger.debug("No vector data found, skip writing.")
            return
        self._root._assert_write_mode()
        fn = fn or self.fn
        fn_geom = fn_geom or self.fn_geom

        # If fn is None check if vector contains only 1D data
        if fn is None:
            # Check if 1D data only is present
            snames = ["y_name", "x_name", "index_dim", "geom_name"]
            sdims = [ds.vector.attrs.get(n) for n in snames if n in ds.vector.attrs]
            if "spatial_ref" in ds:
                sdims.append("spatial_ref")
            for name in list(set(ds.vector._all_names) - set(sdims)):
                dims = ds[name].dims
                # check 1D variables with matching index_dim
                if len(dims) > 1 or dims[0] != ds.vector.index_dim:
                    fn = join(
                        dirname(join(self._root.path, fn_geom)),
                        f"{basename(fn_geom).split('.')[0]}.nc",
                    )
                    self._logger.warning(
                        "2D data found in vector," f"will write data to {fn} instead."
                    )
                    break

        # write to netcdf only
        if fn_geom is None:
            os.makedirs(dirname(join(self._root.path, fn)), exist_ok=True)
            # cannot call directly ds.vector.to_netcdf
            # because of possible PermissionError
            if ogr_compliant:
                ds = ds.vector.ogr_compliant()
            else:
                ds = ds.vector.update_geometry(geom_format="wkt", geom_name="ogc_wkt")
            # write_nc requires dict - use dummy key
            write_nc(
                {"vector": ds},
                fn,
                engine="netcdf4",
                root=self._root.path,
                logger=self._logger,
                **kwargs,
            )
        # write to geojson only
        elif fn is None:
            os.makedirs(dirname(join(self._root.path, fn_geom)), exist_ok=True)
            gdf = ds.vector.to_gdf(**kwargs)
            gdf.to_file(join(self._root.path, fn_geom))
        # write data to netcdf and geometry to geojson
        else:
            os.makedirs(dirname(join(self._root.path, fn_geom)), exist_ok=True)
            # write geometry
            gdf = ds.vector.geometry.to_frame("geometry")
            gdf.to_file(join(self._root.path, fn_geom))
            # write_nc requires dict - use dummy key
            write_nc(
                {"vector": ds.drop_vars("geometry")},
                fn,
                root=self._root.path,
                logger=self._logger,
                **kwargs,
            )

    # Other vector properties
    @property
    def geometry(self) -> gpd.GeoSeries:
        """Returns the geometry of the model vector as gpd.GeoSeries."""
        # check if vector is empty
        if len(self.data.sizes) == 0:
            return None
        else:
            return self.data.vector.geometry

    @property
    def index_dim(self) -> str:
        """Returns the index dimension of the vector."""
        return self.data.vector.index_dim

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in the vector."""
        if self.data.vector.crs is not None:
            return self.data.vector.crs
        self._logger.warning("No CRS found in vector data.")
        return None
