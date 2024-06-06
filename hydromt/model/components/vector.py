# -*- coding: utf-8 -*-
"""HydroMT VectorComponent class definition."""

import os
from os.path import basename, dirname, isfile, join
from typing import TYPE_CHECKING, Optional, Union, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

from hydromt.gis.vector import GeoDataset
from hydromt.io.readers import read_nc
from hydromt.io.writers import write_nc
from hydromt.model import hydromt_step
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent

if TYPE_CHECKING:
    from hydromt.model.model import Model

__all__ = ["VectorComponent"]


class VectorComponent(SpatialModelComponent):
    """ModelComponent class for vector components.

    This class is used to manage vector data in a model (e.g. for polygons of a semi
    distributed model). The vector component data stored in the ``data`` property of
    this class if of the hydromt.gis.vector.GeoDataset type which is an extension of
    xarray.Dataset with a geometry coordinate.
    """

    def __init__(
        self,
        model: "Model",
        *,
        region_component: Optional[str] = None,
        region_filename: str = "vector/vector_region.geojson",
    ) -> None:
        """Initialize a vector component.

        Parameters
        ----------
        model : Model
            Parent model
        region_component : str, optional
            The name of the region component to use as reference for this component's region.
            If None, the region will be set to the bounds of the geometry of this vector component.
        region_filename : str
            The path to use for writing the region data to a file. By default "vector/vector_region.geojson".
        """
        super().__init__(
            model, region_component=region_component, region_filename=region_filename
        )
        self._data: Optional[xr.Dataset] = None

    @property
    def data(self) -> xr.Dataset:
        """Model vector (polygon) data.

        Returns xr.Dataset with a polygon geometry coordinate.
        """
        if self._data is None:
            self._initialize()
        assert self._data is not None
        return self._data

    @property
    def _region_data(self) -> Optional[gpd.GeoDataFrame]:
        if len(self.data.vector) == 0:
            return None
        gdf = self.geometry.to_frame("geometry")
        return gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)

    def _initialize(self, skip_read=False) -> None:
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
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
        self._initialize()
        assert self._data is not None
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")

        # check the type of data
        if isinstance(data, np.ndarray) and "geometry" in self._data:
            index_dim = self.index_dim
            index = self._data[index_dim]
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
                    self.logger.warning("Overwriting vector object with data")
                self._data = data
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
                if dvar in self._data:
                    self.logger.warning(f"Replacing vector variable: {dvar}")
                # check on index coordinate before merging
                dims = data[dvar].dims
                if np.array_equal(
                    data[dims[0]].values, self._data[self.index_dim].values
                ):
                    self._data[dvar] = data[dvar]
                else:
                    raise ValueError(
                        f"Index coordinate of data variable {dvar} "
                        "does not match vector index coordinate"
                    )

    @hydromt_step
    def read(
        self,
        *,
        filename: Optional[str] = "vector/vector.nc",
        geometry_filename: Optional[str] = "vector/vector.geojson",
        **kwargs,
    ) -> None:
        """Read model vector from combined netcdf and geojson file.

        Files are read at <root>/<filename> and geojson file at <root>/<geometry_filename>.

        Three options are possible:
            * The netcdf file contains the attribute data and the geojson file the
                geometry vector data.
            * The netcdf file contains both the attribute and the geometry data.
                (geometry_filename is ignored)
            * The geojson file contains both the attribute and the geometry data.
                (filename is ignored)

        Key-word arguments are passed to :py:meth:`~hydromt.model.Model.read_nc`

        Parameters
        ----------
        filename : str, optional
            netcdf filename relative to model root,
            by default 'vector/vector.nc'
        geometry_filename : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
        kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)

        if filename is None and geometry_filename is None:
            raise ValueError(
                "Both filename and geometry_filename are None, no source file given."
            )

        if filename is not None:
            # Disable lazy loading of data
            # to avoid issues with reading object dtype data
            if "chunks" not in kwargs:
                kwargs["chunks"] = None
            ds = xr.merge(
                read_nc(
                    filename, root=self.root.path, logger=self.logger, **kwargs
                ).values()
            )
            # check if ds is empty (default filename has a value)
            if len(ds.sizes) == 0:
                filename = None
        if geometry_filename is not None and isfile(
            join(self.root.path, geometry_filename)
        ):
            gdf = gpd.read_file(join(self.root.path, geometry_filename))
            # geom + netcdf data
            if filename is not None:
                ds = GeoDataset.from_gdf(gdf, data_vars=ds)
            # geom only
            else:
                ds = GeoDataset.from_gdf(gdf, keep_cols=True, cols_as_data_vars=True)
        # netcdf only
        elif filename is not None:
            ds = GeoDataset.from_netcdf(ds)
        else:
            self.logger.info("No vector data found, skip reading.")
            return

        self.set(data=ds)

    @hydromt_step
    def write(
        self,
        *,
        filename: Optional[str] = "vector/vector.nc",
        geometry_filename: Optional[str] = "vector/vector.geojson",
        ogr_compliant: bool = False,
        **kwargs,
    ) -> None:
        """Write model vector to combined netcdf and geojson files.

        Files are written at <root>/<filename> and at <root>/<geometry_filename> respectively.

        Three options are possible:

            * The netcdf file contains the attribute data and the geojson file the
                geometry vector data. Key-word arguments are passed to
                :py:meth:`~hydromt.model.Model.write_nc`

            * The netcdf file contains both the attribute and the geometry data
                (geometry_filename set to None). Key-word arguments are passed to
                :py:meth:`~hydromt.model.Model.write_nc`

            * The geojson file contains both the attribute and the geometry data
                (filename set to None). This option is possible only if all data variables
                are 1D. Else the data will be written to netcdf. Key-word arguments are
                passed to :py:meth:`~hydromt.vector.GeoDataset.to_gdf`

        Parameters
        ----------
        filename : str, optional
            netcdf filename relative to model root,
            by default 'vector/vector.nc'
        geometry_filename : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
        ogr_compliant : bool
            If filename only, write the netCDF4 file in an ogr compliant format
            This makes it readable as a vector file in e.g. QGIS
            see :py:meth:`~hydromt.vector.GeoBase.ogr_compliant` for more details.
        **kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        ds = self.data
        if len(ds) == 0:
            self.logger.debug("No vector data found, skip writing.")
            return
        self.root._assert_write_mode()

        if filename is None and geometry_filename is None:
            raise ValueError(
                "Both filename and geometry_filename are None, no destination file given. Please provide either filename or geometry_filename."
            )

        # If filename is None check if vector contains only 1D data
        if filename is None:
            assert geometry_filename is not None
            # Check if 1D data only is present
            snames = ["y_name", "x_name", "index_dim", "geom_name"]
            sdims = [ds.vector.attrs.get(n) for n in snames if n in ds.vector.attrs]
            if "spatial_ref" in ds:
                sdims.append("spatial_ref")
            for name in list(set(ds.vector._all_names) - set(sdims)):
                dims = ds[name].dims
                # check 1D variables with matching index_dim
                if len(dims) > 1 or dims[0] != ds.vector.index_dim:
                    filename = join(
                        dirname(join(self.root.path, geometry_filename)),
                        f"{basename(geometry_filename).split('.')[0]}.nc",
                    )
                    self.logger.warning(
                        "2D data found in vector,"
                        f"will write data to {filename} instead."
                    )
                    break

        # write to netcdf only
        if geometry_filename is None:
            assert filename is not None
            os.makedirs(dirname(join(self.root.path, filename)), exist_ok=True)
            # cannot call directly ds.vector.to_netcdf
            # because of possible PermissionError
            if ogr_compliant:
                ds = ds.vector.ogr_compliant()
            else:
                ds = ds.vector.update_geometry(geom_format="wkt", geom_name="ogc_wkt")
            # write_nc requires dict - use dummy key
            write_nc(
                {"vector": ds},
                filename,
                engine="netcdf4",
                root=self.root.path,
                logger=self.logger,
                **kwargs,
            )
        # write to geojson only
        elif filename is None:
            os.makedirs(dirname(join(self.root.path, geometry_filename)), exist_ok=True)
            gdf = ds.vector.to_gdf(**kwargs)
            gdf.to_file(join(self.root.path, geometry_filename))
        # write data to netcdf and geometry to geojson
        else:
            os.makedirs(dirname(join(self.root.path, geometry_filename)), exist_ok=True)
            # write geometry
            gdf = ds.vector.geometry.to_frame("geometry")
            gdf.to_file(join(self.root.path, geometry_filename))
            # write_nc requires dict - use dummy key
            write_nc(
                {"vector": ds.drop_vars("geometry")},
                filename,
                root=self.root.path,
                logger=self.logger,
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
        self.logger.warning("No CRS found in vector data.")
        return None

    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_vector = cast(VectorComponent, other)

        geods = self.data
        other_geods = other_vector.data
        gdf = self.geometry.to_frame("geometry")
        gdf_other = other_vector.geometry.to_frame("geometry")
        try:
            assert_geodataframe_equal(
                gdf, gdf_other, check_like=True, check_less_precise=True
            )
        except AssertionError as e:
            errors["geometry"] = str(e)
        drop_vars = (
            [geods.vector.x_name, geods.vector.y_name]
            if geods.vector.geom_format == "xy"
            else [geods.vector.geom_name]
        )
        ds = geods.drop_vars(drop_vars)
        ds_other = other_geods.drop_vars(drop_vars)
        try:
            xr.testing.assert_allclose(ds, ds_other)
        except AssertionError as e:
            errors["data"] = str(e)

        return len(errors) == 0, errors
