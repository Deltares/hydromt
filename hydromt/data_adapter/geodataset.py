"""Implementation for the geodataset DataAdapter."""

from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pyproj
import xarray as xr

from hydromt._typing import (
    Bbox,
    GeoDatasetSource,
    Geom,
    GeomBuffer,
    NoDataStrategy,
    Predicate,
    StrPath,
    TimeRange,
    Variables,
)
from hydromt._typing.type_def import Number
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.data_adapter.utils import (
    _single_var_as_array,
    has_no_data,
    shift_dataset_time,
)
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.gis.utils import parse_geom_bbox_buffer
from hydromt.io import netcdf_writer, zarr_writer

if TYPE_CHECKING:
    from hydromt.data_source.data_source import SourceMetadata
logger = getLogger(__name__)

__all__ = ["GeoDatasetAdapter", "GeoDatasetSource"]


class GeoDatasetAdapter(DataAdapterBase):
    """DatasetAdapter for GeoDatasets."""

    def to_file(
        self,
        data_root: StrPath,
        data_name: str,
        bbox: Optional[Bbox] = None,
        time_tuple: Optional[TimeRange] = None,
        variables: Optional[List[str]] = None,
        driver: Optional[str] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
        **kwargs,
    ) -> Optional[Tuple[StrPath, Optional[str], Dict[str, Any]]]:
        """Save a data slice to file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        driver : str, optional
            Driver to write file, e.g.: 'netcdf', 'zarr', by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        **kwargs
            Additional keyword arguments that are passed to the `to_zarr`
            function.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see
            :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        obj = self.get_data(
            bbox=bbox,
            time_tuple=time_tuple,
            variables=variables,
            handle_nodata=handle_nodata,
            logger=logger,
            single_var_as_array=variables is None,
        )

        # we'll handle the correct strategy in the except clause
        if obj is None:
            return None

        read_kwargs = {}

        # much better for mem/storage/processing if dtypes are set correctly
        for name, coord in obj.coords.items():
            if coord.values.dtype != object:
                continue

            # not sure if coordinates values of different dtypes
            # are possible, but let's just hope users aren't
            # that mean for now.
            if isinstance(coord.values[0], str):
                obj[name] = obj[name].astype(str)

        if driver is None or driver == "netcdf":
            # always write netcdf
            fn_out = netcdf_writer(
                obj=obj,
                data_root=data_root,
                data_name=data_name,
                variables=variables,
            )
        elif driver == "zarr":
            fn_out = zarr_writer(
                obj=obj, data_root=data_root, data_name=data_name, **kwargs
            )
        else:
            raise ValueError(f"GeoDataset: Driver {driver} unknown.")

        return fn_out, driver, read_kwargs

    def transform(
        self,
        ds: xr.Dataset,
        metadata: "SourceMetadata",
        *,
        bbox: Optional[Bbox] = None,
        geom: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        """Return a clipped, sliced and unified RasterDataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """
        ds = GeoDatasetAdapter._rename_vars(ds, self.rename)
        ds = GeoDatasetAdapter._validate_spatial_coords(ds)
        ds = GeoDatasetAdapter._set_crs(ds, crs=metadata.crs, logger=logger)
        ds = GeoDatasetAdapter._set_nodata(ds, metadata)
        ds = shift_dataset_time(dt=self.unit_add.get("time", 0), ds=ds, logger=logger)
        ds = GeoDatasetAdapter._apply_unit_conversion(
            ds, unit_mult=self.unit_mult, unit_add=self.unit_add, logger=logger
        )
        ds = GeoDatasetAdapter._set_metadata(ds, metadata=metadata)
        maybe_ds = GeoDatasetAdapter._slice_data(
            ds,
            variables=variables,
            geom=geom,
            bbox=bbox,
            buffer=buffer,
            predicate=predicate,
            time_range=time_range,
            logger=logger,
        )

        return _single_var_as_array(maybe_ds, single_var_as_array, variables)

    @staticmethod
    def _validate_spatial_coords(ds: xr.Dataset) -> xr.Dataset:
        if GEO_MAP_COORD in ds.data_vars:
            ds = ds.set_coords(GEO_MAP_COORD)
        try:
            ds.vector.set_spatial_dims()
            idim = ds.vector.index_dim
            if idim not in ds:  # set coordinates for index dimension if missing
                ds[idim] = xr.IndexVariable(idim, np.arange(ds.dims[idim]))
            coords = [ds.vector.x_name, ds.vector.y_name, idim]
            coords = [item for item in coords if item is not None]
            ds = ds.set_coords(coords)
        except ValueError:
            raise ValueError(
                f"GeoDataset: No spatial geometry dimension found in data {ds.name}"
            )
        return ds

    @staticmethod
    def _rename_vars(ds: xr.Dataset, rename: Dict[str, str]) -> xr.Dataset:
        rm = {k: v for k, v in rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    @staticmethod
    def _set_crs(
        ds: xr.Dataset, crs: Union[str, int, None] = None, logger: Logger = logger
    ) -> xr.Dataset:
        # set crs
        if ds.vector.crs is None and crs is not None:
            ds.vector.set_crs(crs)
        elif ds.vector.crs is None:
            raise ValueError(
                f"GeoDataset {ds.name}: CRS not defined in data catalog or data."
            )
        elif crs is not None and ds.vector.crs != pyproj.CRS.from_user_input(crs):
            logger.warning(
                f"GeoDataset {ds.name}: CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds: xr.Dataset,
        variables: Optional[Variables] = None,
        geom: Optional[Geom] = None,
        bbox: Optional[Bbox] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        time_range: Optional[TimeRange] = None,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        """Slice the dataset in space and time.

        Arguments
        ---------
        ds : xarray.Dataset or xarray.DataArray
            The GeoDataset to slice.
        variables : str or list of str, optional.
            Names of variables to return.
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        buffer : float, optional
            Buffer distance [m] applied to the geometry or bbox. By default 0 m.
        predicate : str, optional
            Predicate used to filter the GeoDataFrame, see
            :py:func:`hydromt.gis.utils.filter_gdf` for details.
        handle_nodata : NoDataStrategy, optional
            How to handle no data values. By default NoDataStrategy.RAISE.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        ds : xarray.Dataset
            The sliced GeoDataset.
        """
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                # dummy name, required to create dataset
                # renamed to variable in _single_var_as_array
                ds.name = "data"
            ds = ds.to_dataset()
        elif variables is not None:
            variables = cast(List, np.atleast_1d(variables).tolist())
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise ValueError(f"GeoDataset: variables not found {mvars}")
                ds = ds[variables]
        maybe_ds: Optional[xr.Dataset] = ds
        if time_range is not None:
            maybe_ds = GeoDatasetAdapter._slice_temporal_dimension(
                ds, time_range, logger=logger
            )
        if geom is not None or bbox is not None:
            maybe_ds = GeoDatasetAdapter._slice_spatial_dimension(
                maybe_ds,
                geom=geom,
                bbox=bbox,
                predicate=predicate,
                buffer=buffer,
                logger=logger,
            )
        if has_no_data(ds):
            return None
        else:
            return ds

    @staticmethod
    def _slice_spatial_dimension(
        ds: Optional[xr.Dataset],
        geom: Optional[Geom],
        bbox: Optional[Bbox],
        predicate: Predicate,
        buffer: GeomBuffer = 0,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        else:
            geom = parse_geom_bbox_buffer(geom, bbox, buffer)
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            epsg = geom.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            ds = ds.vector.clip_geom(geom, predicate=predicate)
            if has_no_data(ds):
                return None
            else:
                return ds

    @staticmethod
    def _slice_temporal_dimension(
        ds: Optional[xr.Dataset],
        time_tuple: Optional[TimeRange],
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        else:
            if (
                "time" in ds.dims
                and ds["time"].size > 1
                and np.issubdtype(ds["time"].dtype, np.datetime64)
            ):
                logger.debug(f"Slicing time dim {time_tuple}")
                ds = ds.sel(time=slice(*time_tuple))
            if has_no_data(ds):
                return None
            else:
                return ds

    @staticmethod
    def _set_metadata(ds: xr.Dataset, metadata: "SourceMetadata") -> xr.Dataset:
        if metadata.attrs:
            if isinstance(ds, xr.DataArray):
                name = cast(str, ds.name)
                ds.attrs.update(metadata.attrs[name])
            else:
                for k in metadata.attrs:
                    ds[k].attrs.update(metadata.attrs[k])

        ds.attrs.update(metadata)
        return ds

    @staticmethod
    def _set_nodata(ds: xr.Dataset, metadata: "SourceMetadata") -> xr.Dataset:
        if metadata.nodata is not None:
            if not isinstance(metadata.nodata, dict):
                nodata = {k: metadata.nodata for k in ds.data_vars.keys()}
            else:
                nodata = metadata.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].vector.nodata is None:
                    ds[k].vector.set_nodata(mv)
        return ds

    @staticmethod
    def _apply_unit_conversion(
        ds: xr.Dataset,
        unit_mult: Dict[str, Number],
        unit_add: Dict[str, Number],
        logger: Logger = logger,
    ) -> xr.Dataset:
        unit_names = list(unit_mult.keys()) + list(unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = unit_mult.get(name, 1)
            a = unit_add.get(name, 0)
            da = ds[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.vector.nodata is None or np.isnan(da.vector.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.vector.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds[name] = xr.where(data_bool, da * m + a, nodata)
            ds[name].attrs.update(attrs)  # set original attributes
        return ds
