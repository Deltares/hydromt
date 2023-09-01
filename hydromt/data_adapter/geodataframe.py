"""The Geodataframe adapter implementation."""
import logging
import warnings
from os.path import join
from pathlib import Path
from typing import NewType, Union

import numpy as np
import pyproj

from .. import gis_utils, io
from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)

__all__ = ["GeoDataFrameAdapter", "GeoDataframeSource"]

GeoDataframeSource = NewType("GeoDataframeSource", Union[str, Path])


class GeoDataFrameAdapter(DataAdapter):

    """The Geodataframe adapter implementation."""

    _DEFAULT_DRIVER = "vector"
    _DRIVERS = {
        "xy": "xy",
        "csv": "csv",
        "parquet": "parquet",
        "xls": "xls",
        "xlsx": "xlsx",
    }

    def __init__(
        self,
        path: str,
        driver: str = None,
        filesystem: str = "local",
        crs: Union[int, str, dict] = None,
        nodata: Union[dict, float, int] = None,
        rename: dict = {},
        unit_mult: dict = {},
        unit_add: dict = {},
        meta: dict = {},
        attrs: dict = {},
        driver_kwargs: dict = {},
        name: str = "",  # optional for now
        catalog_name: str = "",  # optional for now
        provider=None,
        version=None,
        **kwargs,
    ):
        """Initiate data adapter for geospatial vector data.

        This object contains all properties required to read supported files into
        a single unified :py:func:`geopandas.GeoDataFrame`.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable} placeholders as well as path
            search pattern using a '*' wildcard.
        driver: {'vector', 'vector_table'}, optional
            Driver to read files with, for 'vector' :py:func:`~geopandas.read_file`,
            for {'vector_table'} :py:func:`hydromt.io.open_vector_from_table`
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
        filesystem: {'local', 'gcs', 's3'}, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            By default, local.
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str);
            proj (str or dict) or wkt (str). Only used if the data has no native CRS.
        nodata: dictionary, float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Nodata values can be differentiated between variables using a dictionary.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native
            data unit to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license',
            'paper_ref', 'paper_doi', 'category'}
        placeholders: dict, optional
            Placeholders to expand yaml entry to multiple entries (name and path)
            based on placeholder values
        attrs: dict, optional
            Additional attributes relating to data variables. For instance unit
            or long name of the variable.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.
        """
        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "GeoDataFrameAdapter driver is deprecated and will be removed "
                "in a future version. Please use 'driver_kwargs' instead.",
                DeprecationWarning,
            )
            driver_kwargs.update(kwargs)
        super().__init__(
            path=path,
            driver=driver,
            filesystem=filesystem,
            nodata=nodata,
            rename=rename,
            unit_mult=unit_mult,
            unit_add=unit_add,
            meta=meta,
            attrs=attrs,
            driver_kwargs=driver_kwargs,
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
        )
        self.crs = crs

    def to_file(
        self,
        data_root,
        data_name,
        bbox=None,
        driver=None,
        variables=None,
        logger=logger,
        **kwargs,
    ):
        """Save a data slice to file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        driver : str, optional
            Driver to write file, e.g.: 'GPKG', 'ESRI Shapefile' or any fiona data type,
            by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        logger : logger object, optional
            The logger object used for logging messages. If not provided, the default
            logger will be used.
        **kwargs
            Additional keyword arguments that are passed to the geopandas driver.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see
            :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataframe`
        """
        kwargs.pop("time_tuple", None)
        gdf = self.get_data(bbox=bbox, variables=variables, logger=logger)

        read_kwargs = {}
        if driver is None:
            _lst = ["csv", "parquet", "xls", "xlsx", "xy", "vector_table"]
            driver = "csv" if self.driver in _lst else "GPKG"
        # always write netcdf
        if driver == "csv":
            fn_out = join(data_root, f"{data_name}.csv")
            if not np.all(gdf.geometry.type == "Point"):
                raise ValueError(
                    f"{data_name} contains other geometries than 'Point' "
                    "which cannot be written to csv."
                )
            gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
            gdf.drop(columns="geometry").to_csv(fn_out, **kwargs)
            read_kwargs["index_col"] = 0
        elif driver == "parquet":
            fn_out = join(data_root, f"{data_name}.parquet")
            if not np.all(gdf.geometry.type == "Point"):
                raise ValueError(
                    f"{data_name} contains other geometries than 'Point' "
                    "which cannot be written to parquet."
                )
            gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
            gdf.drop(columns="geometry").to_parquet(fn_out, **kwargs)
        else:
            driver_extensions = {
                "ESRI Shapefile": ".shp",
            }
            ext = driver_extensions.get(driver, driver).lower()
            fn_out = join(data_root, f"{data_name}.{ext}")
            gdf.to_file(fn_out, driver=driver, **kwargs)
            driver = "vector"

        return fn_out, driver, read_kwargs

    def get_data(
        self,
        bbox=None,
        geom=None,
        buffer=0,
        predicate="intersects",
        logger=logger,
        variables=None,
    ):
        """Return a clipped and unified GeoDataFrame (vector).

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataframe`
        """
        # load
        fns = self._resolve_paths(variables)
        gdf = self._read_data(fns, bbox, geom, buffer, predicate)
        # rename variables and parse crs & nodata
        gdf = self._rename_vars(gdf)
        gdf = self._set_crs(gdf)
        gdf = self._set_nodata(gdf)
        # slice
        gdf = self._slice_data(gdf, variables, geom)
        # uniformize
        gdf = self._apply_unit_conversions(gdf)
        gdf = self._set_metadata(gdf)
        return gdf

    def _resolve_paths(self, variables):
        # storage options for fsspec (TODO: not implemented yet)
        if "storage_options" in self.driver_kwargs:
            # not sure if storage options can be passed to fiona.open()
            # for now throw NotImplemented Error
            raise NotImplementedError(
                "Remote file storage_options not implemented for GeoDataFrame"
            )

        # resolve paths
        fns = super()._resolve_paths(variables=variables)

        return fns

    def _read_data(self, fns, bbox, geom, buffer, predicate):
        if len(fns) > 1:
            raise ValueError(
                f"GeoDataFrame: Reading multiple {self.driver} files is not supported."
            )
        kwargs = self.driver_kwargs.copy()
        path = fns[0]
        logger.debug(f"GeoDataFrame: Reading {self.driver} data from {self.path}")
        if self.driver in [
            "csv",
            "parquet",
            "xls",
            "xlsx",
            "xy",
            "vector",
            "vector_table",
        ]:
            # "csv", "xls", "xlsx", "xy" deprecated use vector_table instead.
            # specific driver should be added to open_vector kwargs
            if "driver" not in kwargs and self.driver in ["csv", "xls", "xlsx", "xy"]:
                warnings.warn(
                    "using the driver setting is deprecated. Please use"
                    "vector_table instead."
                )
                kwargs.update(driver=self.driver)
            # parse bbox and geom to (buffere) geom
            if bbox is not None or geom is not None:
                geom = gis_utils.parse_geom_bbox_buffer(geom, bbox, buffer)
            # Check if file-object is required because of additional options
            gdf = io.open_vector(
                path, crs=self.crs, geom=geom, predicate=predicate, **kwargs
            )
        else:
            raise ValueError(f"GeoDataFrame: driver {self.driver} unknown.")

        return gdf

    def _rename_vars(self, gdf):
        # rename and select columns
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in gdf.columns}
            gdf = gdf.rename(columns=rename)
        return gdf

    def _set_crs(self, gdf):
        if self.crs is not None and gdf.crs is None:
            gdf.set_crs(self.crs, inplace=True)
        elif gdf.crs is None:
            raise ValueError(
                f"GeoDataFrame {self.name}: CRS not defined in data catalog or data."
            )
        elif self.crs is not None and gdf.crs != pyproj.CRS.from_user_input(self.crs):
            logger.warning(
                f"GeoDataFrame {self.name}: CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return gdf

    @staticmethod
    def _slice_data(
        gdf, variables=None, geom=None, bbox=None, buffer=0, predicate="intersects"
    ):
        """Return a clipped GeoDataFrame (vector).

        Arguments
        ---------
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return.
        geom : geopandas.GeoDataFrame/Series, optional
            A geometry defining the area of interest.
        bbox : array-like of floats, optional
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        predicate : str, optional
            Predicate used to filter the GeoDataFrame, see
            :py:func:`hydromt.gis_utils.filter_gdf` for details.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if variables is not None:
            variables = np.atleast_1d(variables).tolist()
            if np.any([var not in gdf.columns for var in variables]):
                raise ValueError(f"GeoDataFrame: Not all variables found: {variables}")
            if "geometry" not in variables:  # always keep geometry column
                variables = variables + ["geometry"]
            gdf = gdf.loc[:, variables]

        if geom is not None or bbox is not None:
            geom = gis_utils.parse_geom_bbox_buffer(geom, bbox, buffer)
            idxs = gis_utils.filter_gdf(gdf, geom=geom, predicate=predicate)
            if idxs.size == 0:
                raise IndexError("GeoDataFrame: No data within spatial domain.")
            gdf = gdf.iloc[idxs]
        return gdf

    def _set_nodata(self, gdf):
        # parse nodata values
        cols = gdf.select_dtypes([np.number]).columns
        if self.nodata is not None and len(cols) > 0:
            if not isinstance(self.nodata, dict):
                nodata = {c: self.nodata for c in cols}
            else:
                nodata = self.nodata
            for c in cols:
                mv = nodata.get(c, None)
                if mv is not None:
                    is_nodata = np.isin(gdf[c], np.atleast_1d(mv))
                    gdf[c] = np.where(is_nodata, np.nan, gdf[c])
        return gdf

    def _apply_unit_conversions(self, gdf):
        # unit conversion
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in gdf.columns]
        if len(unit_names) > 0:
            logger.debug(f"GeoDataFrame: Convert units for {len(unit_names)} columns.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            gdf[name] = gdf[name] * m + a
        return gdf

    def _set_metadata(self, gdf):
        # set meta data
        gdf.attrs.update(self.meta)

        # set column attributes
        for col in self.attrs:
            if col in gdf.columns:
                gdf[col].attrs.update(**self.attrs[col])

        return gdf
