"""The Geodataframe adapter implementation."""
import logging
import warnings
from os.path import join
from pathlib import Path
from typing import NewType, Optional, Tuple, Union

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
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        crs: Optional[Union[int, str, dict]] = None,
        nodata: Optional[Union[dict, float, int]] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        extent: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
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
        filesystem: str, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            If None (default) the filesystem is inferred from the path.
            See :py:func:`fsspec.registry.known_implementations` for all options.
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
        extent: Extent(typed dict), Optional
            Dictionary describing the spatial and time range the dataset covers.
            should be of the form:
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "time_range": [start_datetime, end_datetime],
            }
            bbox coordinates should be in the same CRS as the data, and
            time_range should be inclusive on both sides.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        storage_options: dict, optional
            Additional key-word arguments passed to the fsspec FileSystem object.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional.
        provider: str, optional
            A name to identifiy the specific provider of the dataset requested.
            if None is provided, the last added source will be used.
        version: str, optional
            A name to identifiy the specific version of the dataset requested.
            if None is provided, the last added source will be used.
        """
        driver_kwargs = driver_kwargs or {}
        extent = extent or {}
        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "GeoDataFrameAdapter driver is deprecated and will be removed "
                "in a future version. Please use 'driver_kwargs' instead.",
                DeprecationWarning,
                stacklevel=2,
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
            storage_options=storage_options,
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
        )
        self.crs = crs
        self.extent = extent

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
        gdf = self._read_data(fns, bbox, geom, buffer, predicate, logger=logger)
        self.mark_as_used()  # mark used
        # rename variables and parse crs & nodata
        gdf = self._rename_vars(gdf)
        gdf = self._set_crs(gdf, logger=logger)
        gdf = self._set_nodata(gdf)
        # slice
        gdf = GeoDataFrameAdapter._slice_data(
            gdf, variables, geom, bbox, buffer, predicate, logger=logger
        )
        # uniformize
        gdf = self._apply_unit_conversions(gdf, logger=logger)
        gdf = self._set_metadata(gdf)
        return gdf

    def _read_data(self, fns, bbox, geom, buffer, predicate, logger=logger):
        if len(fns) > 1:
            raise ValueError(
                f"GeoDataFrame: Reading multiple {self.driver} files is not supported."
            )
        kwargs = self.driver_kwargs.copy()
        path = fns[0]
        logger.info(f"Reading {self.name} {self.driver} data from {self.path}")
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
                    "vector_table instead.",
                    stacklevel=2,
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

    def _set_crs(self, gdf, logger=logger):
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
        gdf,
        variables=None,
        geom=None,
        bbox=None,
        buffer=0,
        predicate="intersects",
        logger=logger,
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
            # NOTE if we read with vector driver this is already done ..
            geom = gis_utils.parse_geom_bbox_buffer(geom, bbox, buffer)
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            epsg = geom.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            idxs = gis_utils.filter_gdf(gdf, geom=geom, predicate=predicate)
            if idxs.size == 0:
                raise IndexError("No data within spatial domain.")
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

    def _apply_unit_conversions(self, gdf, logger=logger):
        # unit conversion
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in gdf.columns]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} columns.")
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

    def get_bbox(self, detect=True) -> Tuple[Tuple[float, float, float, float], int]:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.GeoDataframeAdapter.detect_bbox` will be used to detect it.

        Parameters
        ----------
        detect: bool, Optional
            whether to detect the bounding box if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        bbox = self.extent.get("bbox", None)
        crs = self.crs
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def detect_bbox(
        self,
        gdf=None,
    ) -> Tuple[Tuple[float, float, float, float], int]:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched acodring to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDataframeAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.GeoDataframeAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        if gdf is None:
            gdf = self.get_data()

        crs = gdf.geometry.crs.to_epsg()
        bounds = gdf.geometry.total_bounds
        return bounds, crs
