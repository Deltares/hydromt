"""The Geodataframe adapter implementation."""
import logging
import warnings
from os.path import join
from pathlib import Path
from typing import NewType, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from .. import io
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
        if gdf.index.size == 0:
            return None, None

        if driver is None:
            _lst = ["csv", "xls", "xlsx", "xy", "vector_table"]
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
        else:
            driver_extensions = {
                "ESRI Shapefile": ".shp",
            }
            ext = driver_extensions.get(driver, driver).lower()
            fn_out = join(data_root, f"{data_name}.{ext}")
            gdf.to_file(fn_out, driver=driver, **kwargs)
            driver = "vector"

        return fn_out, driver

    def get_data(
        self,
        bbox=None,
        geom=None,
        predicate="intersects",
        buffer=0,
        logger=logger,
        variables=None,
        # **kwargs,  # this is not used, for testing only
    ):
        """Return a clipped and unified GeoDataFrame (vector).

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataframe`
        """
        # If variable is string, convert to list
        if variables:
            variables = np.atleast_1d(variables).tolist()

        if "storage_options" in self.driver_kwargs:
            # not sure if storage options can be passed to fiona.open()
            # for now throw NotImplemented Error
            raise NotImplementedError(
                "Remote file storage_options not implemented for GeoDataFrame"
            )
        _ = self.resolve_paths()  # throw nice error if data not found

        kwargs = self.driver_kwargs.copy()
        # parse geom, bbox and buffer arguments
        clip_str = ""
        if geom is None and bbox is not None:
            # convert bbox to geom with crs EPGS:4326 to apply buffer later
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            clip_str = " and clip to bbox (epsg:4326)"
        elif geom is not None:
            clip_str = f" and clip to geom (epsg:{geom.crs.to_epsg():d})"
        if geom is not None:
            # make sure geom is projected > buffer in meters!
            if geom.crs.is_geographic and buffer > 0:
                geom = geom.to_crs(3857)
            geom = geom.buffer(buffer)  # a buffer with zero fixes some topology errors
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            clip_str = f"{clip_str} [{bbox_str}]"
        if kwargs.pop("within", False):  # for backward compatibility
            predicate = "contains"

        # read and clip
        logger.info(f"GeoDataFrame: Read {self.driver} data{clip_str}.")
        if self.driver in ["csv", "xls", "xlsx", "xy", "vector", "vector_table"]:
            # "csv", "xls", "xlsx", "xy" deprecated use vector_table instead.
            # specific driver should be added to open_vector kwargs
            if "driver" not in kwargs and self.driver in ["csv", "xls", "xlsx", "xy"]:
                kwargs.update(driver=self.driver)
            # Check if file-object is required because of additional options
            gdf = io.open_vector(
                self.path, crs=self.crs, geom=geom, predicate=predicate, **kwargs
            )
        else:
            raise ValueError(f"GeoDataFrame: driver {self.driver} unknown.")

        # rename and select columns
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in gdf.columns}
            gdf = gdf.rename(columns=rename)
        if variables is not None:
            if np.any([var not in gdf.columns for var in variables]):
                raise ValueError(f"GeoDataFrame: Not all variables found: {variables}")
            if "geometry" not in variables:  # always keep geometry column
                variables = variables + ["geometry"]
            gdf = gdf.loc[:, variables]

        # nodata and unit conversion for numeric data
        if gdf.index.size == 0:
            logger.warning(f"GeoDataFrame: No data within spatial domain {self.path}.")
        else:
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

            # unit conversion
            unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
            unit_names = [k for k in unit_names if k in gdf.columns]
            if len(unit_names) > 0:
                logger.debug(
                    f"GeoDataFrame: Convert units for {len(unit_names)} columns."
                )
            for name in list(set(unit_names)):  # unique
                m = self.unit_mult.get(name, 1)
                a = self.unit_add.get(name, 0)
                gdf[name] = gdf[name] * m + a

        # set meta data
        gdf.attrs.update(self.meta)

        # set column attributes
        for col in self.attrs:
            if col in gdf.columns:
                gdf[col].attrs.update(**self.attrs[col])
        return gdf
