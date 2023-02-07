""""""
from __future__ import annotations
from typing import Union
import numpy as np
import xarray as xr
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry.base import BaseGeometry
import shapely
import pyproj
import logging

from hydromt import gis_utils, raster

from osgeo import __version__ as GDAL_VERSION

logger = logging.getLogger(__name__)


class GeoBase(raster.XGeoBase):
    def __init__(self, xarray_obj):
        super(GeoBase, self).__init__(xarray_obj)
        self._geometry = None

    @property
    def _all_names(self):
        names = [n for n in self._obj.coords]
        if isinstance(self._obj, xr.Dataset):
            names = names + [n for n in self._obj.data_vars]
        return names

    def _get_geom_names_types(self, geom_name: str = None) -> tuple[list, list]:
        names, types = [], []
        dvars = self._all_names if geom_name is None else [geom_name]
        for name in dvars:
            if self._obj[name].ndim == 1 and isinstance(
                self._obj[name][0].values.item(), BaseGeometry
            ):
                names.append(name)
                types.append("geom")
            elif self._obj[name].ndim == 1 and isinstance(
                self._obj[name][0].values.item(), str
            ):
                try:
                    shapely.wkt.loads(self._obj[name][0].values.item())
                    names.append(name)
                    types.append("wkt")
                except Exception:
                    pass
        return names, types

    def _discover_xy(self, x_name=None, y_name=None, index_dim=None):
        """Set the spatial and index dimensions of the object.
        Arguments
        ----------
        x_name, y_name, index_dim: str, optional
            The name of the x, y and index dimensions.
        """
        # infer x dim
        if x_name is None:
            for name in raster.XDIMS:
                if name in self._obj:
                    dim0 = (
                        index_dim if index_dim is not None else self._obj[name].dims[0]
                    )
                    if self._obj[name].ndim == 1 and self._obj[name].dims[0] == dim0:
                        x_name = name
                        index_dim = dim0
                        break
        # infer y dim
        if y_name is None and x_name is not None:
            for name in raster.YDIMS:
                if name in self._obj:
                    if self._obj[name].dims[0] == index_dim:
                        y_name = name
                        break
        if (
            x_name is not None
            and y_name is not None
            and self._obj[x_name].ndim == 1
            and self._obj[x_name].dims == self._obj[y_name].dims
        ):
            self.set_attrs(x_name=x_name)
            self.set_attrs(y_name=y_name)
            self.set_attrs(geom_format="xy")
            self.set_attrs(index_dim=index_dim)
        else:
            self.attrs.pop("x_name", None)
            self.attrs.pop("y_name", None)
            self.attrs.pop("geom_format", None)
            self.attrs.pop("index_dim", None)

    def _discover_geom(self, geom_name=None, index_dim=None):
        # check /infer geom dim
        names, types = self._get_geom_names_types(geom_name=geom_name)
        if geom_name is None:
            if index_dim is not None:
                names = [name for name in names if self._obj[name].dims[0] == index_dim]
            # priority for geometry named variable/coord
            if "geometry" in names:
                geom_name = "geometry"
            # otherwise take first of type geom
            elif len(names) >= 1:
                idx = 0
                if "geom" in types:
                    idx = types.index("geom")
                geom_name = names[idx]
        elif geom_name not in names:
            raise ValueError(f"{geom_name} variable not recognized as geometry.")

        if geom_name is not None:
            self.set_attrs(geom_name=geom_name)
            self.set_attrs(geom_format=types[names.index(geom_name)])
            self.set_attrs(index_dim=self._obj[geom_name].dims[0])
        else:
            self.attrs.pop("geom_name", None)
            self.attrs.pop("geom_format", None)
            self.attrs.pop("index_dim", None)

    def set_spatial_dims(
        self,
        geom_name=None,
        x_name=None,
        y_name=None,
        index_dim=None,
        geom_format=None,
    ):
        if geom_format != "xy":
            self._discover_geom(geom_name=geom_name, index_dim=index_dim)
            self.attrs.pop("x_name", None)
            self.attrs.pop("y_name", None)
        if "geom_name" not in self.attrs:
            self._discover_xy(x_name=x_name, y_name=y_name, index_dim=index_dim)
            self.attrs.pop("geom_name", None)
        elif "geom_name" not in self.attrs:
            raise ValueError("No geometry data found.")

    @property
    def geom_format(self):
        if self.get_attrs("geom_format") not in self._obj.dims:
            self.set_spatial_dims()
        if "geom_format" in self.attrs:
            return self.attrs["geom_format"]

    @property
    def geom_name(self):
        if self.get_attrs("geom_name") not in self._obj.dims:
            self.set_spatial_dims()
        if "geom_name" in self.attrs:
            return self.attrs["geom_name"]

    @property
    def geom_type(self):
        """Return geometry type"""
        geom_types = self.geometry.type.values
        if len(set(geom_types)) > 1:
            # TODO: can we safely assume there is multi geom?
            i = ["MULTI" in g for g in geom_types].index(True)
            geom_type = geom_types[i]
        else:
            geom_type = geom_types[0]
        return geom_type

    @property
    def x_name(self):
        if self.get_attrs("x_name") not in self._obj.dims:
            self.set_spatial_dims()
        if "x_name" in self.attrs:
            return self.attrs["x_name"]

    @property
    def y_name(self):
        if self.get_attrs("y_name") not in self._obj.dims:
            self.set_spatial_dims()
        if "y_name" in self.attrs:
            return self.attrs["y_name"]

    @property
    def index_dim(self):
        if self.get_attrs("index_dim") not in self._obj.dims:
            self.set_spatial_dims()
        if "index_dim" in self.attrs:
            return self.attrs["index_dim"]

    @property
    def time_dim(self):
        """Time dimension name."""
        dim = self.get_attrs("time_dim")
        if dim not in self._obj.dims or np.dtype(self._obj[dim]).type != np.datetime64:
            self.set_attrs(time_dim=None)
            tdims = []
            for dim in self._obj.dims:
                if np.dtype(self._obj[dim]).type == np.datetime64:
                    tdims.append(dim)
            if len(tdims) == 1:
                self.set_attrs(time_dim=tdims[0])
        return self.get_attrs("time_dim")

    @property
    def index(self):
        """Return the index values"""
        return self._obj[self.index_dim]

    @property
    def size(self):
        """Return the length of the index array"""
        return self._obj[self.index_dim].size

    @property
    def bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        return self.geometry.total_bounds

    @property
    def geometry(self) -> GeoSeries:
        """Return the geometry of the dataset or array as GeoSeries

        Returns
        -------
        GeoSeries
            A Series object with shapely geometries
        """
        if self._geometry is not None and self._geometry.index.size == self.size:
            return self._geometry
        gtype = self.geom_format
        if gtype not in ["geom", "xy", "wkt"]:
            raise ValueError("No valid geometry found in object.")
        if gtype == "geom":
            geoms = GeoSeries(
                data=self._obj[self.geom_name].values,
                index=self.index.values,
                crs=self.crs,
            )
        elif gtype == "xy":
            geoms = GeoSeries.from_xy(
                x=self._obj[self.x_name].values,
                y=self._obj[self.y_name].values,
                index=self.index.values,
                crs=self.crs,
            )
        elif gtype == "wkt":
            geoms = GeoSeries.from_wkt(
                data=self._obj[self.geom_name].values,
                index=self.index.values,
                crs=self.crs,
            )
        geoms.index.name = self.index_dim
        self._geometry = geoms
        return geoms

    def update_geometry(
        self,
        geometry: GeoSeries = None,
        geom_format: str = None,
        geom_name: str = None,
        x_name: str = None,
        y_name: str = None,
        replace: bool = True,
    ):
        if geom_format is None:
            geom_format = self.geom_format
        if geometry is None:
            geometry = self.geometry
        elif not isinstance(geometry, GeoSeries):
            raise ValueError(f"geometry should be a GeoSeries object")
        elif geometry.size != self.size:
            raise ValueError(
                f'The sizes of geometry and index dim "{self.index_dim}" do not match'
            )

        # update geometry and drop old
        drop_vars = []
        if self.geom_format != geom_format:
            if geom_format != "xy":
                drop_vars = [self.x_name, self.y_name]
            else:
                drop_vars = [self.geom_name]

        index_dim = self.index_dim
        if geom_format == "geom":
            if geom_name is None:
                geom_name = self.attrs.get("geom_name", "geometry")
            elif self.geom_name != geom_name:
                drop_vars.append(self.geom_name)
            coords = {geom_name: (self.index_dim, geometry.values)}
        elif geom_format == "wkt":
            if geom_name is None:
                geom_name = self.attrs.get("geom_name", "ogc_wkt")
            elif self.geom_name != geom_name:
                drop_vars.append(self.geom_name)
            coords = {geom_name: (self.index_dim, geometry.to_wkt().values)}
        elif geom_format == "xy":
            if x_name is None:
                x_name = self.attrs.get("x_name", "x")
            elif self.x_name != x_name:
                drop_vars.append(self.x_name)
            if y_name is None:
                y_name = self.attrs.get("y_name", "y")
            elif self.y_name != y_name:
                drop_vars.append(self.y_name)
            coords = {
                x_name: (self.index_dim, geometry.x.values),
                y_name: (self.index_dim, geometry.y.values),
            }
        obj = self._obj.copy()
        if replace:
            obj = obj.drop_vars(drop_vars, errors="ignore")
        obj = obj.assign_coords(coords)

        # reset spatial dims
        obj.vector._geometry = geometry
        obj.vector.set_spatial_dims(
            index_dim=index_dim,
            geom_format=geom_format,
            geom_name=geom_name,
            x_name=x_name,
            y_name=y_name,
        )
        obj.vector.set_crs(geometry.crs)
        return obj

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self, reducer = None):

        obj = self.update_geometry(geom_format="wkt", geom_name="ogc_wkt")
        obj["ogc_wkt"].attrs = {
            "long_name": "Geometry as ISO WKT",
            "grid_mapping": "spatial_ref",
        }
        index_dim = self.index_dim

        if isinstance(self._obj, xr.DataArray):
            if self._obj.name is None:
                self._obj.name = "values"
            obj = obj.to_dataset()

        if reducer is not None:
            rdims = [dim for dim in obj.dims if dim != index_dim]
            obj = obj.reduce(reducer, dim=rdims)

        dtypes = {"i": "Integer64", "f": "Real", "U": "String"}
        for name, da in obj.data_vars.items():
            if not index_dim in da.dims:
                continue
            if reducer is not None:
                rdims = [dim for dim in obj.dims if dim != index_dim]
                obj[name] = obj[name].reduce(reducer, rdims)
            # set ogr meta data
            dtype = dtypes.get(obj[name].dtype.kind, None)
            if dtype is not None:
                obj[name].attrs.update(
                    {
                        "ogr_field_name": f"{name}",
                        "ogr_field_type": dtype,
                    }
                )
                if dtype == "String":
                    obj[name].attrs.update(
                        {
                            "ogr_field_width": 100,
                        }
                    )
        obj = obj.drop_vars("spatial_ref", errors="ignore")
        obj.vector.set_crs(self.crs)
        obj = obj.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_VERSION}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{self.geom_type}",
            }
        )
        return obj

    def to_geom(self, geom_name) -> Union[xr.DataArray, xr.Dataset]:
        """Converts Dataset/ DataArray with xy or wkt geometries to shapely Geometries.

        Returns
        -------
        xr.Dataset
            Dataset with new geometry coordinates
        """
        return self.update_geometry(geom_format="geom", geom_name=geom_name)

    def to_xy(self, x_name="x", y_name="y") -> Union[xr.DataArray, xr.Dataset]:
        """Converts Dataset/ DataArray with Point geometries to x,y structure.

        Returns
        -------
        xr.Dataset
            Dataset with new x, y coordinates
        """
        return self.update_geometry(geom_format="xy", x_name=x_name, y_name=y_name)

    def to_wkt(
        self, ogr_compliant=False, reducer=None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Converts geometries in Dataset/DataArray to wkt strings.

        Returns
        -------
        xr.Dataset
            Dataset with new ogc_wkt coordinate
        """
        # ogr compliant naming and attrs
        if ogr_compliant:
            obj = self.ogr_compliant(reducer=reducer)
        else:
            obj = self.update_geometry(geom_format="wkt", geom_name="ogc_wkt")
        return obj

    ## clip
    def clip_geom(self, geom, predicate="intersects"):
        """Select all geometries that intersect with the input geometry.

        Arguments
        ---------
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        predicate : {None, 'intersects', 'within', 'contains', \
                     'overlaps', 'crosses', 'touches'}, optional
            If predicate is provided, the input geometry is tested
            using the predicate function against each item in the
            index whose extent intersects the envelope of the input geometry:
            predicate(input_geometry, tree_geometry).
        
        Returns
        -------
        da: xarray.DataArray
            Clipped DataArray
        """
        idx = gis_utils.filter_gdf(self.geometry, geom=geom, predicate=predicate)
        return self._obj.isel({self.index_dim: idx})

    def clip_bbox(self, bbox, crs=4326, buffer=None):
        """Select point locations to bounding box.

        Arguments
        ----------
        bbox: tuple of floats
            (xmin, ymin, xmax, ymax) bounding box
        buffer: float, optional
            buffer around bbox in crs units, None by default.

        Returns
        -------
        da: xarray.DataArray
            Clipped DataArray
        """
        if buffer is not None:
            bbox = np.atleast_1d(bbox)
            bbox[:2] -= buffer
            bbox[2:] += buffer
        idx = gis_utils.filter_gdf(
            self.geometry, bbox=bbox, crs=crs, predicate="intersects"
        )
        return self._obj.isel({self.index_dim: idx})

    ## wrap GeoSeries functions
    # TODO write general wrapper
    def to_crs(self, dst_crs):
        """Transform spatial coordinates to a new coordinate reference system.

        The ``crs`` attribute on the current GeoDataArray must be set.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)

        Returns
        -------
        xr.DataArray
            DataArray with transformed geospatial coordinates
        """
        if self.crs is None:
            raise ValueError("Source CRS is missing. Use da.vector.set_crs(crs) first.")
        geometry = self.geometry.to_crs(pyproj.CRS.from_user_input(dst_crs))
        return self.update_geometry(geometry)

    # Constructers
    # i.e. from other datatypes or files
    @staticmethod
    def from_gdf(gdf: GeoSeries | GeoDataFrame) -> xr.Dataset:
        if isinstance(gdf, GeoSeries):
            gdf = gdf.to_frame("geometry")
        geom_name = gdf._geometry_column_name
        index_dim = gdf.index.name
        if index_dim is None:
            gdf.index.name = "index"
        ds = gdf.to_xarray().set_coords(geom_name)
        ds.vector.set_spatial_dims(geom_name=geom_name, index_dim=index_dim)
        ds.vector.set_crs(gdf.crs)
        return ds

    ## Output methods
    ## Either writes to files or other data types
    def to_gdf(self, reducer=None):
        """Return geopandas GeoDataFrame with Point geometry based on Dataset
        coordinates. If a reducer is passed the Dataset variables are reduced along
        the all non-index dimensions and to a GeoDataFrame column.

        Arguments
        ---------
        reducer: callable
            input to ``xarray.DataArray.reducer`` func argument

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        gdf = self.geometry.to_frame("geometry")
        sdims = [self.y_name, self.x_name, self.index_dim, self.geom_name]
        for name in self._all_names:
            dims = self._obj[name].dims
            if name not in sdims:
                # keep 1D variables with matching index_dim
                if len(dims) == 1 and dims[0] == self.index_dim:
                    gdf[name] = self._obj[name].values
                # keep reduced data variables
                elif reducer is not None and self.index_dim in self._obj[name].dims:
                    rdims = [
                        dim for dim in self._obj[name].dims if dim != self.index_dim
                    ]
                    gdf[name] = self._obj[name].reduce(reducer, dim=rdims)
        return gdf

    def to_netcdf(
        self,
        path: str,
        ogr_compliant: bool = False,
        reducer=None,
        **kwargs,
    ):
        """Export geodataset vectordata to an ogr compliant netCDF4 file

        Parameters
        ----------
        path : str
            Output path for netcdf file.
        ogr_compliant : bool
            write the netCDF4 file in an ogr compliant format
            This makes it readable as a vector file in e.g. QGIS
        """
        self.to_wkt(
            ogr_compliant=ogr_compliant,
            reducer=reducer,
        ).to_netcdf(path, engine="netcdf4", **kwargs)


@xr.register_dataarray_accessor("vector")
class GeoDataArray(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataArray, self).__init__(xarray_obj)

    # Constructers
    # i.e. from other datatypes or files
    @staticmethod
    def from_netcdf(
        path: str,
        parse_geom=True,
        geom_name=None,
        x_name=None,
        y_name=None,
        crs=None,
        **kwargs,
    ) -> xr.DataArray:
        """_summary_

        Parameters
        ----------
        path : str
            _description_

        Returns
        -------
        xr.DataArray
            _description_
        """

        da = xr.open_dataarray(path, **kwargs)
        da.vector.set_spatial_dims(geom_name=geom_name, x_name=x_name, y_name=y_name)
        # force to geom_format "geom"
        if parse_geom:
            da = da.vector.update_geometry(geom_format="geom", replace=True)
        da.vector.set_crs(input_crs=crs)  # try to parse from netcdf if None
        return da


@xr.register_dataset_accessor("vector")
class GeoDataset(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    # Constructers
    # i.e. from other datatypes or filess
    @staticmethod
    def from_gdf(gdf: gpd.GeoDataFrame, geom_format="geom") -> xr.Dataset:
        """Creates Dataset with geospatial coordinates. The Dataset values are
        reindexed to the gdf index.

        Arguments
        ---------
        gdf: geopandas GeoDataFrame
            Spatial coordinates. The index should match the df index and the geometry
            columun may only contain Point geometries. Additional columns are also
            parsed to the xarray DataArray coordinates.

        Returns
        -------
        xr.Dataset
            Dataset with geospatial coordinates
        """
        if isinstance(gdf, GeoSeries):
            if gdf.name is None:
                gdf.name = "geometry"
            gdf = gdf.to_frame()
        if not isinstance(gdf, GeoDataFrame):
            raise ValueError(f"gdf data type not understood {type(gdf)}")
        geom_name = gdf.geometry.name
        ds = gdf.to_xarray().set_coords(geom_name)
        ds.vector.set_spatial_dims(geom_name=geom_name, geom_format=geom_format)
        ds.vector.set_crs(gdf.crs)
        return ds

    @staticmethod
    def from_netcdf(
        path: str,
        parse_geom=True,
        geom_name=None,
        x_name=None,
        y_name=None,
        crs=None,
        **kwargs,
    ) -> xr.Dataset:
        """Create GeoDataset from ogr compliant netCDF4 file

        Parameters
        ----------
        path : str
            Path to the netCDF4 file

        Returns
        -------
        xr.Dataset
            Dataset containing the geospatial data and attributes
        """
        ds = xr.open_dataset(path, **kwargs)
        ds.vector.set_spatial_dims(geom_name=geom_name, x_name=x_name, y_name=y_name)
        # force geom_format= 'geom'
        if parse_geom:
            ds = ds.vector.update_geometry(geom_format="geom", replace=True)
        ds.vector.set_crs(crs)  # try to parse from netcdf if None
        return ds
