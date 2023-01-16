#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
#%%
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import geopandas.array as geoarray
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from osgeo import osr
import pyproj
import shapely
import logging

from hydromt import gis_utils, raster

logger = logging.getLogger(__name__)

#%%
class GeoBase(raster.XGeoBase):
    def __init__(self, xarray_obj):
        super(GeoBase, self).__init__(xarray_obj)
        self._geometry = None  # placeholder

    @property
    def _all_names(self):
        names = [n for n in self._obj.coords]
        if isinstance(self._obj, xr.Dataset):
            names = names + [n for n in self._obj.data_vars]
        return names

    @property
    def _geom_names(self):
        names = []
        for name in self._all_names:
            if self._obj[name].ndim == 1 and isinstance(
                self._obj[name][0].values.item(), BaseGeometry
            ):
                names.append(name)
        return names

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
            self.set_attrs(index_dim=index_dim)
        else:
            self.attrs.pop("x_name", None)
            self.attrs.pop("y_name", None)
            self.attrs.pop("index_dim", None)

    def _discover_geom(self, geom_name=None, index_dim=None):
        # infer geom dim
        if geom_name is None:
            names = self._geom_names
            if index_dim is not None:
                names = [name for name in names if self._obj[name].dims[0] == index_dim]
            if "geometry" in names:
                geom_name = "geometry"
            elif len(names) >= 1:
                geom_name = names[0]
        if (
            geom_name is not None
            and geom_name in self._obj
            and len(self._obj[geom_name].dims) == 1
        ):
            self.set_attrs(geom_name=geom_name)
            self.set_attrs(index_dim=self._obj[geom_name].dims[0])
        else:
            self.attrs.pop("geom_name", None)
            self.attrs.pop("index_dim", None)

    def set_spatial_dims(
        self, geom_name=None, x_name=None, y_name=None, index_dim=None
    ):
        breakpoint()
        self._discover_geom(geom_name=geom_name, index_dim=index_dim)
        if "geom_name" not in self.attrs:
            self._discover_xy(x_name=x_name, y_name=y_name, index_dim=index_dim)
        elif "geom_name" not in self.attrs:
            raise ValueError("No geometry data found. Provide the  geo.set_geometry")

    @property
    def geom_name(self):
        if self.get_attrs("geom_name") not in self._obj.dims:
            self.set_spatial_dims()
        if "geom_name" in self.attrs:
            return self.attrs["geom_name"]

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
    def index(self):
        return self._obj[self.index_dim]

    @property
    def length(self):
        return self._obj[self.index_dim].size

    @property
    def geometry(self):
        if (
            self._geometry is None
            or self._geometry.size != self.length
            or self._geometry.crs != self.crs
        ):
            if self.geom_name is not None:
                geoms = self._obj[self.geom_name].values
                self._geometry = geoarray.from_shapely(geoms, crs=self.crs)
            elif self.x_name is not None and self.y_name is not None:
                self._geometry = geoarray.points_from_xy(
                    self._obj[self.x_name].values,
                    self._obj[self.y_name].values,
                    crs=self.crs,
                )
        return self._geometry

    @property
    def bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        return self.geometry.total_bounds

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
        gdf = gpd.GeoDataFrame(index=self.index, geometry=self.geometry, crs=self.crs)
        gdf.index.name = self.index_dim
        sdims = [self.y_name, self.x_name, self.index_dim, "geometry"]
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
                    gdf[name] = self._obj[name].reduce(reducer, rdim=dims)
        return gdf

    def to_crs(self, dst_crs):
        """Transform spatial coordinates to a new coordinate reference system.

        The ``crs`` attribute on the current GeoDataArray must be set.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)

        Returns
        -------
        da: xarray.DataArray
            DataArray with transformed geospatial coordinates
        """
        if self.crs is None:
            raise ValueError("Source CRS is missing. Use da.vector.set_crs(crs) first.")
        obj = self._obj.copy()
        geoms = self.geometry.to_crs(pyproj.CRS.from_user_input(dst_crs))
        if self.x_name and self.y_name:
            obj = obj.assign_coords(
                points_to_coords(
                    geoms,
                    x_name=self.x_name,
                    y_name=self.y_name,
                    index_dim=self.index_dim,
                )
            )
        if self.geom_name:
            obj = obj.assign_coords(
                {self.geom_name: xr.Variable(self.index_dim, geoms)}
            )
        obj.geo._geometry = geoms
        obj.geo.set_crs(dst_crs)
        return obj

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
        idx = gis_utils.filter_gdf(self.geometry, geom=geom, predicate="intersects")
        return self._obj.isel({self.index_dim: idx})

    def clip_bbox(self, bbox, buffer=None):
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
        idx = gis_utils.filter_gdf(self.geometry, bbox=bbox, predicate="intersects")
        return self._obj.isel({self.index_dim: idx})

    def to_wkt(self):
        obj = self._obj.copy()
        for name in self._geom_names:
            geoms = geoarray.to_wkt(ds_cities.geo.geometry)
            obj[self.geom_name] = xr.Variable(self.index_dim, geoms)
        return obj

    def to_netcdf(self, path, **kwargs):
        self.to_wkt().to_netcdf(path, **kwargs)

    def to_file(self, path, **kwargs):
        self.to_gdf.to_file()


@xr.register_dataarray_accessor("geo")
class GeoDataset(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)

    @staticmethod
    def from_dataset(ds, crs=None, geom_name=None, x_name=None, y_name=None):
        ds.geo.set_spatial_dims(geom_name=geom_name, x_name=x_name, y_name=y_name)
        ds.geo.set_crs(crs)
        return ds


@xr.register_dataset_accessor("geo")
class GeoDataset(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)

    @staticmethod
    def ogr_compliant(ds: xr.Dataset):
        """Create a ogr compliant version of a xarray Dataset

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing geospatial data

        Returns
        -------
        xarray.Dataset
            ogr compliant
        """

        _linkTable = {
            int: "Integer64",
            float: "Real",
            str: "String",
        }

        def Type(val):
            try:
                s = eval(val)
            except Exception:
                s = val
            return type(s)

        def FieldType(lst):
            if float or np.float64 in lst:
                type = float
            else:
                type = int
            if str in lst:
                type = str
            return type

        wkt = [g.wkt for g in ds.geometry.values]

        ## Determine Geometry type
        from osgeo.ogr import CreateGeometryFromWkt

        geom_types = [CreateGeometryFromWkt(g).GetGeometryName() for g in wkt]

        if len(set(geom_types)) > 1:
            i = ["MULTI" in g for g in geom_types].index(True)
            geom_type = geom_types[i]
        else:
            geom_type = geom_types[0]

        del geom_types

        ## Create the geometry DataArray
        ogc_wkt = xr.DataArray(
            data=wkt,
            dims="record",
            attrs={
                "long_name": "Geometry as ISO WKT",
                "grid_mapping": "crs",
            },
        )

        # Set spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(ds.geo.crs.to_epsg())

        crs = xr.DataArray(
            data=int(0),
            attrs={
                "long_name": "CRS definition",
                "crs_wkt": srs.ExportToWkt(),
                "spatial_ref": srs.ExportToWkt(),
            },
        )

        out_ds = xr.Dataset({"ogc_wkt": ogc_wkt})

        out_ds = out_ds.assign({"crs": crs})

        for fld_header, da in ds.data_vars.items():
            types = tuple(map(Type, da.values))

            fld_type = FieldType(types)

            temp_da = xr.DataArray(
                data=da.values,
                dims="record",
                attrs={
                    "ogr_field_name": f"{fld_header}",
                    "ogr_field_type": _linkTable[fld_type],
                },
            )

            if fld_type == str:
                temp_da.attrs.update({"ogr_field_width": 100})
            out_ds = out_ds.assign({f"{fld_header}": temp_da})

            del temp_da

        from osgeo import __version__ as GDAL_verion

        out_ds = out_ds.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_verion}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{geom_type}",
            }
        )
        return out_ds

    @staticmethod
    def from_gdf(gdf):
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
        ds: xarray.Dataset
            Dataset with geospatial coordinates
        """
        if isinstance(gdf, gpd.GeoSeries):
            if gdf.name is None:
                gdf.name = "geometry"
            gdf = gdf.to_frame()
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError(f"gdf data type not understood {type(gdf)}")
        geom_name = gdf.geometry.name
        ds = gdf.to_xarray().set_coords(geom_name)
        ds.geo.set_crs(gdf.crs)
        return ds

    @staticmethod
    def from_dataset(ds, crs=None, geom_name=None, x_name=None, y_name=None):
        ds.geo.set_spatial_dims(geom_name=geom_name, x_name=x_name, y_name=y_name)
        ds.geo.set_crs(crs)
        return ds

    @staticmethod
    def from_nc(nc: str):
        """Create GeoDataset from ogr compliant netCDF4 file

        Parameters
        ----------
        nc : str
            Path to the netCDF4 file

        Returns
        -------
        xarray.Dataset
            Dataset containing the geospatial data and attributes
        """
        temp = xr.open_dataset(nc)
        geom = [shapely.wkt.loads(g) for g in temp.ogc_wkt.values]

        ds = xr.Dataset(
            coords={
                "index": temp.record.values,
                "geometry": ("index", geom),
                "spatial_ref": temp.crs,
            }
        )

        for key, da in temp.drop_vars(["ogc_wkt", "crs"]).data_vars.items():
            temp_da = xr.DataArray(data=da.values, dims="index")
            ds = ds.assign({key: temp_da})

        ds.geo.set_crs(pyproj.CRS.from_wkt(temp.crs.crs_wkt))

        return ds

    def add_data(self, data_vars, coords=None, index_dim=None):
        """Align data along index axis and data to GeoDataset

        Arguments
        ---------
        data_vars: dict-like, DataArray or Dataset
            A mapping from variable names to `xarray.DataArray` objects.
            See :py:func:`xarray.Dataset` for all options.
            Additionally, it accepts `xarray.DataArray` with name property and `xarray.Dataset`.
        coords: sequence or dict of array_like, optional
            Coordinates (tick labels) to use for indexing along each dimension.
        index_dim: str, optional
            Name of index dimension in data_vars

        Returns
        -------
        ds: xarray.Dataset
            merged dataset
        """
        if isinstance(data_vars, xr.DataArray) and data_vars.name is not None:
            data_vars = data_vars.to_dataset()
        if isinstance(data_vars, xr.Dataset):
            ds_data = data_vars
        else:
            ds_data = xr.Dataset(data_vars, coords=coords)
        # check if any data array contain index_dim
        if self.index_dim not in ds_data.dims and index_dim in ds_data:
            ds_data = ds_data.rename({index_dim: self.index_dim})
        elif self.index_dim not in ds_data.dims:
            raise ValueError(f"Index dimension {self.index_dim} not found in dataset.")
        ds_data = ds_data.reindex({self.index_dim: self.index}).transpose(
            self.index_dim, ...
        )
        return xr.merge([self._obj, ds_data])

    def to_nc(
        self,
        root: str,
        fname: str,
    ):
        """Export geodataset vectordata to an ogr compliant netCDF4 file

        Parameters
        ----------
        root : str
            Directory in which the file is written to
        fname: : str
            Name of the file
        """

        temp = self.ogr_compliant(self._obj)

        temp.to_netcdf(f"{root}\\{fname}.nc", engine="netcdf4")

        del temp


def points_to_coords(
    geometry: geoarray.GeometryArray, x_name="x", y_name="y", index_dim="index"
):
    """Returns coordinate dictionary with xarray IndexVariables based on point geometries."""
    if not np.all(geometry.geom_type == "Point"):
        raise ValueError("geometry may only contain points.")
    coords = {
        index_dim: xr.IndexVariable(index_dim, np.arange(geometry.size, dtype="int")),
        x_name: xr.IndexVariable(index_dim, geometry.x),
        y_name: xr.IndexVariable(index_dim, geometry.y),
    }
    return coords


def gdf_to_xarray(gdf, keep_cols):
    geom_col = gdf.geometry.name
    crs = gdf.crs
    index_col = "index" if gdf.index.name is None else gdf.index.name
    geom_coord = xr.Variable(index_col, GeometryArray2(gdf.geometry.values, crs=crs))  #
    ds = gdf.drop(
        columns=geom_col
    ).to_xarray()  # .assign_coords(**{geom_col: gdf.geometry.values})


# #%%
# df = pd.DataFrame(
#     {
#         "city": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
#         "country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
#         "latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
#         "longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
#     }
# )
# df.to_xarray().geo.geometry
# #%%
# ds_cities = GeoDataset.from_dataset(df.to_xarray(), crs=4326)
# gdf_cities = ds_cities.geo.to_gdf()
# ds_cities = ds_cities.geo.to_crs(3857)
# ds_cities

# #%%
# gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# ds = GeoDataset.from_gdf(gdf)
# ds

# #%%
# GeoDataset.from_gdf(gdf['geometry']).geo.add_data(gdf.drop(columns='geometry').to_xarray())

# #%%
# ds_cities.geo.clip_geom(gdf[gdf['name']=='Venezuela'].geometry)
# ds_cities.geo.to_crs(4326).geo.clip_bbox(gdf[gdf['name']=='Venezuela'].total_bounds)

# #%%
# ds

# %%
