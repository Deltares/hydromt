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
from hydromt.raster import XDIMS, YDIMS

from osgeo import __version__ as GDAL_VERSION

logger = logging.getLogger(__name__)


def Type(val):
    # try:
    #     s = eval(val)
    # except Exception:
    #     s = val
    return type(val)


def FieldType(lst):
    if float or np.float64 in lst:
        type = float
    else:
        type = int
    if str in lst:
        type = str
    return type


def DeterGeomType(wkt):
    from osgeo.ogr import CreateGeometryFromWkt

    geom_types = [CreateGeometryFromWkt(g).GetGeometryName() for g in wkt]

    if len(set(geom_types)) > 1:
        i = ["MULTI" in g for g in geom_types].index(True)
        geom_type = geom_types[i]
    else:
        geom_type = geom_types[0]
    del geom_types
    return geom_type


_linkTable = {
    int: "Integer64",
    float: "Real",
    str: "String",
}


class GeoBase(raster.XGeoBase):
    def __init__(self, xarray_obj):
        super(GeoBase, self).__init__(xarray_obj)

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
            self.set_attrs(geom_type="xy")
            self.set_attrs(index_dim=index_dim)
        else:
            self.attrs.pop("x_name", None)
            self.attrs.pop("y_name", None)
            self.attrs.pop("geom_type", None)
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
            self.set_attrs(geom_type=types[names.index(geom_name)])
            self.set_attrs(index_dim=self._obj[geom_name].dims[0])
        else:
            self.attrs.pop("geom_name", None)
            self.attrs.pop("geom_type", None)
            self.attrs.pop("index_dim", None)

    def set_spatial_dims(
        self,
        geom_name=None,
        x_name=None,
        y_name=None,
        index_dim=None,
        geom_type=None,
    ):
        if geom_type != "xy":
            self._discover_geom(geom_name=geom_name, index_dim=index_dim)
        if "geom_name" not in self.attrs:
            self._discover_xy(x_name=x_name, y_name=y_name, index_dim=index_dim)
        elif "geom_name" not in self.attrs:
            raise ValueError("No geometry data found.")

    def set_geometry(
        self,
        geometry: GeoSeries,
        geom_type="geom",
        geom_name="geometry",
        x_name="x",
        y_name="y",
        index_dim=None,
        drop_old_geometry=True,
    ) -> Union[xr.DataArray, xr.Dataset]:
        if index_dim is None:
            index_dim = self.index_dim
        if geometry.size != self.size:
            raise ValueError(
                f'The sizes of geometry and index dim "{index_dim}" do not match'
            )
        if not isinstance(geometry, GeoSeries):
            raise ValueError(f"geometry should be a GeoSeries object")
        obj = self._obj

        if drop_old_geometry:
            for name in ["x_name", "y_name", "geom_name"]:
                if name in self.attrs and self.attrs[name] in obj:
                    obj = obj.drop_vars(self.attrs[name])
                    obj.vector.attrs.pop(name)

        if geom_type in ["geom", "wkt"]:
            if geom_name in self._all_names:
                obj = obj.drop_vars(geom_name)
            if geom_type == "geom":
                obj = obj.assign_coords({geom_name: (index_dim, geometry.values)})
            else:
                obj = obj.assign_coords(
                    {geom_name: (index_dim, geometry.to_wkt().values)}
                )
            obj.vector.set_spatial_dims(index_dim=index_dim, geom_name=geom_name)
        elif geom_type == "xy":
            obj = obj.assign_coords(
                {
                    x_name: (self.index_dim, geometry.x.values),
                    y_name: (self.index_dim, geometry.y.values),
                },
            )
            obj.vector.set_spatial_dims(
                index_dim=index_dim, x_name=x_name, y_name=y_name
            )
        return obj

    @property
    def geom_type(self):
        if self.get_attrs("geom_type") not in self._obj.dims:
            self.set_spatial_dims()
        if "geom_type" in self.attrs:
            return self.attrs["geom_type"]

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
        gtype = self.geom_type
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
        return geoms

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self):
        """ """

        wkt = self.geometry.to_wkt()

        ## Determine Geometry type
        geom_type = DeterGeomType(wkt)

        ## Create the geometry DataArray
        # TODO add coords?
        ogc_wkt = xr.DataArray(
            data=wkt,
            dims="record",
            attrs={
                "long_name": "Geometry as ISO WKT",
                "grid_mapping": "spatial_ref",
            },
        )

        # Set spatial reference
        # srs = pyproj.CRS.from_epsg(self.crs.to_epsg())

        # crs = xr.DataArray(
        #     data=int(1),
        #     attrs={
        #         "long_name": "CRS definition",
        #         "crs_wkt": srs.to_wkt(),
        #         "spatial_ref": srs.to_wkt(),
        #     },
        # )

        return ogc_wkt, geom_type

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
        geoms = self.geometry.to_crs(pyproj.CRS.from_user_input(dst_crs))
        obj = self.set_geometry(
            geoms,
            geom_type=self.geom_type,
            geom_name=self.geom_name,
            y_name=self.y_name,
            x_name=self.x_name,
        )
        obj.vector.set_crs(dst_crs)
        return obj

    def to_xy(self, x_name="x", y_name="y") -> Union[xr.DataArray, xr.Dataset]:
        """Converts Dataset/ DataArray with Point geometries to x,y structure.

        Returns
        -------
        xr.Dataset
            Dataset with new x, y coordinates
        """
        return self.set_geometry(
            self.geometry, geom_type="xy", y_name=y_name, x_name=x_name
        )

    def to_wkt(self) -> Union[xr.DataArray, xr.Dataset]:
        """Converts geometries in Dataset/DataArray to wkt strings.

        Returns
        -------
        xr.Dataset
            Dataset with new ogc_wkt coordinate
        """
        return self.set_geometry(self.geometry, geom_type="wkt", geom_name="ogc_wkt")

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
            self.to_gdf(), bbox=bbox, crs=crs, predicate="intersects"
        )
        return self._obj.isel({self.index_dim: idx})

    # Constructers
    # i.e. from other datatypes or files
    @staticmethod
    def from_gdf(gdf: GeoSeries | GeoDataFrame) -> xr.Dataset:
        if isinstance(gdf, GeoSeries):
            gdf = gdf.to_frame("geometry")
        geom_name = gdf._geometry_column_name
        ds = gdf.to_xarray().set_coords(geom_name)
        ds.vector.set_meta()
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

        if ogr_compliant:
            self.ogr_compliant().to_netcdf(path, engine="netcdf4", **kwargs)

        if self.geom_type == "geom":
            self.to_wkt().to_netcdf(path, engine="netcdf4", **kwargs)


@xr.register_dataarray_accessor("vector")
class GeoDataArray(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataArray, self).__init__(xarray_obj)

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self, reducer=None) -> xr.Dataset:
        """Create a ogr compliant version of a xarray DataArray
        Note(!): The result will not be a DataArray

        Returns
        -------
        xr.Dataset
            ogr compliant
        """

        ogc_wkt, geom_type = super().ogr_compliant()

        out_ds = xr.Dataset()

        out_ds = out_ds.assign_coords(
            {
                "ogc_wkt": ogc_wkt,
            },
        )

        out_ds.vector.set_crs(self.crs.to_epsg())

        types = tuple(map(Type, self._obj.values))

        fld_type = FieldType(types)

        if self._obj.name is None:
            name = "value"
        else:
            name = self._obj.name

        values = None
        if not len(self._obj.dims) == 1 and list(self._obj.dims)[0] == self.index_dim:
            if reducer is not None:
                idd = self._obj.dims.index(self.index_dim)
                reduced_dims = list(range(len(self._obj.dims)))
                reduced_dims.remove(idd)
                values = reducer(self._obj.values, axis=tuple(reduced_dims))

        temp_da = xr.DataArray(
            data=self._obj.values,
            dims="record",
            attrs={
                "ogr_field_name": f"{name}",
                "ogr_field_type": _linkTable[fld_type],
            },
        )

        if fld_type == str:
            temp_da.attrs.update({"ogr_field_width": 100})
        out_ds = out_ds.assign({f"{name}": temp_da})

        del temp_da

        out_ds = out_ds.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_VERSION}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{geom_type}",
            }
        )
        return out_ds

    # Constructers
    # i.e. from other datatypes or files
    @staticmethod
    def from_netcdf(
        path: str, geom_name=None, x_name=None, y_name=None, crs=None, **kwargs
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

        da_tmp = xr.open_dataarray(path, **kwargs)
        da_tmp.vector.set_spatial_dims(
            geom_name=geom_name, x_name=x_name, y_name=y_name
        )
        # force to geom_type "geom"
        da = da_tmp.vector.set_geometry(
            geometry=da_tmp.vector.geometry,
            geom_type="geom",
            geom_name=da_tmp.attrs.get("geom_name", "geometry"),
        )
        da.vector.set_crs(crs=crs)  # try to parse from netcdf if None
        return da

    ## Output methods
    ## Either writes to files or other data types
    def to_gdf(self, reducer=None) -> gpd.GeoDataFrame:
        """Return geopandas GeoDataFrame with Point geometry based on DataArray
        coordinates. If a reducer is passed the DataArray variables are reduced along
        the all non-index dimensions and to a GeoDataFrame column.

        Arguments
        ---------
        reducer: callable
            input to ``xarray.DataArray.reducer`` func argument

        Returns
        -------
        gdf: gpd.GeoDataFrame
            GeoDataFrame
        """
        gdf = super().to_gdf(reducer)

        if self._obj.name is None:
            name = "value"
        else:
            name = self._obj.name

        gdf[name] = self._obj.values

        return gdf


@xr.register_dataset_accessor("vector")
class GeoDataset(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self, reducer=None) -> xr.Dataset:
        """Create a ogr compliant version of a xarray Dataset

        Returns
        -------
        xr.Dataset
            ogr compliant
        """

        ogc_wkt, geom_type = super().ogr_compliant()

        out_ds = xr.Dataset()

        out_ds = out_ds.assign_coords(
            {"ogc_wkt": ogc_wkt},
        )

        # Set the spatial reference with the
        # set_crs form the GeoBase class
        out_ds.vector.set_crs(self.crs.to_epsg())

        for fld_header, da in self._obj.data_vars.items():
            values = None
            if not self.index_dim in da.dims:
                continue
            if not len(da.dims) == 1:
                if reducer is not None:
                    idd = da.dims.index(self.index_dim)
                    reduced_dims = list(range(len(da.dims)))
                    reduced_dims.remove(idd)
                    values = reducer(da.values, axis=tuple(reduced_dims))
                else:
                    continue

            if values is None:
                values = da.values

            types = tuple(map(Type, values))

            fld_type = FieldType(types)

            temp_da = xr.DataArray(
                data=values,
                dims="record",
                attrs={
                    "ogr_field_name": f"{fld_header}",
                    "ogr_field_type": _linkTable[fld_type],
                },
            )

            if fld_type == str:
                temp_da.attrs.update({"ogr_field_width": 100})
            out_ds = out_ds.assign({f"{fld_header}": temp_da})

            del temp_da, values

        out_ds = out_ds.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_VERSION}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{geom_type}",
            }
        )
        return out_ds

    # Constructers
    # i.e. from other datatypes or filess
    @staticmethod
    def from_gdf(gdf: gpd.GeoDataFrame, geom_type="geom") -> xr.Dataset:
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
        ds.vector.set_spatial_dims(geom_name=geom_name, geom_type=geom_type)
        ds.vector.set_crs(gdf.crs)
        return ds

    @staticmethod
    def from_netcdf(
        path: str, geom_name=None, x_name=None, y_name=None, crs=None, **kwargs
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
        ds_tmp = xr.open_dataset(path, **kwargs)
        ds_tmp.vector.set_spatial_dims(
            geom_name=geom_name, x_name=x_name, y_name=y_name
        )
        ds = ds_tmp.vector.set_geometry(
            geometry=ds_tmp.vector.geometry,
            geom_type="geom",
            geom_name=ds_tmp.attrs.get("geom_name", "geometry"),
        )
        ds.vector.set_crs(crs)  # try to parse from netcdf if None
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

    ## Output methods
    ## Either writes to files or other data types


# %%
