# TODO: add back in after raster dataset is implemented
# class GridRegionSpecifyer(BaseModel):
#     """A region specified by a Rasterdataset."""

#     kind: Literal["grid"]
#     source: str
#     buffer: float = 0.0
#     driver_kwargs: Optional[Dict[str, Any]]

#     crs: CRS = Field(default=WGS84)
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def construct(self, DataCatalog) -> GeoDataFrame:
#         """Calculate the actual geometry based on the specification."""
#         raster_dataset = self.data_catalog.get_rasterdataset(
#             self.source, driver_kwargs=self.driver_kwargs
#         )
#         if raster_dataset is None:
#             raise ValueError("raster dataset was not found")
#         crs = raster_dataset.raster.crs
#         coord_index = cast(pd.Index, raster_dataset.coords.to_index())
#         dims_max = cast(np.ndarray, coord_index.max())
#         dims_min = cast(np.ndarray, coord_index.min())

#         # in raster datasets it is guaranteed that y_dim is penultimate dim and x_dim is last dim
#         geom: GeoDataFrame = GeoDataFrame(
#             geometry=[
#                 box(
#                     xmin=dims_min[-1],
#                     ymin=dims_min[-2],
#                     xmax=dims_max[-1],
#                     ymax=dims_max[-2],
#                 )
#             ],
#             crs=crs,
#         )

#         if self.buffer > 0:
#             if geom.crs.is_geographic:
#                 geom = cast(GeoDataFrame, geom.to_crs(3857))
#             geom = geom.buffer(self.buffer)

#         return geom

#     @model_validator(mode="after")
#     def _check_has_source(self) -> "GridRegionSpecifyer":
#         assert self.data_catalog.contains_source(self.source)
#         return self
