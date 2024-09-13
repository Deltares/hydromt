.. _custom_driver:

Custom Drivers
==============

Drivers are the second ingredient necessary to read (custom) datasets. Where as
`URIResolver`s (discussed previously) determines _what_ to read, drivers determine _how_ to
read and slice it. For example, you might use the same `ConventionResolver` to determine
which files to read, but different drivers to read `.nc`, `.zarr` or `.tiff` files. Vice
versa you might also use the same driver but different resolvers to read data depending
on how it was organised. It should therefore be noted that to read a particular dataset
it might be necessary to implement a custom resolver as well as a custom driver.

Implementing a Drivers
^^^^^^^^^^^^^^^^^^^^^^

to function drivers MUST at least implement the following function:

.. code-block:: python

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> Any:
        ...

You may optionally also implement the following function as is appropriate:


.. code-block:: python

    def write(
        self,
        path: StrPath,
        data: Any,
        **kwargs,
    ):
        ...

As can be seen, `read` takes all of the URIs that the `resolver` has produced and will
attempt to read data from them in whatever way is appropriate. Since the contents of the
files themselves might also have to be sliced it is advisable to take the same arguments
as the resolver is able to take.
