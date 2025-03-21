.. _custom_behaviour:


Custom Resolver
===============

In addition to data which may or may not be in custom file formats, to be able to read
data, we must also know how it is laid out. For example the AWS Copernicus DEM data
(https://registry.opendata.aws/copernicus-dem/) is provided as Cloud Optimised GeoTiff,
but has no spacial index. Instead the files are divided in tiles covering the globe and
must be queried using the desired resolution, nothing and easting to find the correct
files. To be able to query this data set effectively we need a way to translate geospatial
data into a list of files which we should load to get the data we want. This is
exactly where ``URIResolver`` s come in.

A `URIResolver` is what we use to generate a list of uris to the files we wish to query.
Not that a uri may be, but does not have to be a local file path. Instead URIs could
also be URLs, REST API paths, socket based file descriptors or anything else one might
be able to read data from. Although it may be tempting to think of a URI as a file
path, we discourage developers from doing so as this may exlucde users from, for
example, talking to databases to query data.

Implementing a Resolver
^^^^^^^^^^^^^^^^^^^^^^^

Ultimately a resolver has only one public method that it must implement:

.. code-block:: python

    def resolve(
        self,
        uri: str,
        *,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[Zoom] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
        variables: Union[int, tuple[float, str], None] = None,
        metadata: Optional[SourceMetadata],
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> List[str]:
        ...

As you can see it takes in a single uri string and will produce from that a list of
other strings. These strings will then be passed to the appropriate driver to be read
out. How the driver will use them will be discussed more indepth in that section, but it
is not the URI resolver's job to any data, (though it may do that if necessary for the
resolution, such as reading metadata from potentially interesting files). You may do this in any way you like, such as, but not limited to:

1. talking to a API or service
2. reading files from disks
3. produce a-priori defined answers

Your resolver may also take any additional arguments it may need to resolve which data
to read. You may add any arguments here you like, although it is encouraged for
backwards compatibility reasons to only add named arguments (i.e. after `*` as above).

Handling non-existent data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Note the `handle_nodata` argument. It is highly encouraged to take this argument and
handle it in a similar way that core does. This argument is an enum that has two
variants: `RAISE`, `WARN` and `IGNORE`. Your users may (eroniously or not) request data
that does not exist. Sometimes this is critical, sometimes not. For example, depending
on the region you are working in, there may or may not be glaciers in your model, but
this should not prevent you from running the model. Users can tell you what you should
do if this does not exist by passing you this enum.

As one might expect, when RAISE is passed and some data does not exist, this will
immediately raise an exception that the user can catch but will not execute any further
code until the user does so. `WARN` and `IGNORE` are slightly different though. They are
identical except for whether a warning will be emitted in the logs or not. Other than
that, both the options will cause the current code to simply not do any more work.
Typically this is done by returning `None` from the function in questions, with all
subsequent functions checking if the input they got from the previous step is `None` and
also returning `None` as well if they do.

We have chosen to return `None` instead of an empty dataset because it is easier to tell
that something has gone wrong, and is easier to check for in Python code. In addition it
makes a clearer distinction between data that was not found vs data that might be
missing. For the sake of consistency we ask that you do the same when implementing your
own resolvers.
