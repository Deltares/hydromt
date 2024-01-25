"""A command line script for testing predefined data catalogs.

This script allows for testing separate data catalog items and the complete data catalog.
When testing the data catalog all the paths of the data catalog sources are
checked if they exist.
Call for testing the data catalog:
    'python test_data_catalog.py path/to/datacatalog.yml'

Separate data catalog sources can be tested by giving the dataset source name as
an extra argument. The argument should be preceded by -ds. It is also possible to specify
the version of the dataset source by supplying the call with a -dsv flag and version
number. The dataset is opened and if the dataset contains geo data the dataset is clipped
to a small bounding box surrounding Deltares Delft office ( 4.333409, 51.962159,
4.42835, 52.006873,).
Example dataset source test call:
    'python test_data_catalog.py path/to/datacatalog.yml -d chelsa -v 1.2
    -r "{'bbox':[4.333409,51.962159,4.42835,52.006873]}"'

In addition the passed data catalog yaml is checked if it is a valid data catalog yaml.

"""
import argparse
import json

from dask.distributed import Client

from hydromt import DataCatalog
from hydromt.log import setuplog
from hydromt.validators.data_catalog import DataCatalogValidator


def test_dataset(args, datacatalog):
    """Tests a given dataset on opening the dataset and minimal processing.

    Datasets containing geo data will be clipped to a small bounding box. This bounding box can
    be given by a user with the -bbox flag.
    """
    source = datacatalog.get_source(source=args.dataset, version=args.dataset_version)
    if args.region:
        region_json = args.region.replace("'", '"')
        bbox = json.loads(region_json)["bbox"]
        print(bbox)
    else:
        bbox = [
            4.333409,
            51.962159,
            4.42835,
            52.006873,
        ]  # small area surrounding Deltares Delft office
    logger.info(
        f"Retrieving {args.dataset} {source.data_type} and clipping by {bbox} bounding box"
    )
    if source.data_type == "GeoDataFrame":
        dataset = datacatalog.get_geodataframe(args.dataset, bbox=bbox)
    elif source.data_type == "GeoDataset":
        dataset = datacatalog.get_geodataset(args.dataset, bbox=bbox)
    elif source.data_type == "RasterDataset":
        dataset = datacatalog.get_rasterdataset(args.dataset, bbox=bbox)
    elif source.data_type == "DataFrame":
        dataset = datacatalog.get_dataframe(args.dataset)
    elif source.data_type == "Dataset":
        dataset = datacatalog.get_dataset(args.dataset)
    assert dataset is not None


def test_data_catalog(args, datacatalog):
    """Tests the paths of the given data catalog."""
    error_count = 0
    logger.info("Checking paths of data catalog sources")
    for source in datacatalog.get_source_names():
        try:
            logger.info(f"Checking paths of {source}")
            datacatalog.get_source(source)._resolve_paths()
        except FileNotFoundError as e:
            logger.error(f"File not found for dataset source {source}: {e}")

            error_count += 1
        except ValueError as e:
            logger.error(
                f"Something went wrong with creating path string for dataset source {source}: {e}"
            )
            error_count += 1
    if error_count > 0:
        logger.error(f"Encountered {error_count} errors")


if __name__ == "__main__":
    client = Client(processes=False)
    logger = setuplog()
    parser = argparse.ArgumentParser("Test predefined data catalog")
    parser.add_argument("data_catalog", help="The data catalog to test")
    parser.add_argument(
        "-d", "--dataset", help="The name of the dataset to test", required=False
    )
    parser.add_argument(
        "-v",
        "--dataset_version",
        help="Optional version of the dataset to test for",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--region",
        help="Bounding box for testing clipping spatial data. Bounding box must"
        " be a list of xmin, ymin, xmax, ymax in WGS84 EPSG:4326 coordinates.",
        required=False,
    )

    args = parser.parse_args()
    datacatalog = DataCatalog(data_libs=args.data_catalog)
    if args.dataset:
        test_dataset(args, datacatalog)
    else:
        test_data_catalog(args, datacatalog)
    # Validate data catalog yaml
    logger.info("Validating data catalog")
    DataCatalogValidator().from_yml(args.data_catalog)
    logger.info("Data catalog is valid!")
