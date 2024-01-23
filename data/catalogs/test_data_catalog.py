"""Script for testing predefined data catalogs."""
import argparse

from dask.distributed import Client

from hydromt import DataCatalog
from hydromt.log import setuplog
from hydromt.validators.data_catalog import DataCatalogValidator

client = Client(processes=False)
logger = setuplog()


error_count = 0


def test_dataset(args, datacatalog):
    """Tests a given dataset on opening the dataset and minimal processing.

    Datasets containing geo data will be clipped to a small bounding box. This bounding box can
    be given by a user with the -bbox flag.
    """
    source = datacatalog.get_source(source=args.dataset, version=args.dataset_version)
    if args.boundingbox:
        bbox = args.boundingbox
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
    logger.info(f"Encountered {error_count} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test predefined data catalog")
    parser.add_argument("data_catalog", help="The data catalog to test")
    parser.add_argument(
        "-ds", "--dataset", help="The name of the dataset to test", required=False
    )
    parser.add_argument(
        "-dsv",
        "--dataset_version",
        help="Optional version of the dataset to test for",
        required=False,
    )
    parser.add_argument(
        "-bbox",
        "--boundingbox",
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
