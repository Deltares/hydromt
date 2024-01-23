"""Script for testing predefined data catalogs."""
import argparse
import os

from hydromt import DataCatalog
from hydromt.log import setuplog

logger = setuplog()

parser = argparse.ArgumentParser("Test predefined data catalog")
parser.add_argument("data_catalog", help="The data catalog to test")
parser.add_argument(
    "-ds", "--dataset", help="The name of the dataset to test", required=False
)
parser.add_argument(
    "-dsv" "--dataset_version",
    help="Optional version of the dataset to test for",
    required=False,
)

args = parser.parse_args()
datacatalog = DataCatalog(data_libs=args.data_catalog)
error_count = 0

if args.dataset:
    # test dataset
    pass

logger.info("Checking paths of data catalog sources")
for source in datacatalog.sources.keys():
    logger.info(f"Checking paths of {source}")
    paths = datacatalog.get_source(source)._resolve_paths()
    for path in paths:
        try:
            assert os.path.exists(path)
        except AssertionError:
            error_count += 1
            logger.error(f"{source} file not found in path: {path}")


# Use resolve paths to check if the files exist
# Validate data catalog yaml with hydromt.validators.data_catalog
logger.info(f"Encountered {error_count} errors")
