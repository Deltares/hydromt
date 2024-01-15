"""Script for testing predefined data catalogs."""
import argparse

from hydromt import DataCatalog

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

if args.dataset:
    # test dataset
    pass

# Validate data catalog yaml with hydromt.validators.data_catalog
