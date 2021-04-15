#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="hydromt",
    description="HydroMT: Build and analyze models like a data-wizard!",
    long_description=readme + "\n\n",
    url="https://github.com/Deltares/hydromt",
    author="Deltares",
    author_email="dirk.eilander@deltares.nl",
    packages=find_packages(),
    package_dir={"hydromt": "hydromt"},
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    install_requires=[
        "click",
        "bottleneck",
        "entrypoints",
        "xarray",
        "rasterio",
        "dask",
        "pandas",
        "pygeos",
        "geopandas>=0.8",
        "numpy",
        "scipy",
        "affine",
        "pyflwdir>=0.4.5",
        "openpyxl",
        "zarr",
        "toml",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "sphinx", "sphinx_rtd_theme", "black"],
        "optional": [],
    },
    entry_points={
        "console_scripts": ["hydromt = hydromt.cli.main:main"],
    },
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="hydrology models data-science hydromt wflow",
)
