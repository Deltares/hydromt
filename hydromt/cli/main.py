# -*- coding: utf-8 -*-
"""command line interface for hydromt models."""

import logging
from json import loads as json_decode
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import numpy as np

from hydromt.data_catalog import DataCatalog
from hydromt.typing import ExportConfigDict
from hydromt.validators.data_catalog import DataCatalogValidator

from .. import __version__, log
from ..models import MODELS
from . import cli_utils

BUILDING_EXE = False
if BUILDING_EXE:
    import sys

    exepath = sys.prefix
    import pyproj

    pyproj_datadir = join(exepath, "proj-data")
    pyproj.datadir.set_data_dir(pyproj_datadir)

logger = logging.getLogger(__name__)


def print_models(ctx, param, value):
    """Print the available models and exit.

    Parameters
    ----------
    ctx : click.Context
        The Click context object.
    param : click.Parameter
        The Click parameter object.
    value : bool
        The value of the parameter.
    """
    if not value:
        return {}
    click.echo(f"{MODELS}")
    ctx.exit()


## common arguments / options
opt_config = click.option(
    "-i",
    "--config",
    type=click.Path(resolve_path=True),
    help="Path to hydroMT configuration file, for the model specific implementation.",
)
export_dest_path = click.argument(
    "export_dest_path",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)
arg_root = click.argument(
    "MODEL_ROOT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)
arg_baseroot = click.argument(
    "BASEMODEL_ROOT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)

region_opt = click.option(
    "-r",
    "--region",
    type=str,
    default="{}",
    callback=cli_utils.parse_json,
    help="Set the region for which to build the model,"
    " e.g. {'subbasin': [-7.24, 62.09]}",
)

verbose_opt = click.option("--verbose", "-v", count=True, help="Increase verbosity.")

quiet_opt = click.option("--quiet", "-q", count=True, help="Decrease verbosity.")

opt_cli = click.option(
    "--opt",
    multiple=True,
    callback=cli_utils.parse_opt,
    help="Method specific keyword arguments, see the method documentation "
    "of the specific model for more information about the arguments.",
)

data_opt = click.option(
    "-d",
    "--data",
    multiple=True,
    help="Path to local yaml data catalog file OR name of predefined data catalog.",
)

deltares_data_opt = click.option(
    "--dd",
    "--deltares-data",
    is_flag=True,
    default=False,
    help='Flag: Shortcut to add the "deltares_data" catalog',
)

overwrite_opt = click.option(
    "--fo",
    "--force-overwrite",
    is_flag=True,
    default=False,
    help="Flag: If provided overwrite existing model files",
)

cache_opt = click.option(
    "--cache",
    is_flag=True,
    default=False,
    help="Flag: If provided cache tiled rasterdatasets",
)

export_config_opt = click.option(
    "-f",
    "--export-config",
    callback=cli_utils.parse_export_config_yaml,
    help="read options from a config file for exporting. options from CLI will "
    "override these options",
)

## MAIN


@click.group()
@click.version_option(__version__, message="hydroMT version: %(version)s")
@click.option(
    "--models",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print availabe model plugins and exit.",
    callback=print_models,
)
@click.pass_context
def main(ctx, models):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}


## BUILD
@main.command(short_help="Build models")
@click.argument(
    "MODEL",
    type=str,
)
@arg_root
@opt_cli
@opt_config
@region_opt
@data_opt
@deltares_data_opt
@overwrite_opt
@cache_opt
@verbose_opt
@quiet_opt
@click.pass_context
def build(
    ctx,
    model,
    model_root,
    opt,
    config,
    region,
    data,
    dd,
    fo,
    cache,
    verbose,
    quiet,
):
    """Build models from scratch.

    Example usage:
    --------------

    To build a wflow model for a subbasin using a point coordinates snapped to cells
    with upstream area >= 50 km2
    hydromt build wflow /path/to/model_root -i /path/to/wflow_config.ini  -r "{'subbasin': [-7.24, 62.09], 'uparea': 50}" -d deltares_data -d /path/to/data_catalog.yml -v

    To build a sfincs model based on a bbox
    hydromt build sfincs /path/to/model_root  -i /path/to/sfincs_config.ini  -r "{'bbox': [4.6891,52.9750,4.9576,53.1994]}"  -d /path/to/data_catalog.yml -v

    """  # noqa: E501
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "build", join(model_root, "hydromt.log"), log_level=log_level, append=False
    )
    logger.info(f"Building instance of {model} model at {model_root}.")
    logger.info("User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt)
    kwargs = opt.pop("global", {})
    # Set region to None if empty string json
    if len(region) == 0:
        region = None
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        # initialize model and create folder structure
        mode = "w+" if fo else "w"
        mod = MODELS.load(model)(
            root=model_root,
            mode=mode,
            logger=logger,
            data_libs=data_libs,
            **kwargs,
        )
        mod.data_catalog.cache = cache
        # build model
        mod.build(region, opt=opt)
    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


## UPDATE
@main.command(
    short_help="Update models",
)
@click.argument(
    "MODEL",
    type=str,
)
@arg_root
@click.option(
    "-o",
    "--model-out",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
    help="Output model folder. Maps in MODEL_ROOT are overwritten if left empty.",
    default=None,
    callback=lambda c, p, v: v if v else c.params["model_root"],
)
@opt_config
@click.option(
    "-c",
    "--components",
    multiple=True,
    help="Model methods from ini file to run",
)
@opt_cli
@data_opt
@deltares_data_opt
@overwrite_opt
@cache_opt
@quiet_opt
@verbose_opt
@click.pass_context
def update(
    ctx,
    model,
    model_root,
    model_out,
    config,
    components,
    opt,
    data,
    dd,
    fo,
    cache,
    verbose,
    quiet,
):
    """Update a specific component of a model.

    Set an output directory to copy the edited model to a new folder, otherwise maps
    are overwritten.

    Example usage:
    --------------

    Update (overwrite!) landuse-landcover based maps in a Wflow model:
    hydromt update wflow /path/to/model_root -c setup_lulcmaps --opt lulc_fn=vito -d /path/to/data_catalog.yml -v

    Update Wflow model components outlined in an .ini configuration file and
    write the model to a directory:
    hydromt update wflow /path/to/model_root  -o /path/to/model_out  -i /path/to/wflow_config.ini  -d /path/to/data_catalog.yml -v
    """  # noqa: E501
    # logger
    mode = "r+" if model_root == model_out else "r"
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog("update", join(model_out, "hydromt.log"), log_level=log_level)
    logger.info(f"Updating {model} model at {model_root} ({mode}).")
    logger.info(f"Output dir: {model_out}")
    # parse settings
    if len(components) == 1 and not isinstance(opt.get(components[0]), dict):
        opt = {components[0]: opt}
    logger.info("User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt)
    kwargs = opt.pop("global", {})
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        # initialize model and create folder structure
        mod = MODELS.load(model)(
            root=model_root,
            mode=mode,
            data_libs=data_libs,
            logger=logger,
            **kwargs,
        )
        mod.data_catalog.cache = cache
        # keep only components + setup_config
        if len(components) > 0:
            opt0 = opt.get("setup_config", {})
            opt = {c: opt.get(c, {}) for c in components}
            opt.update({"setup_config": opt0})
        mod.update(model_out=model_out, opt=opt, forceful_overwrite=fo)
    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


@main.command(
    short_help="Validate config files are correct",
)
@data_opt
@deltares_data_opt
@quiet_opt
@verbose_opt
@click.pass_context
def check(
    ctx,
    data,
    dd,
    quiet: int,
    verbose: int,
):
    """Verify that provided data catalog files are in the correct format.

    Example usage:
    --------------

    hydromt check -d /path/to/data_catalog.yml

    """  # noqa: E501
    # logger
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog("check", join(".", "hydromt.log"), log_level=log_level)
    logger.info(f"Output dir: {export_dest_path}")
    try:
        for cat_path in data:
            logger.info(f"Validating catalog at {cat_path}")
            DataCatalogValidator.from_yml(cat_path)
            logger.info("Catalog is valid!")

    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


## Export
@main.command(
    short_help="Export data",
)
@click.option(
    "-t",
    "--target",
)
@region_opt
@export_dest_path
@export_config_opt
@data_opt
@deltares_data_opt
@overwrite_opt
@quiet_opt
@verbose_opt
@click.pass_context
def export(
    ctx: click.Context,
    export_dest_path: Path,
    target: Optional[Union[str, Path]],
    export_config: Optional[ExportConfigDict],
    region: Optional[Dict[Any, Any]],
    data: Optional[List[Path]],
    dd: bool,
    fo: bool,
    quiet: int,
    verbose: int,
):
    """Export the data from a catalog.

    Example usage:
    --------------

    export the data of in a single source, in a pertcular region
    hydromt export -r "{'subbasin': [-7.24, 62.09], 'uparea': 50}" -t era5_hourly -d ../hydromt/data/catalogs/artifact_data.yml .

    export all data of in a single source
    hydromt export --dd -t era5_hourly .

    export data as detailed in an export config yaml file
    hydromt export -f /path/to/export_config.yaml .
    """  # noqa: E501
    # logger
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "export", join(export_dest_path, "hydromt.log"), log_level=log_level
    )
    logger.info(f"Output dir: {export_dest_path}")

    if data:
        data_libs = list(data)  # add data catalogs from cli
    else:
        data_libs = []

    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!

    if export_config:
        args = export_config.pop("args", {})
        if "catalog" in args.keys():
            data_libs = data_libs + args.pop("catalog")
        time_tuple = args.pop("time_tuple", None)
        region = region or args.pop("region", None)
        if isinstance(region, str):
            region = json_decode(region)
    else:
        time_tuple = None
        region = None

    if target:
        export_targets = [{"source": target}]
    elif export_config:
        export_targets = export_config["sources"]
    else:
        export_targets = None

    try:
        data_catalog = DataCatalog(data_libs=data_libs)
        data_catalog.export_data(
            export_dest_path,
            source_names=export_targets,
            time_tuple=time_tuple,
        )

    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


## CLIP


@main.command(short_help="Clip models.")
@click.argument(
    "MODEL",
    type=str,
)
@arg_root
@click.argument(
    "MODEL_DESTINATION",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)
@click.argument(
    "REGION",
    type=str,
    callback=cli_utils.parse_json,
)
@quiet_opt
@verbose_opt
@click.pass_context
def clip(ctx, model, model_root, model_destination, region, quiet, verbose):
    """Create a new model based on clipped region of an existing model.

    If the existing model contains forcing, they will also be clipped to the new model.

    For options to build wflow models see:

    Example usage to clip a wflow model for a subbasin derived from point coordinates
    snapped to cells with upstream area >= 50 km2
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'subbasin': [-7.24, 62.09], 'wflow_uparea': 50}"

    Example usage basin based on ID from model_root basins map
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'basin': 1}"

    Example usage basins whose outlets are inside a geometry
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'outlet': 'geometry.geojson'}"

    All available option in the clip_staticmaps function help.

    """  # noqa: E501
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "clip", join(model_destination, "hydromt-clip.log"), log_level=log_level
    )
    logger.info(f"Clipping instance of {model} model.")
    logger.info(f"Region: {region}")

    if model != "wflow":
        raise NotImplementedError("Clip function only implemented for wflow model.")
    try:
        mod = MODELS.load(model)(root=model_root, mode="r", logger=logger)
        logger.info("Reading model to clip")
        mod.read()
        mod.set_root(model_destination, mode="w")
        logger.info("Clipping staticmaps")
        mod.clip_grid(region)
        logger.info("Clipping forcing")
        mod.clip_forcing()
        logger.info("Writting clipped model")
        mod.write()
    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
