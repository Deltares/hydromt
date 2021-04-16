# -*- coding: utf-8 -*-
"""command line interface for hydromt models"""

import click
import os
from os.path import join
import glob
import logging
import sys
import inspect

### Uncomment the following lines for building exe
# import sys
# exepath = sys.prefix
# import pyproj
# pyproj_datadir = join(exepath, "proj-data")
# pyproj.datadir.set_data_dir(pyproj_datadir)
###

from . import cli_utils
from .. import config, log, data_adapter
from ..models import MODELS  # global var
from .. import __version__

logger = logging.getLogger(__name__)


## common arguments / options
opt_config = click.option(
    "-i",
    "--config",
    type=click.Path(resolve_path=True),
    help="Path to hydroMT configuration file, see https://deltares.gitlab.io/wflow/hydromt/models/ for the model specific implementation.",
)
arg_root = click.argument(
    "MODEL_ROOT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)
arg_baseroot = click.argument(
    "BASEMODEL_ROOT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)

verbose_opt = click.option("--verbose", "-v", count=True, help="Increase verbosity.")

quiet_opt = click.option("--quiet", "-q", count=True, help="Decrease verbosity.")

opt_cli = click.option(
    "--opt",
    multiple=True,
    callback=cli_utils.parse_opt,
    help="Component specific keyword arguments, see the setup_<component> method "
    "of the specific model for more information about the arguments.",
)

data_opt = click.option(
    "-d",
    "--data",
    multiple=True,
    type=click.Path(resolve_path=True, file_okay=True),
    help="File path to yml data sources file. See https://deltares.gitlab.io/wflow/hydromt/data/index.html for required yml file format.",
)

## MAIN


@click.group()
@click.version_option(__version__)
@click.pass_context
def main(ctx):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}
    # ctx.obj["log_level"] = max(10, 30 - 10 * (verbose - quiet))
    # logging.basicConfig(stream=sys.stderr, level=ctx.obj["log_level"])


## BUILD


@main.command(short_help="Build models")
@click.argument(
    "MODEL",
    type=click.Choice(list(MODELS.keys())),
)
@arg_root
@click.argument(
    "REGION",
    type=str,
    callback=cli_utils.parse_json,
)
@click.option(
    "-r",
    "--res",
    type=float,
    default=None,
    help=f"Model resolution in model src.",
)
@click.option("--build-base/--build-all", default=False, help="Deprecated!")
@opt_cli
@opt_config
@data_opt
@quiet_opt
@verbose_opt
@click.pass_context
def build(
    ctx, model, model_root, region, res, build_base, opt, config, data, verbose, quiet
):
    """Build models from source data.

    \b
    Example usage:
    --------------

    \b
    To build a wflow model for a subbasin using and point coordinates snapped to cells with stream order >= 4
    hydromt build wflow /path/to/model_root "{'subbasin': [-7.24, 62.09], 'strord': 4}" -i /path/to/wflow_config.ini

    \b
    To build a wflow model based on basin ID
    hydromt build wflow /path/to/model_root "{'basin': 230001006}"

    \b
    To build a sfincs model based on a bbox (for Texel)
    hydromt build sfincs /path/to/model_root "{'bbox': [4.6891,52.9750,4.9576,53.1994]}"

    """
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "build", join(model_root, "hydromt.log"), log_level=log_level, append=False
    )
    logger.info(f"Building instance of {model} model at {model_root}.")
    if build_base:
        DeprecationWarning(
            'The "build-base" flag has been deprecated, modify the ini file instead.'
        )
    if len(data) > 0:
        logger.info(f"Additional data sources: {data}")
    logger.info(f"User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt, logger=logger)
    kwargs = opt.pop("global", {})
    data_libs = data + tuple(kwargs.pop("data_libs", []))
    try:
        # initialize model and create folder structure
        mod = MODELS.get(model)(
            root=model_root,
            data_libs=data_libs,
            mode="w",
            logger=logger,
            **kwargs,
        )
        # build model
        mod.build(region, res, opt=opt)
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
    type=click.Choice(list(MODELS.keys())),
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
@click.option(
    "-c",
    "--components",
    multiple=True,
    help="Model components from ini file to run",
)
@opt_cli
@opt_config
@data_opt
@quiet_opt
@verbose_opt
@click.pass_context
def update(
    ctx, model, model_root, model_out, components, opt, data, config, verbose, quiet
):
    """Update a specific component of a model.
    Set an output directory to copy the edited model to a new folder, otherwise maps
    are overwritten.

    \b
    Example usage:
    --------------

    \b
    Update (overwrite) landuse-landcover maps in a wflow model
    hydromt update wflow /path/to/model_root -c setup_lulcmaps --opt source_name=vito

    \b
    Update reservoir maps based on default settings in a wflow model and write to new directory
    hydromt update wflow /path/to/model_root -o /path/to/model_out -c setup_reservoirs
    """
    # logger
    mode = "r+" if model_root == model_out else "r"
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog("update", join(model_out, "hydromt.log"), log_level=log_level)
    logger.info(f"Updating {model} model at {model_root} ({mode}).")
    logger.info(f"Output dir: {model_out}")
    # parse settings
    if len(data) > 0:
        logger.info(f"Additional data sources: {data}")
    if len(components) == 1 and not isinstance(opt.get(components[0]), dict):
        opt = {components[0]: opt}
    logger.info(f"User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt, logger=logger)
    kwargs = opt.pop("global", {})
    data_libs = data + tuple(kwargs.pop("data_libs", []))
    try:
        # initialize model and create folder structure
        mod = MODELS.get(model)(
            root=model_root,
            data_libs=data_libs,  #  from data/data_sources.yml
            mode=mode,
            logger=logger,
            **kwargs,
        )
        # keep only components + setup_config
        if len(components) > 0:
            opt0 = opt.get("setup_config", {})
            opt = {c: opt.get(c, {}) for c in components}
            opt.update({"setup_config": opt0})
        mod.update(model_out=model_out, opt=opt)
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
    type=click.Choice(list(MODELS.keys())),
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

    \b
    Example usage to clip a wflow model for a subbasin derived from point coordinates
    snapped to cells with stream order >= 4
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'subbasin': [-7.24, 62.09], 'wflow_streamorder': 4}"

    \b
    Example usage basin based on ID from model_root basins map
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'basin': 1}"

    \b
    Example usage basins whose outlets are inside a geometry
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'outlet': 'geometry.geojson'}"

    All available option in the clip_staticmaps function help.

    """
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "clip", join(model_destination, "hydromt-clip.log"), log_level=log_level
    )
    logger.info(f"Clipping instance of {model} model.")
    logger.info(f"Region: {region}")

    if model != "wflow":
        raise NotImplementedError("Clip function only implemented for wflow model.")
    try:
        mod = MODELS.get(model)(root=model_root, mode="r", logger=logger)
        logger.info("Reading model to clip")
        mod.read()
        logger.info("Clipping staticmaps")
        mod.clip_staticmaps(region)
        logger.info("Clipping forcing")
        mod.clip_forcing(model_destination)
        logger.info("Writting clipped model")
        mod.set_root(model_destination, mode="w")
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
