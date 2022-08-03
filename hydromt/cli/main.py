# -*- coding: utf-8 -*-
"""command line interface for hydromt models"""

import click
from os.path import join
import logging
import warnings
import numpy as np

### Uncomment the following lines for building exe
# import sys
# exepath = sys.prefix
# import pyproj
# pyproj_datadir = join(exepath, "proj-data")
# pyproj.datadir.set_data_dir(pyproj_datadir)
###

from . import cli_utils
from .. import log
from ..models import ENTRYPOINTS, model_plugins
from .. import __version__

logger = logging.getLogger(__name__)

_models = list(ENTRYPOINTS.keys())


def print_models(ctx, param, value):
    if not value:
        return {}
    mod_lst = []
    for name, ep in ENTRYPOINTS.items():
        mod_lst.append(f"{name} (v{ep.distro.version})")
    mods = ", ".join(mod_lst)
    click.echo(f"hydroMT model plugins: {mods:s}")
    ctx.exit()


## common arguments / options
opt_config = click.option(
    "-i",
    "--config",
    type=click.Path(resolve_path=True),
    help="Path to hydroMT configuration file, for the model specific implementation.",
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
    help='Shortcut to add the "deltares_data" catalog',
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

    # ctx.obj["log_level"] = max(10, 30 - 10 * (verbose - quiet))
    # logging.basicConfig(stream=sys.stderr, level=ctx.obj["log_level"])


## BUILD


@main.command(short_help="Build models")
@click.argument(
    "MODEL",
    type=str,
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
@deltares_data_opt
@quiet_opt
@verbose_opt
@click.pass_context
def build(
    ctx,
    model,
    model_root,
    region,
    res,
    build_base,
    opt,
    config,
    data,
    dd,
    verbose,
    quiet,
):
    """Build models from scratch.

    \b
    Example usage:
    --------------

    \b
    To build a wflow model for a subbasin using and point coordinates snapped to cells with stream order >= 4
    hydromt build wflow /path/to/model_root "{'subbasin': [-7.24, 62.09], 'strord': 4}" -i /path/to/wflow_config.ini -d deltares_data -d /path/to/data_catalog.yml -v

    \b
    To build a sfincs model based on a bbox
    hydromt build sfincs /path/to/model_root "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -i /path/to/sfincs_config.ini -d /path/to/data_catalog.yml -v

    """
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog(
        "build", join(model_root, "hydromt.log"), log_level=log_level, append=False
    )
    logger.info(f"Building instance of {model} model at {model_root}.")
    if build_base:
        warnings.warn(
            'The "build-base" flag has been deprecated, modify the ini file instead.',
            DeprecationWarning,
        )
    logger.info(f"User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt)
    kwargs = opt.pop("global", {})
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        if model not in _models:
            raise ValueError(f"Model unknown : {model}, select from {_models}")
        # initialize model and create folder structure
        mod = model_plugins.load(ENTRYPOINTS[model], logger=logger)(
            root=model_root,
            mode="w",
            logger=logger,
            data_libs=data_libs,
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
@click.option(
    "-c",
    "--components",
    multiple=True,
    help="Model methods from ini file to run",
)
@opt_cli
@opt_config
@data_opt
@deltares_data_opt
@quiet_opt
@verbose_opt
@click.pass_context
def update(
    ctx, model, model_root, model_out, components, opt, data, dd, config, verbose, quiet
):
    """Update a specific component of a model.
    Set an output directory to copy the edited model to a new folder, otherwise maps
    are overwritten.

    \b
    Example usage:
    --------------

    \b
    Update (overwrite!) landuse-landcover based maps in a Wflow model
    hydromt update wflow /path/to/model_root -c setup_lulcmaps --opt lulc_fn=vito -d /path/to/data_catalog.yml -v

    \b
    Update Wflow model components outlined in an .ini configuration file and write the model to a directory
    hydromt update wflow /path/to/model_root -o /path/to/model_out -i /path/to/wflow_config.ini -d /path/to/data_catalog.yml -v
    """
    # logger
    mode = "r+" if model_root == model_out else "r"
    log_level = max(10, 30 - 10 * (verbose - quiet))
    logger = log.setuplog("update", join(model_out, "hydromt.log"), log_level=log_level)
    logger.info(f"Updating {model} model at {model_root} ({mode}).")
    logger.info(f"Output dir: {model_out}")
    # parse settings
    if len(components) == 1 and not isinstance(opt.get(components[0]), dict):
        opt = {components[0]: opt}
    logger.info(f"User settings:")
    opt = cli_utils.parse_config(config, opt_cli=opt)
    kwargs = opt.pop("global", {})
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        if model not in _models:
            raise ValueError(f"Model unknown : {model}, select from {_models}")
        # initialize model and create folder structure
        mod = model_plugins.load(ENTRYPOINTS[model], logger=logger)(
            root=model_root,
            mode=mode,
            data_libs=data_libs,
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
        if model not in _models:
            raise ValueError(f"Model unknown : {model}, select from {_models}")
        mod = model_plugins.load(ENTRYPOINTS[model], logger=logger)(
            root=model_root, mode="r", logger=logger
        )
        logger.info("Reading model to clip")
        mod.read()
        mod.set_root(model_destination, mode="w")
        logger.info("Clipping staticmaps")
        mod.clip_staticmaps(region)
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
