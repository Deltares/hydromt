# -*- coding: utf-8 -*-
"""command line interface for hydromt models."""

import logging
from ast import literal_eval
from datetime import datetime
from json import loads as json_decode
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from pydantic import ValidationError

from hydromt import __version__
from hydromt._io import write_yaml
from hydromt._typing.error import NoDataStrategy
from hydromt._typing.type_def import StrPath
from hydromt._utils import log
from hydromt._validators import Format
from hydromt._validators.data_catalog_v0x import DataCatalogV0Validator
from hydromt._validators.data_catalog_v1x import DataCatalogV1Validator
from hydromt._validators.model_config import HydromtModelSetup
from hydromt.cli import _utils
from hydromt.data_catalog import DataCatalog
from hydromt.plugins import PLUGINS

logger = logging.getLogger(__name__)

HYDROMT_LOG_PATH = "hydromt.log"


def print_available_models(ctx, param, value):
    """Print the available models and exit.

    Parameters
    ----------
    ctx : click.Context
        The Click context object.
    value : bool
        The value of the parameter.
    """
    # param is required by click though we don't use it
    if not value:
        return {}
    click.echo(f"{PLUGINS.model_summary()}")
    ctx.exit()


def print_available_components(ctx, _param, value):
    """Print the available components and exit.

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
    click.echo(PLUGINS.component_summary())
    ctx.exit()


def print_available_plugins(ctx, _param, value):
    """Print the available plugins and exit.

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
    click.echo(f"{PLUGINS.plugin_summary()}")
    ctx.exit()


## common arguments / options
opt_config = click.option(
    "-i",
    "--config",
    type=click.Path(resolve_path=True),
    help="Path to hydroMT configuration file, for the model specific implementation.",
)
req_config = click.option(
    "-i",
    "--config",
    type=click.Path(resolve_path=True),
    help="Path to hydroMT configuration file, for the model specific implementation.",
    required=True,
)
export_dest_path = click.argument(
    "export_dest_path",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)
arg_root = click.argument(
    "MODEL_ROOT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
)

region_opt = click.option(
    "-r",
    "--region",
    type=str,
    default="{}",
    callback=_utils.parse_json,
    help="Set the region for which to build the model,"
    " e.g. {'subbasin': [-7.24, 62.09]}",
)

verbose_opt = click.option("--verbose", "-v", count=True, help="Increase verbosity.")

quiet_opt = click.option("--quiet", "-q", count=True, help="Decrease verbosity.")

data_opt = click.option(
    "-d",
    "--data",
    multiple=True,
    help="Path to local yaml data catalog file OR name of predefined data catalog.",
)

format_option = click.option(
    "--format",
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
error_on_empty = click.option(
    "--error-on-empty",
    is_flag=True,
    default=False,
    help="Flag: Raise an error when attempting to export empty dataset instead of continuing",
)

cache_opt = click.option(
    "--cache",
    is_flag=True,
    default=False,
    help="Flag: If provided cache tiled rasterdatasets",
)

## MAIN


@click.group()
@click.version_option(__version__, message="HydroMT version: %(version)s")
@click.option(
    "--models",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print available model plugins and exit.",
    callback=print_available_models,
)
@click.option(
    "--components",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print available component plugins and exit.",
    callback=print_available_components,
)
@click.option(
    "--plugins",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print available component plugins and exit.",
    callback=print_available_plugins,
)
@click.pass_context
def main(ctx, models, components, plugins):
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
@req_config
@data_opt
@deltares_data_opt
@overwrite_opt
@cache_opt
@verbose_opt
@quiet_opt
@click.pass_context
def build(
    _ctx: click.Context,
    model,
    model_root,
    config,
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

    To build a wflow model:
    hydromt build wflow /path/to/model_root -i /path/to/wflow_config.yml
    -d deltares_data -d /path/to/data_catalog.yml -v

    To build a sfincs model:
    hydromt build sfincs /path/to/model_root  -i /path/to/sfincs_config.yml
    -d /path/to/data_catalog.yml -v
    """  # noqa: E501
    log_level = max(10, 30 - 10 * (verbose - quiet))
    log._setuplog(join(model_root, HYDROMT_LOG_PATH), log_level=log_level, append=False)
    logger.info(f"Building instance of {model} model at {model_root}.")
    logger.info("User settings:")
    opt = _utils.parse_config(config)
    if "steps" not in opt:
        error_msg = (
            f"It seems your workflow file at {config} does not "
            "contain a `steps` section. Perhaps you're using a v0.x format? "
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    kwargs = opt.pop("global", {})
    modeltype = opt.pop("modeltype", model)
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        # initialize model and create folder structure
        mode = "w+" if fo else "w"
        if modeltype not in PLUGINS.model_plugins:
            raise ValueError("Unknown model")
        mod = PLUGINS.model_plugins[modeltype](
            root=model_root,
            mode=mode,
            data_libs=data_libs,
            **kwargs,
        )
        mod.data_catalog.cache = cache
        # build model
        mod.build(steps=opt["steps"])

    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        log._wait_and_remove_file_handlers(logger)  # Release locks on logs


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
@req_config
@data_opt
@deltares_data_opt
@overwrite_opt
@cache_opt
@quiet_opt
@verbose_opt
@click.pass_context
def update(
    _ctx: click.Context,
    model,
    model_root,
    model_out,
    config,
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

    Update Wflow model components outlined in an .yml configuration file and
    write the model to a directory:
    hydromt update wflow /path/to/model_root  -o /path/to/model_out  -i /path/to/wflow_config.yml  -d /path/to/data_catalog.yml -v
    """  # noqa: E501
    # logger
    mode = "r+" if model_root == model_out else "r"
    log_level = max(10, 30 - 10 * (verbose - quiet))
    log._setuplog(join(model_out, HYDROMT_LOG_PATH), log_level=log_level)
    logger.info(f"Updating {model} model at {model_root} ({mode}).")
    logger.info(f"Output dir: {model_out}")
    # parse settings
    logger.info("User settings:")
    opt = _utils.parse_config(config)

    if "steps" not in opt:
        error_msg = (
            f"It seems your workflow file at {config} does not "
            "contain a `steps` section. Perhaps you're using a v0.x format? "
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    kwargs = opt.pop("global", {})
    modeltype = opt.pop("modeltype", model)
    if modeltype not in PLUGINS.model_plugins:
        raise ValueError("Unknown model")
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(data)  # add data catalogs from cli
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs = ["deltares_data"] + data_libs  # prepend!
    try:
        # initialize model and create folder structure
        mod = PLUGINS.model_plugins[modeltype](
            root=model_root,
            mode=mode,
            data_libs=data_libs,
            **kwargs,
        )
        mod.data_catalog.cache = cache
        mod.update(model_out=model_out, steps=opt["steps"], forceful_overwrite=fo)
    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        log._wait_and_remove_file_handlers(logger)  # Release locks on logs


@main.command(
    short_help="Validate config / data catalog / region",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model name, e.g. wflow, sfincs, etc. to validate config file.",
)
@opt_config
@data_opt
@quiet_opt
@verbose_opt
@click.option("--format", type=click.Choice(list(Format)), default=Format.v1)
@click.option("--upgrade", is_flag=True)
@click.pass_context
def check(
    _ctx: click.Context,
    model: Optional[str],
    config,
    data,
    quiet: int,
    verbose: int,
    format: Format,
    upgrade: bool,
):
    """
    Verify that provided data catalog and config files are in the correct format.

    Additionally region bbox and geom can also be validated.

    Example usage:
    --------------

    Check data catalog file:
    hydromt check -d /path/to/data_catalog.yml -v

    Check data catalog and grid_model config file:
    hydromt check -m grid_model -d /path/to/data_catalog.yml -i /path/to/model_config.yml -v

    With region:
    hydromt check -m grid_model -d /path/to/data_catalog.yml -i /path/to/model_config.yml -r '{'bbox': [-1,-1,1,1]}' -v

    """  # noqa: E501
    # logger
    log_level = max(10, 30 - 10 * (verbose - quiet))
    log._setuplog(join(".", HYDROMT_LOG_PATH), log_level=log_level)
    failed = False
    try:
        for cat_path in data:
            logger.info(f"Validating catalog at {cat_path}")
            try:
                if format == Format.v0:
                    v0_catalog = DataCatalogV0Validator.from_yml(cat_path)
                    if upgrade:
                        v1_catalog = DataCatalogV1Validator.from_v0(v0_catalog)

                        p = Path(cat_path)
                        write_yaml(
                            p.with_stem(f"{p.stem}_v1"),
                            v1_catalog.model_dump(
                                exclude_unset=True,
                                exclude_defaults=True,
                                exclude_none=True,
                            ),
                        )

                # if we add more versions don't forget to update this
                # statement here
                else:
                    DataCatalogV1Validator.from_yml(cat_path)

                logger.info(f"Catalog {cat_path} is valid!")
            except ValidationError as e:
                failed = True
                msg = f"Catalog {cat_path} has the following error(s): {e}"
                logger.error(msg)

        if config:
            logger.info(f"Validating config at {config}")
            if format == Format.v0:
                logger.error(
                    "Because of the dynamic nature of workflow files, v0.x files cannot be "
                    "checked by hydromt v1. Skipping..."
                )
            else:
                try:
                    config_dict = _utils.parse_config(config)
                    if model:
                        config_dict["modeltype"] = model

                    if "steps" not in config_dict:
                        raise ValueError(
                            f"No `steps` section in workflow yaml {config}. Perhaps it is a v0.x file?"
                        )

                    HydromtModelSetup(**config_dict)
                    logger.info(f"Workflow yaml {config} is valid!")

                except (ValidationError, ValueError) as e:
                    failed = True
                    msg = f"Workflow yaml {config} has the following error(s): {e}"
                    logger.error(msg)

        if failed:
            # We've already presentend the errors above
            # now we simply raise without extra msg to fail the process
            # so it has the correct exit code
            raise ValueError("hydromt check command failed. See error messages above for the details")

    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        log._wait_and_remove_file_handlers(logger)  # Release locks on logs


## Export
@main.command(
    short_help="Export data",
)
@click.option(
    "-s",
    "--source",
    multiple=True,
    help="Name of the data source to export.",
)
@click.option(
    "-t",
    "--time-range",
    help="Time tuple as a list of two strings, e.g. ['2010-01-01', '2022-12-31']",
)
@click.option(
    "-b",
    "--bbox",
    help="a bbox in EPSG:4236 designating the region of which to export the data",
)
@region_opt
@export_dest_path
@opt_config
@data_opt
@deltares_data_opt
@overwrite_opt
@error_on_empty
@quiet_opt
@verbose_opt
@click.pass_context
def export(
    _ctx: click.Context,
    export_dest_path: Path,
    source: Optional[str],
    time_range: Optional[str],
    bbox: Optional[Tuple[float, float, float, float]],
    config: Optional[Path],
    region: Optional[Dict[Any, Any]],
    data: Optional[List[StrPath]],
    dd: bool,
    fo: bool,
    error_on_empty: bool,
    quiet: int,
    verbose: int,
):
    """Export the data from a catalog.

    Example usage:
    --------------

    export the data of a single data source, in a particular region, for a particular time range
    hydromt export -r "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -s era5_hourly -d ../hydromt/data/catalogs/artifact_data.yml -t '["2010-01-01", "2022-12-31"]' path/to/output_dir

    export a single data source from the deltares data catalog without time/space slicing
    hydromt export -d deltares_data -s era5_hourly path/to/output_dir

    export data as detailed in an export config yaml file
    hydromt export -i /path/to/export_config.yaml path/to/output_dir
    """  # noqa: E501
    # logger
    log_level = max(10, 30 - 10 * (verbose - quiet))
    log._setuplog(join(export_dest_path, HYDROMT_LOG_PATH), log_level=log_level)
    logger.info(f"Output dir: {export_dest_path}")

    if error_on_empty:
        handle_nodata = NoDataStrategy.RAISE
    else:
        handle_nodata = NoDataStrategy.IGNORE

    data_libs: List[StrPath] = list(data) if data else []
    if dd and "deltares_data" not in data_libs:  # deltares_data from cli
        data_libs.insert(0, "deltares_data")

    sources: List[str] = []
    if source:
        if isinstance(source, str):
            sources = [source]
        else:
            sources = list(source)

    # these need to be defined even if config does not exist
    unit_conversion = True
    meta = {}
    append = False

    if config:
        config_dict = _utils.parse_config(config)["export_data"]
        if "data_libs" in config_dict.keys():
            data_libs = data_libs + config_dict.pop("data_libs")
        time_range = config_dict.pop("time_range", None)
        region = region or config_dict.pop("region", None)
        if isinstance(region, str):
            region = json_decode(region)

        sources = sources + config_dict["sources"]

        unit_conversion = config_dict.pop("unit_conversion", True)
        meta = config_dict.pop("meta", {})
        append = config_dict.pop("append", False)

    data_catalog = DataCatalog(data_libs=data_libs)
    _ = data_catalog.sources  # initialise lazy loading

    if time_range:
        if isinstance(time_range, str):
            tup = literal_eval(time_range)
        else:
            tup = time_range
        time_start = datetime.strptime(tup[0], "%Y-%m-%d")
        time_end = datetime.strptime(tup[1], "%Y-%m-%d")
        time_tup = (time_start, time_end)
    else:
        time_tup = None

    if isinstance(bbox, str):
        bbox = literal_eval(bbox)

    try:
        data_catalog.export_data(
            export_dest_path,
            source_names=sources,
            bbox=bbox,
            time_range=time_tup,
            unit_conversion=unit_conversion,
            metadata=meta,
            append=append,
            handle_nodata=handle_nodata,
            force_overwrite=fo,
        )

    except Exception as e:
        logger.exception(e)  # catch and log errors
        raise
    finally:
        log._wait_and_remove_file_handlers(logger)  # Release locks on logs


if __name__ == "__main__":
    main()
