import itertools
import entrypoints
import logging

logger = logging.getLogger(__name__)


def discover(path=None, logger=logger):
    """Discover hydromt models:
    - Find 'hydromt.models' entrypoints provided by any Python packages in the
      environment.

    Parameters
    ----------
    path : str or None
        Default is ``sys.path``.

    Returns
    -------
    drivers : dict
        Name mapped to model class.
    """
    # Discover drivers via entrypoints.
    group = entrypoints.get_group_named("hydromt.models", path=path)
    group_all = entrypoints.get_group_all("hydromt.models", path=path)
    if len(group_all) != len(group):
        # There are some name collisions. Let's go digging for them.
        for name, matches in itertools.groupby(group_all, lambda ep: ep.name):
            matches = list(matches)
            if len(matches) != 1:
                winner = group[name]
                logger.debug(
                    f"There are {len(matches)} 'hydromt.models' entrypoints for the name {name}: {matches}."
                    f"The match {winner} is selected."
                )

    plugins = {}
    for name, ep in group.items():
        logger.debug(
            f"Discovered model plugin '{name} = {ep.module_name}.{ep.object_name}'"
        )

        try:
            plugins[ep.name] = ep.load()
        except (ModuleNotFoundError, AttributeError) as err:
            logger.exception(f"Error while loading entrypoint {name}: {str(err)}")
            continue
        logger.debug(
            f"Loaded model plugin '{name} = {ep.module_name}.{ep.object_name}'"
        )
    return plugins
