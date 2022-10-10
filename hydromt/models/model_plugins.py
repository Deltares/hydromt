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
    eps : dict
        Entrypoints dict
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

    eps = {}
    for name, ep in group.items():
        logger.debug(
            f"Discovered model plugin '{name} = {ep.module_name}.{ep.object_name}' ({ep.distro.version})"
        )
        eps[ep.name] = ep
    return eps


def load(ep, module=None, logger=logger):
    """Load entrypoint and return plugin model class

    Parameters
    ----------
    ep : entrypoint
        discovered entrypoint

    Returns
    -------
    model_class : Model
        plugin model class
    """
    try:
        # plugins[ep.name] = ep.load()
        model_class = ep.load()
        if module is not None:
            setattr(module, model_class.__name__, model_class)
        logger.debug(
            f"Loaded model plugin '{ep.name} = {ep.module_name}.{ep.object_name}' ({ep.distro.version})"
        )
        return model_class
    except (ModuleNotFoundError, AttributeError) as err:
        logger.exception(f"Error while loading entrypoint {ep.name}: {str(err)}")
        return None
