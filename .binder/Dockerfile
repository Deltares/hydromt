# currently we haven't published an official hydromt image yet, so for now
# I import from an export I've put on my own account. this should be moved
# to the official image as soon as it's published.
FROM savente/hydromt:slim as binder
# Binder hard requires all of these steps when they build the imae
# therefore these steps aren't taken sooner

ENV HOME=/home/mambauser \
    NUMBA_CACHE_DIR=${HOME}/.cahce/numba\
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
WORKDIR ${HOME}
ENTRYPOINT ["micromamba","run","-n","hydromt"]
CMD ["jupyter","notebook","--port=8888","--ip=0.0.0.0"]
