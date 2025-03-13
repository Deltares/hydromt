FROM deltares/hydromt:slim AS binder
# Binder hard requires all of these steps when they build the image
# therefore these steps aren't taken sooner

ENV HOME=/home/deltares \
    NUMBA_CACHE_DIR=${HOME}/.cahce/numba\
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
WORKDIR ${HOME}
ENTRYPOINT ["pixi", "run", "-e", "slim-latest"]
CMD ["pixi", "run", "-e", "slim-latest" ,"jupyter","notebook","--port=8888","--ip=0.0.0.0"]
