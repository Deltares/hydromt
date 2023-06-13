ARG NB_USER=hydromt
ARG NB_UID=1000

# currently we haven't published an official hydromt image yet, so for now
# I import from an export I've put on my own account. this should be moved
# to the official image as soon as it's published.
FROM savente/hydromt:current as binder
# Binder hard requires all of these steps when they build the imae
# therefore these steps aren't taken sooner
USER root
ENV NB_USER=hydromt \
    NB_UID=1000
RUN mv /home/mambauser /home/${NB_USER} \
 && groupadd -r ${NB_USER}\
 && usermod -l ${NB_USER} -d ${HOME} -u ${NB_UID} mambauser \
 && chown -R ${NB_USER} ${HOME}
USER ${NB_USER}
ENV HOME=/home/${NB_USER} \
    NUMBA_CACHE_DIR=${HOME}/.cahce/numba\
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
WORKDIR ${HOME}
ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["jupyter","notebook","--port=8888","--ip=0.0.0.0"]
