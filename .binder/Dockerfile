ARG NB_USER=hydromt
ARG NB_UID=1000

FROM savente/hydromt:current as binder
# Binder hard requires all of these steps when they build the imae
# therefore these steps aren't taken sooner
USER root
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
