
## Set up the base environment to install hydromt into
FROM mambaorg/micromamba:1.4-bullseye-slim AS env
USER root
RUN apt-get update \
 && apt-get install -y --fix-missing --no-install-recommends  python3-dev python3-pip \
 && apt-get autoremove \
 && apt-get clean \
 && python3 -m pip install tomli

ENV HOME=/home/hydromt
WORKDIR ${HOME}

COPY pyproject.toml make_env.py ${HOME}/
RUN python3 make_env.py full \
 && micromamba env create -f environment.yml -y \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete

## Actually install hydromt
FROM  mambaorg/micromamba:1.4-jammy AS dev
USER root
ENV HOME=/home/hydromt
WORKDIR ${HOME}
COPY --from=env /opt /opt/
COPY pyproject.toml README.rst ${HOME}/
COPY tests/ ${HOME}/tests
COPY hydromt/ ${HOME}/hydromt
COPY data/ ${HOME}/data
COPY examples/ ${HOME}/examples
RUN micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete


FROM  mambaorg/micromamba:1.4-jammy AS cli
ARG NB_USER=hydromt
ARG NB_UID=1000
COPY --from=dev /opt /opt/
COPY --from=dev /home/hydromt /home/hydromt/
USER root
ENV HOME=/home/hydromt \
    NUMBA_CACHE_DIR=${HOME}/.cahce/numba\
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
WORKDIR ${HOME}
RUN groupadd -r ${NB_USER}\
 && usermod -l ${NB_USER} -d ${HOME} -u ${NB_UID} mambauser \
 && chown -R ${NB_USER} ${HOME}
USER ${NB_USER}
ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["hydromt", "--models"]

FROM cli AS jupyter
CMD ["micromamba","run","-n", "hydromt", "jupyter", "lab", "--ip=0.0.0.0"]
