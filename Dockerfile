
ARG OPT_DEPS="full"

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
RUN python3 make_env.py ${OPT_DEPS} \
 && micromamba env create -f environment.yml -y \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete

FROM  mambaorg/micromamba:1.4-jammy AS full
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
 && ls /opt/conda \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete


## Actually install hydromt
FROM  mambaorg/micromamba:1.4-jammy AS test
ENV HOME=/home/hydromt
WORKDIR ${HOME}
COPY --from=env /opt /opt/
COPY pyproject.toml README.rst ${HOME}/
COPY tests/ ${HOME}/tests
COPY hydromt/ ${HOME}/hydromt
COPY data/ ${HOME}/data
RUN micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete

## Actually install hydromt
FROM  mambaorg/micromamba:1.4-jammy AS jupyter
ENV HOME=/home/hydromt
WORKDIR ${HOME}
COPY --from=env /opt /opt/
COPY pyproject.toml README.rst ${HOME}/
COPY hydromt/ ${HOME}/hydromt
COPY data/ ${HOME}/data
COPY examples/ ${HOME}/examples
RUN micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete



FROM  mambaorg/micromamba:1.4-jammy AS cli
USER root
ENV HOME=/home/hydromt
WORKDIR ${HOME}
COPY --from=env /opt /opt/
COPY pyproject.toml README.rst ${HOME}/
COPY hydromt/ ${HOME}/hydromt
RUN micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && ls /opt/conda \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete
ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["hydromt", "--models"]
