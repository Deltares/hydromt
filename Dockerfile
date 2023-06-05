FROM mambaorg/micromamba:1.4-bullseye-slim as env

USER root

RUN apt-get update \
 && apt-get install -y --fix-missing --no-install-recommends libgdal-dev gcc lzma-dev python3-dev python3-pip \
 && python3 -m pip install tomli

RUN groupadd -r hydromt \
 && useradd -m -r -g hydromt hydromt

ENV HOME /home/hydromt
ENV NUMBA_CACHE_DIR=${HOME}/.cahce/numba
ENV USE_PYGEOS=0
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${HOME}
COPY pyproject.toml ${HOME}/pyproject.toml
COPY make_env.py ${HOME}/make_env.py
RUN python3 make_env.py full
RUN chown -R hydromt ${HOME}
USER hydromt
RUN micromamba env create -f environment.yml -y

FROM env as base
COPY data ${HOME}/data
COPY README.rst ${HOME}
COPY tests ${HOME}/tests
COPY hydromt ${HOME}/hydromt
RUN micromamba run -n hydromt pip install .

ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["hydromt", "--models"]
