FROM continuumio/miniconda3:23.3.1-0 as base

RUN apt-get update \
 && apt-get install -y --fix-missing --no-install-recommends libgdal-dev gcc lzma-dev python3-dev python3-pip \
 && python3 -m pip install tomli

RUN groupadd -r hydromt \
 && useradd -m -r -g hydromt hydromt

ENV HOME /home/hydromt
ENV NUMBA_CACHE_DIR=${HOME}/.cahce/numba
ENV USE_PYGEOS=0
RUN chown -R hydromt ${HOME}

COPY hydromt ${HOME}/hydromt
COPY tests ${HOME}/tests
COPY data ${HOME}/data
COPY README.rst ${HOME}
COPY pyproject.toml ${HOME}/pyproject.toml
COPY make_env.py ${HOME}/make_env.py
WORKDIR ${HOME}
RUN python make_env.py full


RUN conda env create -f environment.yml \
 && conda run -n hydromt pip install .

FROM base as cli
ENTRYPOINT ["conda","run","-n", "hydromt", "hydromt"]
CMD ["--models"]

 FROM base as test
 CMD ["conda", "run","-n","hydromt" , "pytest"]
