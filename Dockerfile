FROM continuumio/miniconda3:23.3.1-0 as base

RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends libgdal-dev gcc lzma-dev python3-dev python3-pip && python3 -m pip install toml-cli

RUN groupadd -r hydromt && useradd -m -r -g hydromt hydromt 
ENV HOME /home/hydromt
ENV NUMBA_CACHE_DIR=${HOME}/.cahce/numba
ENV USE_PYGEOS=0
RUN chown -R hydromt ${HOME}

COPY hydromt ${HOME}/hydromt
COPY tests ${HOME}/tests
COPY data ${HOME}/data
COPY README.rst ${HOME}
COPY pyproject.toml ${HOME}/pyproject.toml
WORKDIR ${HOME}


RUN conda create -n hydromt -c conda-forge $(toml get --toml-path=pyproject.toml "project.dependencies" | sed "s/\[//g;s/\]//g;s/,//g") $(toml get --toml-path=pyproject.toml "project.optional-dependencies.all"  | sed "s/\[//g;s/\]//g;s/,//g;s/pyet//g" | sed "s/''//g") hydromt -y 
# USER hydromt

FROM base as cli
USER hydromt
ENTRYPOINT ["conda","run","-n", "hydromt", "hydromt"]
CMD ["--models"]

 FROM base as test

 RUN conda install -n hydromt -c conda-forge $(toml get --toml-path=pyproject.toml "project.optional-dependencies.test" | sed "s/\[//g;s/\]//g;s/,//g") -y
 USER hydromt
# CMD ["/bin/bash"]
 CMD ["conda", "run","-n","hydromt" , "pytest"]

#FROM base as docs

# FROM base as binder
# python3 -m pip install --no-cache-dir notebook jupyterlab 
# ENTRYPOINT ["/bin/bash"]



