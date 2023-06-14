
## Set up the base environment to install hydromt into
FROM mambaorg/micromamba:1.4-alpine AS min
ARG OPT_DEPS=""
ENV HOME=/home/mambauser \
    OPT_DEPS=$OPT_DEPS
WORKDIR ${HOME}

COPY min-environment.yml pyproject.toml README.rst ${HOME}/
COPY hydromt/ ${HOME}/hydromt
RUN micromamba env create -f min-environment.yml -y \
 && micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
 && rm min-environment.yml 
ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["hydromt", "--models"]


# FROM  mambaorg/micromamba:1.4-alpine AS jupyter
# ARG OPT_DEPS=""
# ENV HOME=/home/mambauser \
#     OPT_DEPS=[$OPT_DEPS]
# WORKDIR ${HOME}
# COPY --from=min /opt /opt/
# COPY --from=min ${HOME} ${HOME}/
# COPY data/ ${HOME}/data
# COPY examples/ ${HOME}/examples
# RUN micromamba run -n hydromt pip install ".$OPT_DEPS" \
#  && micromamba clean -ayf \
#  && find /opt/conda/ -follow -type f -name '*.a' -delete \
#  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
#  && find /opt/conda/ -follow -type f -name '*.js.map' -delete\
#  && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

# FROM  mambaorg/micromamba:1.4-alpine AS test
# ARG OPT_DEPS=""
# ENV HOME=/home/hydromt \
#     OPT_DEPS=$OPT_DEPS
# WORKDIR ${HOME}
# COPY --from=min /opt /opt/
# COPY --from=min ${HOME} ${HOME}/
# COPY tests/ ${HOME}/tests
# COPY data/ ${HOME}/data
# RUN micromamba run -n hydromt pip install ".[$OPT_DEPS]" \
#  && micromamba clean -ayf \
#  && find /opt/conda/ -follow -type f -name '*.a' -delete \
#  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
#  && find /opt/conda/ -follow -type f -name '*.js.map' -delete

# FROM  mambaorg/micromamba:1.4-alpine AS full
# ARG OPT_DEPS=""
# USER root
# ENV HOME=/home/hydromt \
#     OPT_DEPS="[$OPT_DEPS]"
# WORKDIR ${HOME}
# COPY --from=env /opt /opt/
# COPY pyproject.toml README.rst ${HOME}/
# COPY tests/ ${HOME}/tests
# COPY hydromt/ ${HOME}/hydromt
# COPY data/ ${HOME}/data
# COPY examples/ ${HOME}/examples
# RUN micromamba run -n hydromt pip install ".$OPT_DEPS" \
#  && micromamba clean -ayf \
#  && ls /opt/conda \
#  && find /opt/conda/ -follow -type f -name '*.a' -delete \
#  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
#  && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
#  && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete



