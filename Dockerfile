FROM  mambaorg/micromamba:1.4-alpine AS min
ENV HOME=/home/mambauser
WORKDIR ${HOME}
USER mambauser
COPY min-environment.yml pyproject.toml README.rst ${HOME}/
RUN micromamba create -f min-environment.yml -y --no-pyc \
 && micromamba clean -ayf \
 && rm -rf ${HOME}/.cache \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete  \
 && rm min-environment.yml
COPY data/ ${HOME}/data
COPY examples/ ${HOME}/examples
COPY tests/ ${HOME}/tests
COPY hydromt/ ${HOME}/hydromt
RUN micromamba run -n hydromt pip install . --no-cache-dir --no-compile --disable-pip-version-check --no-deps\
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete
 ENTRYPOINT [ "micromamba", "run", "-n", "hydromt" ]
 CMD ["hydromt","--models"]

FROM  mambaorg/micromamba:1.4-alpine AS full
ENV HOME=/home/mambauser
WORKDIR ${HOME}
USER mambauser
COPY full-environment.yml pyproject.toml README.rst ${HOME}/
RUN micromamba create -f full-environment.yml -y --no-pyc \
 && micromamba clean -ayf \
 && rm -rf ${HOME}/.cache \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete  \
 && rm full-environment.yml
COPY data/ ${HOME}/data
COPY examples/ ${HOME}/examples
COPY tests/ ${HOME}/tests
COPY hydromt/ ${HOME}/hydromt
RUN micromamba run -n hydromt pip install . --no-cache-dir --no-compile --disable-pip-version-check --no-deps\
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete
 ENTRYPOINT [ "micromamba", "run", "-n", "hydromt" ]
 CMD ["hydromt","--models"]

FROM  mambaorg/micromamba:1.4-alpine AS slim
ENV HOME=/home/mambauser
WORKDIR ${HOME}
USER mambauser
COPY slim-environment.yml pyproject.toml README.rst ${HOME}/
RUN micromamba create -f slim-environment.yml -y --no-pyc \
 && rm -rf ${HOME}/.cache \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete  \
 && rm slim-environment.yml
COPY data/ ${HOME}/data
COPY examples/ ${HOME}/examples
COPY hydromt/ ${HOME}/hydromt
RUN micromamba run -n hydromt pip install . --no-cache-dir --no-compile --disable-pip-version-check --no-deps\
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete
 ENTRYPOINT [ "micromamba", "run", "-n", "hydromt" ]
 CMD ["hydromt","--models"]
