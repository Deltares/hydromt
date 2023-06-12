## Set up the base environment to install hydromt into
FROM mambaorg/micromamba:1.4-bullseye-slim AS env
WORKDIR /home/mambauser/
RUN micromamba install -c conda-forge -n base tomli -y
COPY pyproject.toml make_env.py /home/mambauser/
RUN micromamba run -n base python make_env.py full \
 && micromamba env create -f environment.yml -y \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete

## Actually install hydromt
FROM  mambaorg/micromamba:1.4-jammy AS dev
ENV HOME=/home/mambauser
WORKDIR ${HOME}
COPY --from=env /opt /opt/
COPY pyproject.toml README.rst ${HOME}/
COPY tests ${HOME}/tests
COPY hydromt ${HOME}/hydromt
COPY data ${HOME}/data
COPY examples ${HOME}/examples
RUN micromamba run -n hydromt pip install . \
 && micromamba clean -ayf \
 && find /opt/conda/ -follow -type f -name '*.a' -delete \
 && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
 && find /opt/conda/ -follow -type f -name '*.js.map' -delete


FROM  mambaorg/micromamba:1.4-jammy AS cli
COPY --from=dev /opt /opt/
COPY --from=dev /home/mambauser /home/mambauser/
ENV NUMBA_CACHE_DIR=/home/mambauser/.cahce/numba \
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
USER mambauser
WORKDIR /home/mambauser
ENTRYPOINT ["micromamba","run","-n", "hydromt"]
CMD ["hydromt", "--models"]
