FROM ghcr.io/prefix-dev/pixi:0.61.0-focal
RUN groupadd deltares && useradd deltares --uid 1000 -m -g deltares

USER deltares
WORKDIR /home/deltares

ENV PATH=/home/deltares/.pixi/bin:$PATH
ENV HOME=/home/deltares/

COPY pixi.lock pyproject.toml README.rst ./
COPY ./hydromt ./hydromt
COPY ./docs ./docs
COPY examples ./examples

RUN pixi install --locked -e slim-latest \
    && rm -rf .cache \
    && find .pixi -type f -name "*.pyc" -delete

ENV HOME=/home/deltares \
    NUMBA_CACHE_DIR=${HOME}/.cache/numba \
    USE_PYGEOS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1
WORKDIR ${HOME}

ENTRYPOINT ["pixi", "run", "-e", "slim-latest"]
CMD ["pixi", "run", "-e", "slim-latest" , "jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0"]
