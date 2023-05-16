FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.4 as base


RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends libgdal-dev gcc lzma-dev python3-dev python3-pip && python3 -m pip install --no-cache-dir notebook jupyterlab

RUN groupadd -r hydromt && useradd -m -r -g hydromt hydromt --uid 1000
ENV HOME /home/hydromt
RUN chown -R hydromt ${HOME}

COPY hydromt ${HOME}/hydromt
COPY tests ${HOME}/tests
COPY data ${HOME}/data
COPY README.rst ${HOME}
COPY pyproject.toml ${HOME}/pyproject.toml
WORKDIR ${HOME}

USER hydromt
# CMD ["/bin/bash"]
RUN pip install .

# FROM base as cli
# ENTRYPOINT ["hydromt"]
# CMD ["--models"]

FROM base as test
RUN export PATH="/home/hydromt/.local/bin:$PATH" && python3 -m pip install pytest pytest-mock responses
CMD ["python3" , "-m", "pytest"]

#FROM base as docs

# FROM base as binder
# ENTRYPOINT ["/bin/bash"]



