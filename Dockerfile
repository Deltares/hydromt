ARG BASE_IMAGE=debian:bookworm-slim

# Stage 1: install hydromt and dependencies
FROM $BASE_IMAGE as stage1
ARG PIXIENV
WORKDIR /home/hydromt
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH=/root/.pixi/bin:$PATH
COPY pixi.toml pixi.lock pyproject.toml README.rst ./
COPY hydromt/ ./hydromt
RUN pixi run --locked -e ${PIXIENV} install-hydromt \
  && find . -follow -delete -type f -name *.pyc

# Stage 2: copy hydromt and dependencies to final image.
# Prevents copying of caches and other temporary files.
FROM $BASE_IMAGE as base
COPY --from=stage1 /root/.pixi /root/.pixi
COPY --from=stage1 /home/hydromt /home/hydromt
COPY pixi.toml pixi.lock pyproject.toml README.rst ./
COPY data/ ./data
COPY hydromt/ ./hydromt
# Workaround: write a file that runs pixi with correct environment.
# This is needed because the argument is not passed to the entrypoint.
ENV RUNENV="${PIXIENV}"
RUN echo "pixi run --locked -e ${RUNENV} \$@" > /run_pixi.sh
ENTRYPOINT ["sh", "/run_pixi.sh"]
CMD ["hydromt","--models"]

FROM base as full
COPY examples/ ./examples
COPY tests/ ./tests

FROM base as slim

FROM base as min
