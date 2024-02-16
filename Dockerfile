FROM debian:bookworm-slim as base
ARG PIXIENV
WORKDIR /home/hydromt
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH=/root/.pixi/bin:$PATH
COPY pixi.toml pixi.lock pyproject.toml README.rst ./
COPY data/ ./data
COPY examples/ ./examples
COPY hydromt/ ./hydromt
RUN pixi run -e ${PIXIENV} install-hydromt
ENTRYPOINT [ "pixi", "run" ]
CMD ["hydromt","--models"]

FROM base as min
COPY tests/ ./tests

FROM base as full
COPY tests/ ./tests

FROM base as slim
