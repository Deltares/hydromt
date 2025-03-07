FROM debian:bookworm-slim AS base
ARG PIXIENV
RUN apt-get update && apt-get install -y curl && apt clean && useradd deltares --uid 1000

USER deltares
WORKDIR /home/deltares

RUN curl --proto "=https" -fsSL https://pixi.sh/install.sh | bash
ENV PATH=/home/deltares/.pixi/bin:$PATH
COPY pixi.toml pixi.lock pyproject.toml README.rst ./
COPY data/ ./data
COPY hydromt/ ./hydromt
RUN pixi install --locked -e ${PIXIENV} \
  && rm -rf .cache \
  && find .pixi -type f -name "*.pyc" -delete

# Workaround: write a file that runs pixi with correct environment.
# This is needed because the argument is not passed to the entrypoint.
ENV RUNENV="${PIXIENV}"
RUN echo "pixi run --locked -e ${RUNENV} \$@" > run_pixi.sh \
  && chown deltares:deltares run_pixi.sh \
  && chmod u+x run_pixi.sh
ENTRYPOINT ["sh", "run_pixi.sh"]
CMD ["hydromt","--models"]

FROM base AS full
USER deltares
COPY examples/ ./examples
COPY tests/ ./tests

FROM base AS slim
USER deltares
COPY examples/ ./examples

FROM base AS min
USER deltares
