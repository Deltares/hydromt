FROM debian:bookworm-slim as base
ARG PIXIENV
RUN apt-get update && apt-get install -y curl

RUN useradd deltares
USER deltares
WORKDIR /home/deltares

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH=/home/deltares/.pixi/bin:$PATH
COPY --chown=deltares:deltares --chmod=744 pixi.toml pixi.lock pyproject.toml README.rst ./
COPY --chown=deltares:deltares --chmod=744 data/ ./data
COPY --chown=deltares:deltares --chmod=744 hydromt/ ./hydromt
RUN pixi run --locked -e ${PIXIENV} install-hydromt \
  && rm -rf .cache \
  && find . -type f -name "*.pyc" -delete

# Workaround: write a file that runs pixi with correct environment.
# This is needed because the argument is not passed to the entrypoint.
ENV RUNENV="${PIXIENV}"
RUN echo "pixi run --locked -e ${RUNENV} \$@" > run_pixi.sh \
  && chmod +x run_pixi.sh
ENTRYPOINT ["sh", "run_pixi.sh"]
CMD ["hydromt","--models"]

FROM base as full
USER deltares
COPY --chown=deltares:deltares --chmod=744 examples/ ./examples
COPY --chown=deltares:deltares --chmod=744 tests/ ./tests

FROM base as slim
USER deltares
COPY --chown=deltares:deltares --chmod=744 examples/ ./examples

FROM base as min
USER deltares
