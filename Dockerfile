FROM condaforge/mambaforge:23.1.0-1 
COPY . /hydromt
WORKDIR /hydromt

# SHELL ["/bin/bash","-c"]
RUN mamba env create -f ./envs/hydromt-dev.yml 
RUN mamba install hydromt -c conda-forge

ENTRYPOINT ["mamba", "run", "-n","hydromt-dev", "hydromt"]
# ENTRYPOINT ["python", "-m", "hydromt"]