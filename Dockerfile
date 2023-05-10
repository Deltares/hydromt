FROM condaforge/mambaforge:23.1.0-1 
COPY hydromt /hydromt/hydromt/
COPY envs /hydromt/envs
COPY README.rst /hydromt/
WORKDIR /hydromt
RUN groupadd -r hydromt && useradd -r -g hydromt hydromt

RUN mamba env create -f ./envs/hydromt-dev.yml 
RUN mamba install hydromt -c conda-forge -y 
USER hydromt 

ENTRYPOINT ["mamba", "run", "-n","hydromt-dev", "hydromt"]
