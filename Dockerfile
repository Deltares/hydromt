FROM condaforge/mambaforge:23.1.0-1
COPY . /hydromt

WORKDIR /hydromt


SHELL ["/bin/bash", "--login", "-c"]
RUN mamba env create -f ./envs/hydromt-dev.yml && mamba init bash 
cmd ["/bin/bash"]
# RUN mamba activate hydromt-dev && pip install . 
# ENTRYPOINT ["python", "-m", "hydromt"]