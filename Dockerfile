FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.4

RUN apt-get update && apt-get install libgdal-dev gcc lzma-dev python3-dev python3-pip -y --fix-missing

COPY hydromt /hydromt/hydromt
COPY requirements.txt /hydromt/requirements.txt
COPY README.rst /hydromt/
COPY pyproject.toml /hydromt/pyproject.toml
WORKDIR /hydromt


RUN pip install -r requirements.txt
RUN pip install .

ENTRYPOINT ["hydromt"]
