PY_ENV_MANAGER		?= mamba
DOCKER_USER_NAME 	?= deltares
OPT_DEPS			?= ""
ENV_NAME			?= hydromt
SPHINXBUILD   	 	 = sphinx-build
SPHINXPROJ    	 	 = hydromt
SOURCEDIR     	 	 = docs
BUILDDIR      	 	 = docs/_build

.PHONY: clean html

dev: pyproject.toml
	python make_env.py full -n hydromt-dev
	$(PY_ENV_MANAGER) env create -f environment.yml
	$(PY_ENV_MANAGER) run -n hydromt-dev pip install -e .
	$(PY_ENV_MANAGER) run -n hydromt-dev pre-commit install

env: pyproject.toml
	@# the subst is to make sure the is always exactly one "" around OPT_DEPS so people can
	@# specify it both as OPT_DEPS=extra,io and OPT_DEPS="extra,io"
	@# Note that if you use this receipt you will not get in editable install
	python make_env.py "$(subst ",,$(OPT_DEPS))" -n $(ENV_NAME)
	$(PY_ENV_MANAGER) env create -f environment.yml
	$(PY_ENV_MANAGER) run -n $(ENV_NAME) pip install .

min-environment.yml:
	python make_env.py -o min-environment.yml

slim-environment.yml:
	python make_env.py "slim" -o slim-environment.yml

full-environment.yml:
	python make_env.py "full" -o full-environment.yml

docker-min: min-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt:min --target=min .

docker-slim: slim-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt:slim --target=slim .
	docker build -t $(DOCKER_USER_NAME)/hydromt:latest --target=slim .

docker-full: full-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt:full --target=full .

docker: docker-min docker-slim docker-full

pypi:
	git clean -xdf
	git restore -SW .
	flit build
	python -m twine check dist/*

clean:
	rm -f *environment.yml
	rm -rf $(BUILDDIR)
	rm -rf dist
	rm -rf docs/_generated
	rm -rf docs/_examples

docker-clean:
	docker images =reference="*hydromt*" -q | xargs --no-run-if-empty docker rmi -f
	docker system prune -f


html:
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

# some aliases
docs: html
doc: html
