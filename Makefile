PY_ENV_MANAGER		?= micromamba
DOCKER_USER_NAME 	?= deltares
OPT_DEPS			?= ""
SPHINXBUILD   	 	 = sphinx-build
SPHINXPROJ    	 	 = hydromt
SOURCEDIR     	 	 = docs
BUILDDIR      	 	 = docs/_build

.PHONY: clean html

dev: full-environment.yml
	$(PY_ENV_MANAGER) create -f full-environment.yml -y
	$(PY_ENV_MANAGER) -n hydromt run pip install .
	$(PY_ENV_MANAGER) -n hydromt run pre-commit install

env:
	@# the subst is to make sure the is always exactly one "" around OPT_DEPS so people can
	@# specify it both as OPT_DEPS=extra,io and OPT_DEPS="extra,io"
	python3 make_env.py "$(subst ",,$(OPT_DEPS))"
	$(PY_ENV_MANAGER) create -f environment.yml -y
	$(PY_ENV_MANAGER) -n hydromt run pip install .

min-environment.yml:
	python3 make_env.py -o min-environment.yml

slim-environment.yml:
	python3 make_env.py "slim" -o slim-environment.yml

full-environment.yml:
	python3 make_env.py "full" -o full-environment.yml

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
	rm -rf $(BUILDDIR)/*
	rm -rf dist

docker-clean:
	docker images =reference="*hydromt*" -q | xargs --no-run-if-empty docker rmi -f
	docker system prune -f


html:
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

# some aliases
docs: html
doc: html
