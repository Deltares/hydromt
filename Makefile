PY_ENV_MANAGER		?= micromamba
DOCKER_USER_NAME 	?= deltares
OPT_DEPS			?= ""
SPHINXBUILD   	 	 = sphinx-build
SPHINXPROJ    	 	 = hydromt
SOURCEDIR     	 	 = docs
BUILDDIR      	 	 = docs/_build

.PHONY: clean html

html:
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

# some aliases
docs: html
doc: html

env:
	@# the subst is to make sure the is always exactly one "" around OPT_DEPS so people can
	@# specify it both as OPT_DEPS=extra,io and OPT_DEPS="extra,io"
	python3 make_env.py "$(subst ",,$(OPT_DEPS))"
	$(PY_ENV_MANAGER) create -f environment.yml -y
	$(PY_ENV_MANAGER) -n hydromt run pip install .


docker:

min-environment.yml:
	python3 make_env.py -o min-environment.yml

jupyter-environment.yml:
	python3 make_env.py "jupyter,extra" -o jupyter-environment.yml

test-environment.yml:
	python3 make_env.py "test" -o test-environment.yml

dev-environment.yml:
	python3 make_env.py "dev" -o dev-environment.yml

full-environment.yml:
	python3 make_env.py "full" -o full-environment.yml

all-environment.yml:
	python3 make_env.py "all" -o all-environment.yml

docker-min: min-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt-min:latest .
	docker tag $(DOCKER_USER_NAME)/hydromt-min:latest $(DOCKER_USER_NAME)/hydromt-min:$$(git rev-parse --short HEAD) 
	
docker-jupyter: jupyter-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt-jupyter:latest .
	docker tag $(DOCKER_USER_NAME)/hydromt-jupyter:latest $(DOCKER_USER_NAME)/hydromt-jupyter:$$(git rev-parse --short HEAD)

docker-full: full-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt-full:latest .
	docker tag $(DOCKER_USER_NAME)/hydromt-full:latest $(DOCKER_USER_NAME)/hydromt-full:$$(git rev-parse --short HEAD)

docker-test: test-environment.yml
	docker build -t $(DOCKER_USER_NAME)/hydromt-test:latest .
	docker tag $(DOCKER_USER_NAME)/hydromt-test:latest $(DOCKER_USER_NAME)/hydromt-test:$$(git rev-parse --short HEAD)

	
docker: docker-min docker-jupyter docker-full docker-test

pypi:
	git clean -xdf
	git restore -SW .
	flit build
	python -m twine check dist/*

clean:
	rm -f *environment.yml
	rm -rf $(BUILDDIR)/*
	rm -rf dist
