PY_ENV_MANAGER		?= micromamba
DOCKER_USER_NAME 	?= deltares
SPHINXBUILD   	 	 = sphinx-build
SPHINXPROJ    	 	 = hydromt
SOURCEDIR     	 	 = docs
BUILDDIR      	 	 = docs/_build

.PHONY: clean html

html:
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

docs: html

env: environment.yml
	$(PY_ENV_MANAGER) create -f environment.yml -y
	$(PY_ENV_MANAGER) run -n hydromt pip install .

environment.yml: pyproject.toml make_env.py
	python3 make_env.py full

docker:
	docker build -t hydromt --target=cli .
	docker tag hydromt $(DOCKER_USER_NAME)/hydromt:latest

binder: .binder Dockerfile
	repo2docker .

pypi:
	python -m pip install --upgrade pip
	python -m pip install flit wheel twine
	git clean -xdf
	git restore -SW .
	flit build
	python -m twine check dist/*

clean:
	rm -f environment.yml
	rm -rf $(BUILDDIR)/*
	rm -rf dist
