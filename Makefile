PY_ENV_MANAGER		?= micromamba
DOCKER_USER_NAME 	?= deltares
OPT_DEPS			?= full
SPHINXBUILD   	 	 = sphinx-build
SPHINXPROJ    	 	 = hydromt
SOURCEDIR     	 	 = docs
BUILDDIR      	 	 = docs/_build

.PHONY: clean html

html:
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

docs: html
doc: html

env:
	pip install tomli
	@# the subst is to make sure the is always exactly one "" around OPT_DEPS so people can
	@# specify it both as OPT_DEPS=extra,io and OPT_DEPS="extra,io"
	python3 make_env.py "$(subst ",,$(OPT_DEPS))"
	$(PY_ENV_MANAGER) create -f environment.yml -y
	$(PY_ENV_MANAGER) -n hydromt run pip install '.[$(subst ",,$(OPT_DEPS))]'


docker:
	docker build -t hydromt --target=cli .
	docker tag hydromt $(DOCKER_USER_NAME)/hydromt:latest

pypi:
	git clean -xdf
	git restore -SW .
	flit build
	python -m twine check dist/*

clean:
	rm -f environment.yml
	rm -rf $(BUILDDIR)/*
	rm -rf dist
