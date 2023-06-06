PY_ENV_MANAGER	?= micromamba
DOCKER_USER_NAME ?= deltares
SPHINXBUILD   	 = sphinx-build
SPHINXPROJ    	 = hydromt
SOURCEDIR     	 = docs
BUILDDIR      	 = docs/_build

.PHONY: clean

html: 
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" 

env:
	python make_env.py full $(PY_VERSION)
	$(PY_ENV_MANAGER) create -f environment.yml -y
	$(PY_ENV_MANAGER) -n hydromt run pip install .

docker: 
	docker build -t hydromt-env --target=env -f Dockerfile .
	docker build -t hydromt-dev --target=dev -f Dockerfile .
	docker build -t hydromt-prod --target=prod -f Dockerfile .

tag:
	docker tag hydromt-env $(DOCKER_USER_NAME)/hydromt-env:latest
	docker tag hydromt-dev $(DOCKER_USER_NAME)/hydromt-dev:latest
	docker tag hydromt-prod $(DOCKER_USER_NAME)/hydromt-prod:latest

build: check-clean
	flit build
	python -m twine check dist/*

check-clean:
	@$(eval GIT_STATUS := $(shell git status --porcelain))
	@if [ -n "$(GIT_STATUS)" ]; then \
		echo "Aborting because working directory is not clean. Please clean and try again."; \
		exit 1; \
	fi

clean:
	rm -f environment.yml
	rm -rf $(BUILDDIR)/*
	rm -f $$(ls | grep Dockerfile | grep -vE "*-template")
	docker rmi -f $$( docker images "hydromt-*" -q )
