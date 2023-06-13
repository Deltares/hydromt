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

# env:
# 	pip install tomli
# 	@# the subst is to make sure the is always exactly one "" around OPT_DEPS so people can
# 	@# specify it both as OPT_DEPS=extra,io and OPT_DEPS="extra,io"
# 	python3 make_env.py "$(subst ",,$(OPT_DEPS))"
# 	$(PY_ENV_MANAGER) create -f environment.yml -y
# 	$(PY_ENV_MANAGER) -n hydromt run pip install '.[$(subst ",,$(OPT_DEPS))]'


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

# The receipts below brought to you by ChatGPT
SHELL=/bin/bash

profiles := profile1 profile2 profile3 profile4

# Calculate the length of the profiles list
length := $(words $(profiles))

# Calculate the total number of subsets
num_subsets := $$(( 2 ** $(length) ))

# Target for the environment stress test
linux_env_stress_test:
	@# loop over the powerset of the available profiles
	@for ((i=0; i<$(num_subsets); i++)); do \
		subset=(); \
		for j in $$(seq 0 $$(expr $(length) - 1)); do \
			# Check if the j-th bit of i is set \
			if (( (i & (1<<j)) > 0 )); then \
				# Add the corresponding profile to the subset \
				subset+=("$$( echo "$(profiles)" | cut -d' ' -f$$(expr $$j + 1))"); \
			fi; \
		done; \
		# Convert the subset array to a comma-separated string \
		subset_string=$$(IFS=,; echo "$${subset[*]}"); \
		# Skip iteration if subset_string is empty \
		if [ -z "$${subset_string}" ]; then \
			continue; \
		fi; \
		# Output the command for the specific subset \
		echo make env "$${subset_string}"; \
	done

windows_env_stress_test:
	@# loop over the powerset of the available profiles
	@for /L %%i in (0,1,$(num_subsets)) do \
		set "subset="; \
		for /L %%j in (0,1,$(shell set /a "$(length)-1")) do \
			set "bit=$(( (%%i >> %%j) & 1 ))"; \
			if [ ! $(bit) == 0 ]; then \
				setlocal EnableDelayedExpansion; \
				set "index=%%j"; \
				for /F "tokens=!index! delims= " %%p in ("$(profiles)") do \
					set "profile=%%p"; \
					endlocal & set "subset=!subset!$(comma)!profile!"; \
				shift; \
			fi; \
		done; \
		if not defined subset (
			continue \
		); \
		setlocal EnableDelayedExpansion; \
		echo make env "!subset!"; \
		endlocal
