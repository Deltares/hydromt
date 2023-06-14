#!/bin/bash

# this script brought to you curtesy of Sam ft. ChatGPT

PY_ENV_MANAGER="${PY_ENV_MANAGER:-micromamba}"

# Define the profiles array
# profiles=(io extra dev test doc jupyter)
profiles=(io extra dev test doc jupyter deprecated)

# Calculate the length of the profiles array
length=${#profiles[@]}

# Calculate the total number of subsets
num_subsets=$((2 ** length))

# Array to store subsets with errors
err_subsets=()

# Loop over the subsets
for ((i = 0; i < num_subsets; i++)); do
    subset=()

    # Construct each subset
    for ((j = 0; j < length; j++)); do
        # Check if the j-th bit of i is set
        if (((i & (1 << j)) > 0)); then
            # Add the corresponding profile to the subset
            subset+=("${profiles[j]}")
        fi
    done

    # Convert the subset array to a comma-separated string
    subset_string=$(
        IFS=,
        echo "${subset[*]}"
    )

    # Skip iteration if subset_string is empty
    if [ -z "$subset_string" ]; then
        continue
    fi

    # install the specific subset
    make env OPT_DEPS=$subset_string

    # check if anything went wrong. if so, add the subset to err_subsets
    if [ $? -ne 0 ]; then
        err_subsets+=("$subset_string")
        continue
    fi

    # try to import hydromt to see if it works
    $PY_ENV_MANAGER run -n hydromt python -c "import hydromt"

    # check if anything went wrong. if so, add the subset to err_subsets
    if [ $? -ne 0 ]; then
        err_subsets+=("$subset_string")
        continue
    fi

    # if test is part of the install, also run the tests
    if grep -q "test" <<<$subset_string; then
        timeout 10m $PY_ENV_MANAGER run -n hydromt pytest

        # check if anything went wrong. if so, add the subset to err_subsets
        if [ $? -ne 0 ]; then
            err_subsets+=("$subset_string")
            continue
        fi
    fi

    # if doc is part of the install, also generate the docs
    if grep -q "doc" <<<$subset_string; then
        timeout 10m make html

        # check if anything went wrong. if so, add the subset to err_subsets
        if [ $? -ne 0 ]; then
            err_subsets+=("$subset_string")
            continue
        fi
    fi
done

# Print results
if [ ${#err_subsets[@]} -eq 0 ]; then
    echo "We made it!"
else
    echo "Subsets with errors:"
    for err_subset in "${err_subsets[@]}"; do
        echo "- $err_subset"
    done
fi
