#!/bin/bash

# this script brought to you curtesy of Sam ft. ChatGPT

PY_ENV_MANAGER="${PY_ENV_MANAGER:-micromamba}"

# Define the profiles array
profiles=(io extra dev test doc jupyter deprecated)

# Calculate the length of the profiles array
length=${#profiles[@]}

# Calculate the total number of subsets
num_subsets=$((2 ** length))

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

    # check if anything went wrong. if so report and abort
    if [ $? -ne 0 ]; then
        echo "an error occured when installing subset [$subset_string]"
        break
    fi

    # try to import hydromt to see if it works
    $PY_ENV_MANAGER run -n hydromt python -c "import hydromt; print(hydromt.__version__)"

    # check if anything went wrong. if so report and abort
    if [ $? -ne 0 ]; then
        echo "an error occured when importing with subset [$subset_string]"
        break
    fi
    # echo
done
