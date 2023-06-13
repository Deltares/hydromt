"""A simple script to generate enviroment.yml files from pyproject.toml."""

import argparse
import re
from typing import List

from tomli import load


# our quick and dirty implementation of recursive depedencies
def _parse_profile(profile_str: str, opt_deps) -> List[str]:
    if profile_str is None or profile_str in [[], "", [""], "min", ["min"]]:
        return []

    parsed = []
    queue = ["hydromt[" + x.strip() + "]" for x in profile_str.split(",")]
    while len(queue) > 0:
        dep = queue.pop(0)
        if dep == "":
            continue
        m = pat.match(dep)
        if m:
            # if we match the patern, all list elts have to be dependenciy groups
            dep_groups = m.groups(0)[0].split(",")
            unknown_dep_groups = set(dep_groups) - set(opt_deps.keys())
            if len(unknown_dep_groups) > 0:
                raise RuntimeError(f"unknown dependency group(s): {unknown_dep_groups}")
            queue.extend(dep_groups)
            continue

        if dep in opt_deps:
            queue.extend([x.strip() for x in opt_deps[dep]])
        else:
            parsed.append(dep)

    return parsed


pat = re.compile(r"\s*hydromt\[(.*)\]\s*")
parser = argparse.ArgumentParser()

parser.add_argument("profile", default="full", nargs="?")
parser.add_argument("--output", "-o", default="environment.yml")
parser.add_argument("--channel", "-c", default="conda-forge")

args = parser.parse_args()

# will sadly have to maintian this manually :(
deps_not_in_conda = ["sphinx_autosummary_accessors", "sphinx_design", "pyet", "flint"]
with open("pyproject.toml", "rb") as f:
    toml = load(f)


deps = toml["project"]["dependencies"]
opt_deps = toml["project"]["optional-dependencies"]

extra_deps = _parse_profile(args.profile, opt_deps)

deps_to_install = deps.copy()
deps_to_install.extend(extra_deps)
conda_deps = []
pip_deps = []
for dep in deps_to_install:
    if dep in deps_not_in_conda:
        pip_deps.append(dep)
    else:
        conda_deps.append(dep)

# the list(set()) is to remove duplicates
conda_deps_to_install_string = "\n- ".join(sorted(list(set(conda_deps))))
env_spec = f"""
name: hydromt

channels:
    - {args.channel}

dependencies:
- {conda_deps_to_install_string}
"""
if len(pip_deps) > 0:
    pip_deps_to_install_string = "\n  - ".join(sorted(list(set(pip_deps))))
    env_spec += f"""- pip:
  - {pip_deps_to_install_string}
"""

with open(args.output, "w") as out:
    out.write(env_spec)
