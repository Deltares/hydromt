"""A simple script to generate enviroment.yml files from pyproject.toml."""

import argparse
import re
from sys import version_info
from typing import List

if version_info.minor >= 11:
    from tomllib import load
else:
    from tomli import load


# our quick and dirty implementation of recursive depedencies
def _parse_profile(profile_str: str, opt_deps: dict, project_name: str) -> List[str]:
    if profile_str is None or profile_str == "":
        return []

    pat = re.compile(r"\s*" + project_name + r"\[(.*)\]\s*")
    parsed = []
    queue = [f"{project_name}[{x.strip()}]" for x in profile_str.split(",")]
    while len(queue) > 0:
        dep = queue.pop(0)
        if dep == "":
            continue
        m = pat.match(dep)
        if m:
            # if we match the patern, all list elts have to be dependenciy groups
            dep_groups = [d.strip() for d in m.groups(0)[0].split(",")]
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


parser = argparse.ArgumentParser()

parser.add_argument("profile", default="dev,test", nargs="?")
parser.add_argument("--output", "-o", default="environment.yml")
parser.add_argument("--channels", "-c", default=None)
parser.add_argument("--name", "-n", default=None)
parser.add_argument(
    "--py-version", "-p", default="3.11", choices=["3.9", "3.10", "3.11"]
)
parser.add_argument("-r", "--release", action="store_true")
args = parser.parse_args()

#
with open("pyproject.toml", "rb") as f:
    toml = load(f)
deps = toml["project"]["dependencies"]
opt_deps = toml["project"]["optional-dependencies"]
project_name = toml["project"]["name"]
# specific conda_install settings
install_config = toml["tool"].get("make_env", {})
deps_not_in_conda = install_config.get("deps_not_in_conda", [])
channels = install_config.get("channels", ["conda-forge"])
if args.channels is not None:
    channels.extend(args.channels.split(","))
    channels = list(set(channels))

# parse environment name
name = args.name
if name is None:
    name = project_name
print(f"Environment name: {name}")

# parse dependencies groups and flavours
# "min" equals no optional dependencies
if args.release:
    deps_to_install = ["hydromt"]
else:
    deps_to_install = deps.copy()
if args.profile not in ["", "min"]:
    extra_deps = _parse_profile(args.profile, opt_deps, project_name)
    deps_to_install.extend(extra_deps)

conda_deps = []
pip_deps = []
for dep in deps_to_install:
    if dep in deps_not_in_conda:
        pip_deps.append(dep)
    else:
        conda_deps.append(dep)
if args.py_version is not None:
    conda_deps.append(f"python~={args.py_version}.0")

# add pip as a conda dependency if we have pip deps
if len(pip_deps) > 0:
    conda_deps.append("pip")

# the list(set()) is to remove duplicates
conda_deps_to_install_string = "\n- ".join(sorted(list(set(conda_deps))))
channels_string = "\n- ".join(set(channels))

# create environment.yml
env_spec = f"""name: {name}

channels:
- {channels_string}

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
