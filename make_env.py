import argparse

from tomli import load

parser = argparse.ArgumentParser()

parser.add_argument(
    "profile", choices=["full", "test", "doc", "min"], default="full", nargs="?"
)
parser.add_argument("--output", "-o", default="environment.yml")
parser.add_argument("--channel", "-c", default="conda-forge")

args = parser.parse_args()

# will sadly have to maintian this manually :(
deps_not_in_conda = ["sphinx_autosummary_accessors", "sphinx_design", "pyet"]
with open("pyproject.toml", "rb") as f:
    toml = load(f)


deps = toml["project"]["dependencies"]
extra_deps = []

if args.profile == "full":
    extra_deps.extend(toml["project"]["optional-dependencies"]["all"])
    extra_deps.extend(toml["project"]["optional-dependencies"]["test"])
    extra_deps.extend(toml["project"]["optional-dependencies"]["doc"])
elif args.profile == "test":
    extra_deps.extend(toml["project"]["optional-dependencies"]["test"])
elif args.profile == "doc":
    extra_deps.extend(toml["project"]["optional-dependencies"]["doc"])
elif args.profile == "min":
    pass
else:
    raise RuntimeWarning(f"Uknown profile: {args.profile}")

deps_to_install = deps.copy()
deps_to_install.extend(extra_deps)
conda_deps = []
pip_deps = []
for dep in deps_to_install:
    if dep in deps_not_in_conda:
        pip_deps.append(dep)
    else:
        conda_deps.append(dep)

conda_deps_to_install_string = "\n- ".join(conda_deps)
pip_deps_to_install_string = "\n  - ".join(pip_deps)
env_spec = f"""
name: hydromt

channels:
    - {args.channel}

dependencies:
- {conda_deps_to_install_string}
- pip:
  - {pip_deps_to_install_string}
"""

with open(args.output, "w") as out:
    out.write(env_spec)
