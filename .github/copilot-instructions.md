# Copilot Instructions

## Python Environment (pixi)

This project uses [pixi](https://pixi.sh) for environment management and the **Python Environments** VS Code extension to manage them. When VS Code opens a new terminal, the extension automatically initializes the pixi shell for the correct environment.

When running command-line commands from within Copilot, replicate this behavior by activating the pixi shell first before executing any programs:

- **PowerShell**: `Invoke-Expression (& pixi shell-hook --shell powershell | Out-String)`
- **bash/zsh**: `eval "$(pixi shell-hook --shell bash)"`

The project's environments are defined in `pyproject.toml` under `[tool.pixi.environments]`. When the Python Environments extension activates a pixi environment, it sets the `PIXI_ENVIRONMENT` environment variable. Use that variable to activate the same environment that the user has selected, falling back to `default` if it is not set.

Example (PowerShell):

```powershell
$pixi_env = if ($env:PIXI_ENVIRONMENT) { $env:PIXI_ENVIRONMENT } else { "default" }
Invoke-Expression (& pixi shell-hook --shell powershell --environment $pixi_env | Out-String)
python -m pytest
```

Example (bash/zsh):

```bash
pixi_env="${PIXI_ENVIRONMENT:-default}"
eval "$(pixi shell-hook --shell bash --environment "$pixi_env")"
python -m pytest
```

Alternatively, prefix individual commands with `pixi run` to execute them inside the selected environment without activating the shell globally:

```powershell
$pixi_env = if ($env:PIXI_ENVIRONMENT) { $env:PIXI_ENVIRONMENT } else { "default" }
pixi run --environment $pixi_env pytest
```
