---
name: pr
description: Create a pull request for the current session following HydroMT conventions. Use when the user wants to open a PR, says "create PR", "open a pull request", or "submit my changes".
---

# Create Pull Request (HydroMT)

Create a pull request for the current session's changes, following HydroMT's commit
conventions, PR template, and changelog requirement.

Use the **GitHub MCP server** to create the PR — do NOT use the `gh` CLI.

---

## Workflow

### 1. Run hygiene checks

```bash
pixi run lint
```

Fix any lint errors before proceeding. Do **not** skip pre-commit hooks.

### 2. Commit any uncommitted changes

If `git status --short` shows uncommitted changes, use the `/commit` skill.

**HydroMT commit convention** (inferred from `git log --oneline -20`):
- Mix of Conventional Commits (`fix:`, `feat:`, `chore:`, `docs:`) and free-form.
- Subject ≤ 72 characters.
- Reference the related issue number when one exists, e.g. `fix: ... (fixes #1234)`.
- PR number is appended automatically by GitHub merge — do not add it manually.

Always include the Co-authored-by trailer:
```
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

### 3. Push the branch

Push to `origin` if not already pushed:
```bash
git push -u origin HEAD
```

### 4. Verify the changelog

Check that `docs/changelog.rst` has been updated. If the change is user-facing
and the file was not touched, **stop and ask the user** to add a changelog entry
before proceeding.

### 5. Determine the base branch

- Default base: `main`.
- If the current branch name starts with `release/`, base onto `main` and note
  this in the PR description.

### 6. Write the PR title

Follow the repository convention:
- Conventional prefix when applicable: `fix:`, `feat:`, `docs:`, `chore:`.
- Or free-form short summary matching existing commit style.
- ≤ 72 characters, no trailing period.

### 7. Write the PR description

Fill in the HydroMT PR template sections:

```markdown
## Issue addressed

Fixes #<issue number>   <!-- omit if no issue -->

## Explanation

<What changed, why, and any design decisions made.>

## General Checklist

- [x] Updated tests or added new tests
- [x] Branch is up to date with `main`
- [x] Tests & pre-commit hooks pass
- [x] Updated documentation
- [x] Updated changelog.rst

## Data/Catalog checklist   <!-- include only if data_catalog/ files changed -->

- [x] `data/catalogs/predefined_catalogs.yml` has not been modified.
- [x] None of the old `data_catalog.yml` files have been changed
- [x] `data/changelog.rst` has been updated   <!-- if catalog files changed -->
```

Omit the **Data/Catalog checklist** section entirely if no files under
`hydromt/data_catalog/` or `data/` were changed.

Mark checklist items `[ ]` (unchecked) for anything that has **not** been done,
so the reviewer knows what is still outstanding.

### 8. Create the pull request

Use the GitHub MCP server `create_pull_request` tool with:
- `owner`: `Deltares`
- `repo`: `hydromt`
- `head`: current branch name
- `base`: target branch (default `main`)
- `title`: PR title from step 6
- `body`: PR description from step 7
- `draft`: `false` unless the user explicitly asks for a draft

After creation, report the PR URL to the user.
