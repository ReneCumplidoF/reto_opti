## Simple project automation using uv

.PHONY: help ensure-uv venv install sync upgrade freeze run kernel notebook extras-notebook check clean clean-cache

# Configurable variables
UV ?= $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)
PY ?= .venv/bin/python
SCRIPT ?= ejemplo_uso.py
ARGS ?=

help:
	@echo "Targets:"
	@echo "  ensure-uv        Install uv if missing"
	@echo "  venv             Create local virtualenv at .venv (via uv)"
	@echo "  install          Install deps from requirements.txt into .venv"
	@echo "  sync             Ensure .venv matches requirements.txt exactly"
	@echo "  upgrade          Upgrade deps from requirements.txt"
	@echo "  freeze           Write current env to requirements.lock"
	@echo "  run              Run $(SCRIPT) with uv (ARGS='...' to pass args)"
	@echo "  kernel           Install Jupyter kernel for this .venv"
	@echo "  extras-notebook  Install optional Jupyter tooling (jupyter, ipykernel)"
	@echo "  notebook         Start Jupyter Lab using .venv"
	@echo "  check            Quick import check (numpy/matplotlib/scipy)"
	@echo "  clean            Remove .venv"
	@echo "  clean-cache      Remove __pycache__ and .ipynb_checkpoints"

# Ensure uv is present (no-op if found)
ensure-uv:
	@$(UV) --version >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)

# Create venv with uv (uses system python)
venv: ensure-uv
	$(UV) venv .venv

# Install deps into .venv from requirements.txt
install: venv
	$(UV) pip install -r requirements.txt -p $(PY)

# Make the environment match requirements.txt exactly (uninstall extras)
sync: venv
	$(UV) pip sync -r requirements.txt -p $(PY)

# Upgrade packages according to requirements.txt
upgrade: venv
	$(UV) pip install -U -r requirements.txt -p $(PY)

# Freeze current environment to a lock file (informational)
freeze: venv
	$(UV) pip freeze -p $(PY) > requirements.lock
	@echo "Wrote requirements.lock"

# Run a script inside the venv without manual activate
run: venv
	$(UV) run -p $(PY) python $(SCRIPT) $(ARGS)

# Install Jupyter kernel for this venv
kernel: venv
	$(UV) run -p $(PY) python -m ipykernel install --user --name reto_opti --display-name "Python (.venv)"

# Optional extras for notebooks
extras-notebook: venv
	$(UV) pip install jupyter ipykernel -p $(PY)

# Start Jupyter Lab using the .venv (installs extras if needed)
notebook: extras-notebook
	$(UV) run -p $(PY) jupyter lab

# Quick sanity check of core deps
check: venv
	$(UV) run -p $(PY) python - <<'PY'
import numpy, matplotlib, scipy
print('OK: numpy', numpy.__version__, 'matplotlib', matplotlib.__version__, 'scipy', scipy.__version__)
PY

# Clean venv (warning: removes .venv)
clean:
	rm -rf .venv

# Remove Python caches and notebook checkpoints
clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
