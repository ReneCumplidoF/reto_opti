# Simple project automation with uv

.PHONY: venv install run demo kernel clean

# Create venv with uv (uses system python)
venv:
	~/.local/bin/uv venv .venv

# Install deps into .venv from requirements.txt
install:
	~/.local/bin/uv pip install -r requirements.txt -p .venv/bin/python

# Run a script inside the venv without manual activate
run:
	~/.local/bin/uv run -p .venv/bin/python python ejemplo_uso.py

# Optional: install Jupyter kernel for this venv
kernel:
	. .venv/bin/activate && python -m ipykernel install --user --name reto_opti --display-name "Python (.venv)"

# Clean venv (warning: removes .venv)
clean:
	rm -rf .venv
