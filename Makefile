PYTHON ?= .venv/bin/python
NB := $(wildcard notebooks/*.ipynb)

.PHONY: setup setup-data test kernel notebooks build study clean all

setup:
	$(PYTHON) -m pip install -e ".[dev]"

# Rebuilding the dataset needs one extra package.  Nothing else does -- the
# notebooks read the committed results/ and the tests never go near the network.
setup-data:
	$(PYTHON) -m pip install -e ".[dev,data]"

# Stream Nasdaq's seven published ITCH days and keep the panel's messages.
# About 31 GB is transferred and 90 GB inflated; the raw files are consumed from
# the socket and never written, so roughly 1.5 GB of parquet is all that lands.
# Resumable: days already extracted are skipped.
data: setup-data
	$(PYTHON) scripts/build_dataset.py

data-quick: setup-data
	$(PYTHON) scripts/build_dataset.py --quick

# Turn the extracted messages into the committed measurements in results/.
# Resumable in the same way: symbol-days already in panel.csv are skipped.
study:
	$(PYTHON) scripts/run_study.py

test:
	$(PYTHON) -m pytest

# Regenerate the notebooks from scripts/build_notebooks.py.  They are built
# rather than edited in place, which is what keeps them thin.
build:
	$(PYTHON) scripts/build_notebooks.py

# Register an ipykernel pointing at *this* environment.  Without it jupyter may
# resolve "python3" to a different interpreter that has never heard of hfx, and
# the notebooks fail on the first import for a reason unrelated to the code.
kernel:
	$(PYTHON) -m pip install -q ipykernel
	$(PYTHON) -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (hfx)"

# Re-run every notebook from a clean kernel and write the outputs back in place.
# A notebook that raises stops the build: committed outputs are reproducible.
notebooks: kernel
	JUPYTER_DATA_DIR=$(CURDIR)/.venv/share/jupyter \
	PYDEVD_DISABLE_FILE_VALIDATION=1 \
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 $(NB)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

all: test build notebooks
