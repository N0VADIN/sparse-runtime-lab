python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install --no-build-isolation -e ".[dev]"
python -m pip install -e ".[dev]" --no-build-isolation
srl --help
srl export-metadata --help
srl profile --help
srl bench dense --help
srl report --help
