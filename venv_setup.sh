rm -rf ./venv
export TFF_VENV="$(pwd)/venv/bin"
"${PYTHON_PATH}" -m venv ./venv
python -m pip install --upgrade pip

brew install graphviz
