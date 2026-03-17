#!/usr/bin/env bash

REQUIRED_PYTHON="3.9"
NOTEBOOK1="experiment1_naivedml.ipynb"
NOTEBOOK2="experiment2_baselinedml.ipynb"
NOTEBOOK2="experiment3_sotadml.ipynb"
KERNEL_NAME="user"
DISPLAY_NAME="experiment-kernel"

# --- Check Python version ---
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

if [ "$(printf '%s\n' "$REQUIRED_PYTHON" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON" ]; then
  echo "Python $REQUIRED_PYTHON or later is required. Current version: $PYTHON_VERSION"
  exit 1
fi

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install notebook ipykernel jq

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

for NOTEBOOK in "$NOTEBOOK1" "$NOTEBOOK2" "$NOTEBOOK3"; do
  if [ -f "$NOTEBOOK" ]; then
    jq --arg name "$KERNEL_NAME" --arg display "$DISPLAY_NAME" \
       '.metadata.kernelspec = {
          "name": $name,
          "display_name": $display,
          "language": "python"
        }' \
       "$NOTEBOOK" > tmp_notebook.json && mv tmp_notebook.json "$NOTEBOOK"
  else
    echo "Notebook $NOTEBOOK not found!"
  fi
done


# --- Open the notebook twice ---
python -m jupyter notebook "$NOTEBOOK1" &
sleep 1
python -m jupyter notebook "$NOTEBOOK2" &
sleep 1
python -m jupyter notebook "$NOTEBOOK3" &

deactivate
