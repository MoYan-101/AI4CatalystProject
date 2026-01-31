#!/usr/bin/env bash
set -e
PY=""
if command -v conda >/dev/null 2>&1; then
  if [ -z "$CONDA_ENV" ]; then
    CONDA_ENV="comms310"
  fi
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    PY="conda run --no-capture-output -n $CONDA_ENV python -u"
  fi
fi
if [ -z "$PY" ]; then
  PY_BIN="./.venv/bin/python3"
  if [ -x "$PY_BIN" ]; then
    PY="$PY_BIN -u"
  else
    PY="python3 -u"
  fi
fi
if [ -z "$RUN_ID" ]; then
  BASE_RUN_ID=$(date +"%Y%m%d_%H%M%S")
else
  BASE_RUN_ID="$RUN_ID"
fi

if [ -z "$OVERFIT_ALPHA_LIST" ]; then
  echo -n "Enter overfit_penalty_alpha values (comma-separated, blank = use config): "
  read -r OVERFIT_ALPHA_LIST || true
fi

ALPHAS=()
if [ -z "$OVERFIT_ALPHA_LIST" ]; then
  ALPHAS+=("")
else
  ALPHAS_RAW=$(echo "$OVERFIT_ALPHA_LIST" | tr ',' ' ')
  for a in $ALPHAS_RAW; do
    ALPHAS+=("$a")
  done
fi

export PYTHONUNBUFFERED=1
for alpha in "${ALPHAS[@]}"; do
  if [ -n "$alpha" ]; then
    export OVERFIT_ALPHA="$alpha"
    RUN_ID="${BASE_RUN_ID}_alpha${alpha}"
  else
    unset OVERFIT_ALPHA
    RUN_ID="${BASE_RUN_ID}"
  fi
  export RUN_ID
  echo "[INFO] RUN_ID=$RUN_ID OVERFIT_ALPHA=${OVERFIT_ALPHA:-<config>}"
  $PY train.py
  $PY inference.py
  $PY visualization.py
done
