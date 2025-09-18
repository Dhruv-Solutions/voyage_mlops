#!/usr/bin/env bash
set -e
export DATA_DIR="${DATA_DIR:-./data}"
export MODEL_DIR="${MODEL_DIR:-./models}"
export MLFLOW_DIR="${MLFLOW_DIR:-./mlruns}"
export TEST_FRAC="${TEST_FRAC:-0.2}"
python ml/train.py
echo "Models saved to $MODEL_DIR and MLflow run logged at $MLFLOW_DIR (or at $MLFLOW_TRACKING_URI if set)"
