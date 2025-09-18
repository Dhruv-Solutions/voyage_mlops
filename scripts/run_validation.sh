#!/usr/bin/env bash
set -e
export DATA_DIR="${DATA_DIR:-./data}"
export EXPECT_DIR="${EXPECT_DIR:-./ml/expectations}"
export VALIDATION_OUT="${VALIDATION_OUT:-./validation}"
python ml/validate_data.py
echo "Validation report at ${VALIDATION_OUT}/validation_report.json"
