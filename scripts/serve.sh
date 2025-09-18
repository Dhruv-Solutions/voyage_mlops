#!/usr/bin/env bash
set -e
export DATA_DIR="${DATA_DIR:-./data}"
export MODEL_DIR="${MODEL_DIR:-./models}"
uvicorn server.app_server:app --host 0.0.0.0 --port 8080
