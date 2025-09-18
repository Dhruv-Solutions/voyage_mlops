
# Voyage Analytics — Production Pack (with MLflow & Compose)

### Local (no Docker)
```bash
pip install -r requirements.txt
bash scripts/run_validation.sh
bash scripts/train_all.sh
bash scripts/serve.sh
# http://localhost:8080/docs
```

### Docker
```bash
docker build -t voyage-prod .
docker run --rm -p 8080:8080 voyage-prod
```

### Docker Compose (MLflow + App)
`docker-compose.yml` runs:
- **MLflow tracking** at http://localhost:5000
- **Voyage app** at http://localhost:8080

Start:
```bash
docker compose up -d mlflow
docker compose build voyage
docker compose up -d voyage
```

Train (logs to MLflow):
```bash
docker compose run --rm voyage bash scripts/train_all.sh
```

Environment:
- `TEST_FRAC`, `SPLIT_DATE`, `MAX_ROWS_FLIGHTS`, `MAX_ROWS_HOTELS`
- `MLFLOW_TRACKING_URI` (defaults to local file store if not set)
