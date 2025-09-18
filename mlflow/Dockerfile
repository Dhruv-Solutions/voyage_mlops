FROM python:3.11-slim

WORKDIR /mlflow
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir mlflow==2.12.2
EXPOSE 5000
CMD ["bash", "-lc", "mlflow server \
  --backend-store-uri sqlite:////mlflow/db/mlflow.db \
  --default-artifact-root /mlflow/mlruns \
  --host 0.0.0.0 --port 5000"]
