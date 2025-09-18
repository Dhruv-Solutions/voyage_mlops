
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY ml ./ml
COPY data ./data
COPY models ./models
COPY scripts ./scripts

EXPOSE 8080
CMD ["bash","scripts/serve.sh"]
