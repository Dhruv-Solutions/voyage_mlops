
# Voyage Streamlit UI (Beautiful Edition)

Drop the `ui/` folder and `docker-compose.ui.yml` into your project root (where your main `docker-compose.yml` is).

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.ui.yml build ui
docker compose -f docker-compose.yml -f docker-compose.ui.yml up -d ui
# then open http://localhost:8501
```
