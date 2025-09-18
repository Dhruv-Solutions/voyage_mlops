
# Voyage UI Add-on & API Patch

This add-on provides a Streamlit UI and an optional API patch to handle "HH:MM" flight time.

## 1) Add the UI
1. Copy the `ui/` folder and `docker-compose.ui.yml` into your project root (same folder as your `docker-compose.yml`).
2. Build & run:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.ui.yml build ui
   docker compose -f docker-compose.yml -f docker-compose.ui.yml up -d ui
   # Open http://localhost:8501
   ```

## 2) Patch the API (recommended)

In `server/app_server.py`, replace the `/predict/flight_price` function with this version that converts "HH:MM" to minutes and aligns columns:

```python
@app.post("/predict/flight_price")
def predict_flight_price(inp: FlightPriceIn):
    if flight_model is None:
        raise HTTPException(status_code=503, detail="Flight model not loaded.")
    try:
        row = {
            "from": inp.from_city,
            "to": inp.to_city,
            "flightType": inp.flightType,
            "agency": inp.agency,
            "distance": inp.distance,
            "time": inp.time or "2:30",
            "date": inp.date
        }
        X = pd.DataFrame([row])
        X = _prep_flight_features(X)
        if "time" in X.columns:
            X["time"] = X["time"].apply(_time_to_minutes)
        try:
            pre = flight_model.named_steps["prep"]
            num_cols = list(pre.transformers_[0][2])
            cat_cols = list(pre.transformers_[1][2])
            expected = num_cols + cat_cols
            for c in expected:
                if c not in X.columns:
                    X[c] = None
            X = X[expected]
        except Exception:
            pass
        pred = float(flight_model.predict(X)[0])
        return {"estimated_price": pred}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail={"error": str(e), "trace": traceback.format_exc()})
```

After patching:
```bash
docker compose build voyage
docker compose up -d voyage
```
Now POST /predict/flight_price accepts `"time": "2:10"` safely.
