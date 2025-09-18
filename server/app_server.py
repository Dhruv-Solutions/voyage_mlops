
import os
from datetime import datetime
from typing import Optional
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import ConfigDict  # add this import

app = FastAPI(title="Voyage Analytics API", version="2.0.0")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

FLIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "flight_price_model_ridge.pkl")
HOTEL_MODEL_PATH  = os.path.join(MODEL_DIR, "hotel_price_model_rf.pkl")

USERS_PATH   = os.path.join(DATA_DIR, "users.csv")
HOTELS_PATH  = os.path.join(DATA_DIR, "hotels.csv")
FLIGHTS_PATH = os.path.join(DATA_DIR, "flights.csv")
BUNDLES_PATH = os.path.join(DATA_DIR, "trip_bundle_recommendations.csv")

flight_model = None
hotel_model = None
users_df = None
hotels_df = None
flights_df = None
bundles_df = None

def _time_to_minutes(x):
    try:
        s = str(x)
        if s.replace(".","",1).isdigit():
            return float(s)
        parts = s.split(":")
        if len(parts) >= 2:
            h = int(parts[0]); m = int(parts[1])
            sec = int(parts[2]) if len(parts) > 2 else 0
            return h * 60 + m + sec/60.0
    except Exception:
        return None
    return None

def _prep_flight_features(d: pd.DataFrame) -> pd.DataFrame:
    df = d.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.weekday
    for c in ["from","to","flightType","agency"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    if "time_minutes" not in df.columns and "time" in df.columns:
        df["time_minutes"] = df["time"].apply(_time_to_minutes)
    for c in ["travelCode","userCode"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

def _prep_hotel_features(d: pd.DataFrame) -> pd.DataFrame:
    df = d.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.weekday
    for c in ["name","place"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    for c in ["travelCode","userCode","total"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

@app.on_event("startup")
def _load():
    global flight_model, hotel_model, users_df, hotels_df, flights_df, bundles_df
    if os.path.exists(FLIGHT_MODEL_PATH):
        flight_model = joblib.load(FLIGHT_MODEL_PATH)
    if os.path.exists(HOTEL_MODEL_PATH):
        hotel_model = joblib.load(HOTEL_MODEL_PATH)
    if os.path.exists(USERS_PATH):
        users_df = pd.read_csv(USERS_PATH)
    if os.path.exists(HOTELS_PATH):
        hotels_df = pd.read_csv(HOTELS_PATH)
        for c in ["days","price","total"]:
            if c in hotels_df.columns:
                hotels_df[c] = pd.to_numeric(hotels_df[c], errors="coerce")
    if os.path.exists(FLIGHTS_PATH):
        flights_df = pd.read_csv(FLIGHTS_PATH)
        if "price" in flights_df.columns:
            flights_df["price"] = pd.to_numeric(flights_df["price"], errors="coerce")
    if os.path.exists(BUNDLES_PATH):
        try:
            bundles_df = pd.read_csv(BUNDLES_PATH)
        except Exception:
            bundles_df = None

class FlightPriceIn(BaseModel):
    from_city: str = Field(..., alias="from")
    to_city: str = Field(..., alias="to")
    flightType: str
    agency: str
    distance: Optional[float] = None
    time: Optional[str] = "2:30"
    date: Optional[datetime] = None

class HotelPriceIn(BaseModel):
    place: str
    days: int = Field(ge=1, le=60, default=3)
    date: Optional[datetime] = None
    name: Optional[str] = "placeholder hotel"

class BundleIn(BaseModel):
    from_city: Optional[str] = None
    to_city: Optional[str] = None
    place: Optional[str] = None
    userCode: Optional[str] = None
    travelCode: Optional[str] = None
    top_k: int = Field(3, ge=1, le=20)

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_flight_model": flight_model is not None,
        "has_hotel_model": hotel_model is not None,
        "users": None if users_df is None else int(len(users_df)),
        "hotels": None if hotels_df is None else int(len(hotels_df)),
        "flights": None if flights_df is None else int(len(flights_df)),
        "bundles_cached": None if bundles_df is None else int(len(bundles_df)),
    }

@app.post("/predict/flight_price")
def predict_flight_price(inp: FlightPriceIn):
    if flight_model is None:
        raise HTTPException(status_code=503, detail="Flight model not loaded.")
    try:
        # Build one-row DF from request
        row = {
            "from": inp.from_city,
            "to": inp.to_city,
            "flightType": inp.flightType,
            "agency": inp.agency,
            "distance": inp.distance,
            "time": inp.time or "2:30",   # may be "HH:MM" or numeric-like string
            "date": inp.date
        }
        X = pd.DataFrame([row])

        # Same feature engineering as training
        X = _prep_flight_features(X)

        # 🔧 Ensure 'time' is numeric (minutes) if the model expects it
        pre = flight_model.named_steps["prep"]
        num_cols = list(pre.transformers_[0][2])          # numeric cols used in training
        cat_cols = list(pre.transformers_[1][2])          # categorical cols used in training

        if "time" in X.columns:
            # Convert "2:10" → minutes; "130" → 130.0; None stays None
            X["time"] = X["time"].apply(_time_to_minutes)

        # 🔧 Add any missing columns with None so ColumnTransformer won't error
        expected = num_cols + cat_cols
        for c in expected:
            if c not in X.columns:
                X[c] = None
        X = X[expected]

        pred = float(flight_model.predict(X)[0])
        return {"estimated_price": pred}
    except Exception as e:
        # Return a helpful message instead of a blank 500
        import traceback
        raise HTTPException(status_code=400, detail={"error": str(e), "trace": traceback.format_exc()})


@app.post("/predict/hotel_price")
def predict_hotel_price(inp: HotelPriceIn):
    if hotel_model is None:
        raise HTTPException(status_code=503, detail="Hotel model not loaded.")
    row = {
        "name": inp.name or "placeholder hotel",
        "place": inp.place,
        "days": inp.days,
        "date": inp.date
    }
    X = pd.DataFrame([row])
    X = _prep_hotel_features(X)
    nightly = float(hotel_model.predict(X)[0])
    return {"nightly_price": nightly, "hotel_total": nightly * float(inp.days)}
