
import os, json, pandas as pd, numpy as np, mlflow, mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import prep_flights, prep_hotels, time_based_split

DATA_DIR = os.environ.get("DATA_DIR","./data")
MODEL_DIR = os.environ.get("MODEL_DIR","./models")
MLFLOW_DIR = os.environ.get("MLFLOW_DIR","./mlruns")
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.2"))
SPLIT_DATE_OVERRIDE = os.environ.get("SPLIT_DATE", "")
MAX_ROWS_FLIGHTS = int(os.environ.get("MAX_ROWS_FLIGHTS","15000"))
MAX_ROWS_HOTELS  = int(os.environ.get("MAX_ROWS_HOTELS","20000"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"file:{os.path.abspath(MLFLOW_DIR)}")

mlflow.set_experiment("voyage_analytics_prod")

def fit_regression(X_train, y_train, X_test, y_test, model):
    pipe = model
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return pipe, preds, {"rmse": rmse, "r2": r2}

def train_flights():
    df = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    df = prep_flights(df)
    df = df.dropna(subset=["price"])
    if len(df) > MAX_ROWS_FLIGHTS:
        df = df.sample(MAX_ROWS_FLIGHTS, random_state=42)
    train, test = time_based_split(df, "date", test_frac=TEST_FRAC, split_date=SPLIT_DATE_OVERRIDE or None)

    target = "price"
    feature_cols = [c for c in df.columns if c not in [target,"date","travelCode","userCode"]]
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train[target], test[target]

    num_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
    ])

    model = Pipeline([("prep", pre), ("model", Ridge(alpha=1.0))])

    with mlflow.start_run(run_name="flights_regression_time_split"):
        mlflow.log_params({
            "algo": "Ridge",
            "rows": len(df),
            "test_frac": TEST_FRAC,
            "split_date_override": SPLIT_DATE_OVERRIDE,
            "features": len(feature_cols)
        })
        model, preds, metrics = fit_regression(X_train, y_train, X_test, y_test, model)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        pred_df = X_test.copy()
        pred_df["actual_price"] = y_test
        pred_df["pred_price"] = preds
        sample_path = os.path.join(MODEL_DIR, "flight_price_predictions_sample.csv")
        pred_df.head(2000).to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path, artifact_path="predictions")

        model_path = os.path.join(MODEL_DIR, "flight_price_model_ridge.pkl")
        import joblib
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="flight_price_model")
    return True

def train_hotels():
    df = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    df = prep_hotels(df)
    df = df.dropna(subset=["price"])
    if len(df) > MAX_ROWS_HOTELS:
        df = df.sample(MAX_ROWS_HOTELS, random_state=42)
    train, test = time_based_split(df, "date", test_frac=TEST_FRAC, split_date=SPLIT_DATE_OVERRIDE or None)

    target = "price"
    feature_cols = [c for c in df.columns if c not in [target,"date","travelCode","userCode","total"]]
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train[target], test[target]

    num_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
    ])

    model = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1))])

    with mlflow.start_run(run_name="hotels_regression_time_split"):
        mlflow.log_params({
            "algo": "RandomForestRegressor",
            "rows": len(df),
            "test_frac": TEST_FRAC,
            "split_date_override": SPLIT_DATE_OVERRIDE,
            "features": len(feature_cols),
            "n_estimators": 60
        })
        model, preds, metrics = fit_regression(X_train, y_train, X_test, y_test, model)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        pred_df = X_test.copy()
        pred_df["actual_price"] = y_test
        pred_df["pred_price"] = preds
        sample_path = os.path.join(MODEL_DIR, "hotel_price_predictions_sample.csv")
        pred_df.head(2000).to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path, artifact_path="predictions")

        model_path = os.path.join(MODEL_DIR, "hotel_price_model_rf.pkl")
        import joblib
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="hotel_price_model")
    return True

if __name__ == "__main__":
    ok1 = train_flights()
    ok2 = train_hotels()
    print("Training finished:", ok1, ok2)
