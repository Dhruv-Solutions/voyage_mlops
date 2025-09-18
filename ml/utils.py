
import pandas as pd, numpy as np
from typing import Tuple, Optional

def prep_flights(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["month"] = d["date"].dt.month
        d["dayofweek"] = d["date"].dt.weekday
    for c in ["from","to","flightType","agency"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower()
    if "time" in d.columns:
        def time_to_minutes(x):
            try:
                s = str(x)
                if s.replace(".","",1).isdigit():
                    return float(s)
                parts = s.split(":")
                if len(parts) >= 2:
                    h=int(parts[0]); m=int(parts[1]); sec=int(parts[2]) if len(parts)>2 else 0
                    return h*60 + m + sec/60.0
            except Exception:
                return np.nan
            return np.nan
        d["time_minutes"] = d["time"].apply(time_to_minutes)
    if "distance" in d.columns:
        d["distance"] = pd.to_numeric(d["distance"], errors="coerce")
    return d

def prep_hotels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["month"] = d["date"].dt.month
        d["dayofweek"] = d["date"].dt.weekday
    for c in ["name","place"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower()
    for c in ["days","price","total"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def time_based_split(df: pd.DataFrame, date_col: str, test_frac: float=0.2, split_date: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d = d.sort_values(date_col)
    d = d[~d[date_col].isna()]
    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        cutoff = d[date_col].quantile(1 - test_frac)
    train = d[d[date_col] <= cutoff]
    test  = d[d[date_col] > cutoff]
    return train, test
